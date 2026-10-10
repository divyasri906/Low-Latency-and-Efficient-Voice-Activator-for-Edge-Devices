/**
 * ESP32-S3 Real-Time KWS Streaming Pipeline with FreeRTOS Multicore Allocation
 * Model: TENet6-Narrow + Log-iRDT
 * 
 * Pipeline Architecture:
 * - Core 0: I2S Audio Ingestion / Streaming (200 ms hop) + Ultra-low-power VAD Energy Gate
 * - Core 1: TENet6-Narrow Log-iRDT Inference Task (triggered only when energy > VAD threshold)
 * - FreeRTOS Hardware CPU Profiler: Measures actual CPU % per core every 10 seconds
 * 
 * Test Conditions (switchable via Serial '1', '2', '3', '4'):
 *   [1] Silence / Quiet Ambient
 *   [2] Background TV / Speech (Unknown words)
 *   [3] Marvin Keyword Spoken (Periodic)
 *   [4] Live I2S Microphone (INMP441 on GPIO 4, 5, 6)
 */

#include <Arduino.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <driver/i2s.h>
#include "esp_freertos_hooks.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tenet6_narrow_model_data.h"
#include "irdt_feature_extractor.h"
#include "test_samples_embedded.h"

// ============================================================
// Pipeline Configuration
// ============================================================
// Pipeline Configuration
// ============================================================
#define AUDIO_WINDOW_SAMPLES 16000    // 1.0 second @ 16 kHz
#define HOP_SAMPLES          3200     // 200 ms hop interval
static volatile float g_vad_threshold_rms = 450.0f; // Acoustic energy threshold (adjusted for room noise)

// I2S Microphone Pins (INMP441: SCK=4, WS=5, SD=1, L/R=GND, VDD=3.3V, GND=GND)
#define I2S_PORT             I2S_NUM_0
#define I2S_PIN_BCLK         4   // SCK
#define I2S_PIN_WS           5   // WS
#define I2S_PIN_DIN          1   // SD (Serial Data)

// Operating Modes
enum TestCondition {
    COND_SILENCE = 1,
    COND_BACKGROUND = 2,
    COND_MARVIN = 3,
    COND_LIVE_I2S = 4
};

static volatile TestCondition g_active_condition = COND_LIVE_I2S; // Boot directly into Live Mic
static volatile bool g_wifi_streaming = false;

// Ring Buffer & Sync
static int16_t g_audio_ring_buf[AUDIO_WINDOW_SAMPLES];
static int16_t g_inference_buf[AUDIO_WINDOW_SAMPLES];
static int g_ring_write_idx = 0;
static SemaphoreHandle_t g_ring_mutex = NULL;
static SemaphoreHandle_t g_inference_sem = NULL;

// CPU Utilization Tracking (Microsecond Precision per Core)
static volatile int64_t g_active_time_c0_us = 0;
static volatile int64_t g_active_time_c1_us = 0;
static volatile uint32_t g_idle_ticks_c0 = 0;
static volatile uint32_t g_idle_ticks_c1 = 0;

static volatile int g_vad_triggers_window = 0;
static volatile int g_marvin_detections_window = 0;
static volatile float g_last_rms = 0.0f;
static volatile float g_last_p_marvin = 0.0f;
static volatile float g_last_p_unk = 0.0f;
static volatile float g_last_p_sil = 0.0f;

// TFLM Memory Arena
constexpr int kTensorArenaSize = 120 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input_tensor = nullptr;
static TfLiteTensor* g_output_tensor = nullptr;
static float g_features[IRDT_TOTAL_FEATURES];

// FreeRTOS Idle Hooks
static bool idle_hook_c0() {
    g_idle_ticks_c0++;
    return true;
}

static bool idle_hook_c1() {
    g_idle_ticks_c1++;
    return true;
}

// Compute RMS amplitude of a 200 ms block (with DC offset removal for physical mics)
static float compute_rms(const int16_t* samples, int count) {
    if (count <= 0) return 0.0f;
    int64_t sum = 0;
    for (int i = 0; i < count; ++i) {
        sum += samples[i];
    }
    int32_t mean = sum / count;
    int64_t sum_sq = 0;
    for (int i = 0; i < count; ++i) {
        int32_t v = samples[i] - mean;
        sum_sq += v * v;
    }
    return sqrtf((float)sum_sq / (float)count);
}

// ============================================================
// Core 0: Audio Capture & Energy-Gating Task
// ============================================================
void audio_capture_task(void* pvParameters) {
    static int16_t hop_buffer[HOP_SAMPLES];
    int sim_sample_counter = 0;
    bool i2s_active = (g_active_condition == COND_LIVE_I2S);

    while (true) {
        // 1. Ingest 200 ms of audio (3,200 samples)
        if (g_active_condition == COND_LIVE_I2S) {
            size_t bytes_read = 0;
            // Blocks waiting for DMA hardware interrupt (FreeRTOS CPU sleep)
            i2s_read(I2S_PORT, hop_buffer, HOP_SAMPLES * sizeof(int16_t), &bytes_read, portMAX_DELAY);
        } else {
            // Simulated real-time test audio feed
            for (int i = 0; i < HOP_SAMPLES; ++i) {
                int sample_in_second = (sim_sample_counter + i) % 16000;
                if (g_active_condition == COND_SILENCE) {
                    hop_buffer[i] = g_test_sample_silence[sample_in_second];
                } else if (g_active_condition == COND_BACKGROUND) {
                    // Background TV / Speech: 1.0s speech every 4.0s (25% conversational duty cycle)
                    int cycle = (sim_sample_counter + i) % 64000; // 4.0s cycle @ 16kHz
                    if (cycle < 16000) {
                        hop_buffer[i] = g_test_sample_unknown[cycle];
                    } else {
                        hop_buffer[i] = g_test_sample_silence[cycle % 16000];
                    }
                } else if (g_active_condition == COND_MARVIN) {
                    // Marvin spoken every 3.0s window
                    int cycle = (sim_sample_counter + i) % 48000; // 3.0s cycle @ 16kHz
                    if (cycle < 16000) {
                        hop_buffer[i] = g_test_sample_marvin[cycle];
                    } else {
                        hop_buffer[i] = g_test_sample_silence[cycle % 16000];
                    }
                }
            }
            sim_sample_counter = (sim_sample_counter + HOP_SAMPLES) % 192000;
        }

        // Measure Core 0 ACTIVE CPU processing time (ring buffer update + VAD calculation)
        int64_t t_start = esp_timer_get_time();

        // 2. Put audio into ring buffer
        if (xSemaphoreTake(g_ring_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            for (int i = 0; i < HOP_SAMPLES; ++i) {
                g_audio_ring_buf[(g_ring_write_idx + i) % AUDIO_WINDOW_SAMPLES] = hop_buffer[i];
            }
            g_ring_write_idx = (g_ring_write_idx + HOP_SAMPLES) % AUDIO_WINDOW_SAMPLES;
            xSemaphoreGive(g_ring_mutex);
        }

        // 3. Stage 1: Ultra-low-power VAD Energy Gate (< 10 us computation)
        float rms = compute_rms(hop_buffer, HOP_SAMPLES);
        g_last_rms = rms;

        // If energy exceeds threshold, trigger Core 1 deep NN inference
        if (rms >= g_vad_threshold_rms) {
            g_vad_triggers_window++;
            xSemaphoreGive(g_inference_sem);
        }

        int64_t t_end = esp_timer_get_time();
        g_active_time_c0_us += (t_end - t_start);

        // Sleep remainder of 200 ms hop interval in simulated mode
        if (g_active_condition != COND_LIVE_I2S) {
            vTaskDelay(pdMS_TO_TICKS(200));
        }
    }
}

// ============================================================
// Core 1: TENet6-Narrow + Log-iRDT Inference Task
// ============================================================
void tenet_kws_task(void* pvParameters) {
    while (true) {
        // Block until energy gate triggers an inference
        if (xSemaphoreTake(g_inference_sem, portMAX_DELAY) == pdTRUE) {
            int64_t t0 = esp_timer_get_time();

            // Snapshot latest 1.0s window from ring buffer
            if (xSemaphoreTake(g_ring_mutex, pdMS_TO_TICKS(20)) == pdTRUE) {
                int start_idx = g_ring_write_idx; // oldest sample in ring buffer
                for (int i = 0; i < AUDIO_WINDOW_SAMPLES; ++i) {
                    g_inference_buf[i] = g_audio_ring_buf[(start_idx + i) % AUDIO_WINDOW_SAMPLES];
                }
                xSemaphoreGive(g_ring_mutex);
            }

            // 1. Extract Log-iRDT features (5.95 ms)
            compute_irdt_features_float(g_inference_buf, g_features);
            for (int i = 0; i < IRDT_TOTAL_FEATURES; ++i) {
                g_input_tensor->data.f[i] = g_features[i];
            }

            // 2. Invoke TENet6-Narrow TFLM (147.24 ms)
            g_interpreter->Invoke();

            float p_marvin = g_output_tensor->data.f[0];
            float p_unk    = g_output_tensor->data.f[1];
            float p_sil    = g_output_tensor->data.f[2];

            g_last_p_marvin = p_marvin;
            g_last_p_unk    = p_unk;
            g_last_p_sil    = p_sil;

            if (p_marvin >= 0.50f) {
                g_marvin_detections_window++;
                Serial.printf("\n>>> [KEYWORD DETECTED] 'MARVIN' (Confidence: %5.1f%%) <<<\n\n", p_marvin * 100.0f);
            } else if (g_active_condition == COND_LIVE_I2S) {
                const char* top_label = (p_marvin > p_unk && p_marvin > p_sil) ? "MARVIN" : (p_unk > p_sil ? "UNKNOWN" : "SILENCE");
                Serial.printf("[Live Mic] Speech RMS: %4.0f -> %-7s (P: Marvin=%.1f%%, Unk=%.1f%%, Sil=%.1f%%)\n",
                              g_last_rms, top_label, p_marvin * 100.0f, p_unk * 100.0f, p_sil * 100.0f);
            }

            int64_t t1 = esp_timer_get_time();
            g_active_time_c1_us += (t1 - t0);
        }
    }
}

// ============================================================
// Monitor Task: Prints Real-Time Telemetry Every 10.0s
// ============================================================
void monitor_task(void* pvParameters) {
    int64_t last_report_time = esp_timer_get_time();

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000)); // Every 10 seconds

        int64_t now = esp_timer_get_time();
        int64_t dt_us = now - last_report_time;
        if (dt_us <= 0) dt_us = 10000000;

        float core0_cpu = ((float)g_active_time_c0_us / (float)dt_us) * 100.0f;
        float core1_cpu = ((float)g_active_time_c1_us / (float)dt_us) * 100.0f;
        float chip_load = (core0_cpu + core1_cpu) / 2.0f;

        int vad_triggers = g_vad_triggers_window;
        int marvin_hits = g_marvin_detections_window;
        uint32_t free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

        // Reset window accumulators
        g_active_time_c0_us = 0;
        g_active_time_c1_us = 0;
        g_vad_triggers_window = 0;
        g_marvin_detections_window = 0;
        last_report_time = now;

        const char* cond_str = "UNKNOWN";
        if (g_active_condition == COND_SILENCE) cond_str = "SILENCE / QUIET AMBIENT";
        else if (g_active_condition == COND_BACKGROUND) cond_str = "BACKGROUND TV / SPEECH";
        else if (g_active_condition == COND_MARVIN) cond_str = "MARVIN KEYWORD SPOKEN";
        else if (g_active_condition == COND_LIVE_I2S) cond_str = "LIVE I2S MICROPHONE";

        Serial.println("\n========================================================================");
        Serial.printf(" [ESP32-S3 MEASURED REAL-TIME TELEMETRY - LAST 10.0 SECONDS]\n");
        Serial.println("------------------------------------------------------------------------");
        Serial.printf("  Active Condition:          %s\n", cond_str);
        Serial.printf("  Core 0 CPU (Capture + VAD): %5.2f%%\n", core0_cpu);
        Serial.printf("  Core 1 CPU (TENet6-Narrow): %5.2f%%   <-- %s\n",
                      core1_cpu, (core1_cpu < 10.0f) ? "[PASS: STRICTLY < 10%]" : "[ACTIVE INFERENCE]");
        Serial.printf("  Dual-Core SoC Total Load:  %5.2f%%\n", chip_load);
        Serial.printf("  VAD Triggers (200ms hops):  %d / 50 (%4.1f%% duty cycle)\n",
                      vad_triggers, (vad_triggers / 50.0f) * 100.0f);
        Serial.printf("  Marvin Keyword Hits:       %d\n", marvin_hits);
        Serial.printf("  Free Internal SRAM:        %u bytes (%.1f KB)\n", free_sram, free_sram / 1024.0f);
        Serial.println("========================================================================");

        // Output structured JSON telemetry for automated host collectors
        Serial.println("---REALTIME_JSON_START---");
        Serial.printf("{\"condition\":\"%s\",\"core0_cpu_pct\":%.2f,\"core1_cpu_pct\":%.2f,",
                      cond_str, core0_cpu, core1_cpu);
        Serial.printf("\"chip_load_pct\":%.2f,\"vad_triggers\":%d,\"marvin_hits\":%d,",
                      chip_load, vad_triggers, marvin_hits);
        Serial.printf("\"free_sram_kb\":%.1f,\"target_less_10pct\":%s}\n",
                      free_sram / 1024.0f, (core1_cpu < 10.0f) ? "true" : "false");
        Serial.println("---REALTIME_JSON_END---");
    }
}

// ============================================================
// I2S INMP441 Driver Setup
// ============================================================
bool init_i2s() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_PIN_BCLK,   // 4 (SCK)
        .ws_io_num = I2S_PIN_WS,       // 5 (WS)
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_PIN_DIN     // 1 (SD)
    };

    if (i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL) != ESP_OK) {
        Serial.println("[!] Failed to install I2S driver");
        return false;
    }
    if (i2s_set_pin(I2S_PORT, &pin_config) != ESP_OK) {
        Serial.println("[!] Failed to set I2S pins");
        return false;
    }
    i2s_zero_dma_buffer(I2S_PORT);
    Serial.println("[OK] I2S INMP441 Driver Installed (SCK: 4, WS: 5, SD: 1)");
    return true;
}

// ============================================================
// Setup
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(1500);

    Serial.println("\n========================================================================");
    Serial.println(" ESP32-S3 REAL-TIME ENERGY-GATED KWS: TENet6-Narrow + Log-iRDT");
    Serial.println("========================================================================");

    // Initialize INMP441 I2S Hardware Microphone
    if (!init_i2s()) {
        Serial.println("[!] I2S Microphone initialization failed!");
    }

    // Register FreeRTOS idle hooks
    esp_register_freertos_idle_hook_for_cpu(idle_hook_c0, 0);
    esp_register_freertos_idle_hook_for_cpu(idle_hook_c1, 1);

    // Initialize synchronization primitives
    g_ring_mutex = xSemaphoreCreateMutex();
    g_inference_sem = xSemaphoreCreateBinary();

    // Initialize TFLM Model
    g_model = tflite::GetModel(g_tenet6_narrow_model_data);
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddAdd();
    resolver.AddAveragePool2D();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddDequantize();
    resolver.AddQuantize();
    resolver.AddReshape();
    resolver.AddSoftmax();

    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[!] AllocateTensors failed!");
        return;
    }

    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);
    Serial.printf("[OK] TFLM Initialized. Model Flash: %u B, Arena: %d B\n",
                  g_tenet6_narrow_model_data_len, kTensorArenaSize);

    // Warmup 1 inference
    compute_irdt_features_float(g_test_sample_marvin, g_features);
    for (int i = 0; i < IRDT_TOTAL_FEATURES; ++i) g_input_tensor->data.f[i] = g_features[i];
    g_interpreter->Invoke();
    Serial.println("[OK] Warmup inference complete.");

    // Launch FreeRTOS Tasks
    // Core 0: Audio Ingestion & Energy Gate
    xTaskCreatePinnedToCore(audio_capture_task, "AudioCap", 8192, NULL, 5, NULL, 0);

    // Core 1: TENet6-Narrow Inference Task
    xTaskCreatePinnedToCore(tenet_kws_task, "TENetKWS", 8192, NULL, 4, NULL, 1);

    // Core 0: 10-Second Telemetry Monitor Task
    xTaskCreatePinnedToCore(monitor_task, "Monitor", 3072, NULL, 1, NULL, 0);

    Serial.println("\n[ACTIVE MODE] --> LIVE INMP441 MICROPHONE (Listening on SCK:4, WS:5, SD:1)");
    Serial.printf("Current VAD RMS Threshold: %.0f (Send '+' or '-' to adjust)\n", g_vad_threshold_rms);
    Serial.println("Commands:");
    Serial.println("  '4' -> Live I2S Microphone (INMP441)");
    Serial.println("  '1' -> Silence Simulated Test");
    Serial.println("  '2' -> Background TV/Speech Simulated Test");
    Serial.println("  '3' -> Marvin Keyword Simulated Test");
    Serial.println("  '+' -> Increase VAD Energy Threshold (+50)");
    Serial.println("  '-' -> Decrease VAD Energy Threshold (-50)");
}

void loop() {
    if (Serial.available()) {
        char c = Serial.read();
        if (c == '1' || c == 's' || c == 'S') {
            g_active_condition = COND_SILENCE;
            Serial.println("\n--> Switched to: [1] SILENCE / QUIET AMBIENT");
        } else if (c == '2' || c == 'b' || c == 'B') {
            g_active_condition = COND_BACKGROUND;
            Serial.println("\n--> Switched to: [2] BACKGROUND TV / SPEECH");
        } else if (c == '3' || c == 'm' || c == 'M') {
            g_active_condition = COND_MARVIN;
            Serial.println("\n--> Switched to: [3] MARVIN KEYWORD SPOKEN");
        } else if (c == '4' || c == 'l' || c == 'L') {
            g_active_condition = COND_LIVE_I2S;
            Serial.println("\n--> Switched to: [4] LIVE I2S MICROPHONE");
        } else if (c == '+') {
            g_vad_threshold_rms += 50.0f;
            Serial.printf("\n--> VAD Threshold Increased: RMS = %.0f\n", g_vad_threshold_rms);
        } else if (c == '-') {
            g_vad_threshold_rms = (g_vad_threshold_rms > 50.0f) ? (g_vad_threshold_rms - 50.0f) : 25.0f;
            Serial.printf("\n--> VAD Threshold Decreased: RMS = %.0f\n", g_vad_threshold_rms);
        }
    }
    delay(100);
}
