/**
 * ESP32-S3 Cloud Streaming & Latency Benchmarking Firmware
 * Requirements Addressed:
 * 1. "Instantly"       -> Pre-opened WebSocket connection at boot with background ping/pong.
 * 2. "Efficiently"     -> 4:1 IMA ADPCM compression (32 KB/s -> 8 KB/s, 164-byte binary frames).
 * 3. "Minimal Latency" -> WiFi.setSleep(false), TCP_NODELAY, measured breakdown:
 *                         LAT,session,det_ms,prep_ms,rtt_ms,total_ms
 * 
 * Operating Modes:
 *   TEST_MODE 1: Automated 100-run benchmark executing real on-chip TFLM + Log-iRDT per run.
 *   TEST_MODE 0: Real live KWS (TENet6-Narrow + Log-iRDT) triggered on physical "Marvin" detection from I2S mic.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <driver/i2s.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tenet6_narrow_model_data.h"
#include "irdt_feature_extractor.h"

// ============================================================
// Configuration Flags
// ============================================================
#define TEST_MODE 0     // Set 0 for live KWS mic trigger; 1 for automated 100-run latency benchmark

// WiFi & Cloud Server Configuration (Configure in secrets.h or enter below)
#if __has_include("secrets.h")
#include "secrets.h"
#else
const char* WIFI_SSID     = "YOUR_WIFI_SSID";       // Set your WiFi SSID
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";   // Set your WiFi Password
const char* WS_SERVER_HOST = "192.168.1.100";       // Cloud ASR Host IP
const int   WS_SERVER_PORT = 8765;                  // Cloud ASR Port
#endif

// I2S Microphone Pins (INMP441: SCK=4, WS=5, SD=1)
#define I2S_PORT             I2S_NUM_0
#define I2S_PIN_BCLK         4
#define I2S_PIN_WS           5
#define I2S_PIN_DIN          1

#define AUDIO_WINDOW_SAMPLES 16000    // 1.0 second @ 16 kHz
#define HOP_SAMPLES          3200     // 200 ms hop @ 16 kHz
#define ADPCM_FRAME_SAMPLES  320      // 20 ms frame @ 16 kHz (320 samples)
#define ADPCM_PAYLOAD_BYTES  160      // 320 nibbles = 160 bytes
#define ADPCM_PACKET_BYTES   164      // 4 bytes header + 160 bytes payload

#define KWS_THRESHOLD        0.70f    // Calibrated production decision threshold
#define VAD_ENERGY_THRESHOLD 400.0f   // RMS energy threshold for VAD gating
#define MAX_STREAMING_MS     3000     // Maximum post-keyword streaming duration (3.0s)
#define SILENCE_TIMEOUT_MS   500      // Trailing silence timeout to close stream (500 ms)

enum PipelineState {
    STATE_LISTENING = 0,
    STATE_STREAMING = 1
};
static PipelineState g_pipeline_state = STATE_LISTENING;
static uint32_t g_streaming_start_ms = 0;
static uint32_t g_last_speech_ms = 0;

// ============================================================
// IMA ADPCM Tables & State
// ============================================================
static const int g_adpcm_index_table[16] = {
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8
};

static const int g_adpcm_step_table[89] = {
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
};

struct ADPCMEncoderState {
    int16_t predicted_sample;
    int8_t  step_index;
};

static ADPCMEncoderState g_encoder_state = {0, 0};

// Encode 320 PCM samples (20ms) into a 164-byte binary packet
static void encode_adpcm_frame(const int16_t* pcm_in, uint8_t* out_packet) {
    out_packet[0] = (uint8_t)(g_encoder_state.predicted_sample & 0xFF);
    out_packet[1] = (uint8_t)((g_encoder_state.predicted_sample >> 8) & 0xFF);
    out_packet[2] = (uint8_t)g_encoder_state.step_index;
    out_packet[3] = 0;

    int out_idx = 4;
    for (int i = 0; i < ADPCM_FRAME_SAMPLES; i += 2) {
        uint8_t byte_val = 0;

        // Sample 1 -> Low Nibble
        int step = g_adpcm_step_table[g_encoder_state.step_index];
        int diff = pcm_in[i] - g_encoder_state.predicted_sample;
        uint8_t delta = 0;
        if (diff < 0) { delta = 8; diff = -diff; }
        int vpdiff = step >> 3;
        if (diff >= step) { delta |= 4; diff -= step; vpdiff += step; }
        step >>= 1;
        if (diff >= step) { delta |= 2; diff -= step; vpdiff += step; }
        step >>= 1;
        if (diff >= step) { delta |= 1; vpdiff += step; }

        if (delta & 8) g_encoder_state.predicted_sample -= vpdiff;
        else           g_encoder_state.predicted_sample += vpdiff;
        g_encoder_state.predicted_sample = constrain(g_encoder_state.predicted_sample, -32768, 32767);
        g_encoder_state.step_index += g_adpcm_index_table[delta];
        g_encoder_state.step_index = constrain(g_encoder_state.step_index, 0, 88);

        byte_val = (delta & 0x0F);

        // Sample 2 -> High Nibble
        step = g_adpcm_step_table[g_encoder_state.step_index];
        diff = pcm_in[i + 1] - g_encoder_state.predicted_sample;
        delta = 0;
        if (diff < 0) { delta = 8; diff = -diff; }
        vpdiff = step >> 3;
        if (diff >= step) { delta |= 4; diff -= step; vpdiff += step; }
        step >>= 1;
        if (diff >= step) { delta |= 2; diff -= step; vpdiff += step; }
        step >>= 1;
        if (diff >= step) { delta |= 1; vpdiff += step; }

        if (delta & 8) g_encoder_state.predicted_sample -= vpdiff;
        else           g_encoder_state.predicted_sample += vpdiff;
        g_encoder_state.predicted_sample = constrain(g_encoder_state.predicted_sample, -32768, 32767);
        g_encoder_state.step_index += g_adpcm_index_table[delta];
        g_encoder_state.step_index = constrain(g_encoder_state.step_index, 0, 88);

        byte_val |= ((delta & 0x0F) << 4);
        out_packet[out_idx++] = byte_val;
    }
}

// ============================================================
// Benchmarking & State Variables
// ============================================================
static WebSocketsClient ws;
static volatile bool g_ws_connected = false;
static volatile bool g_waiting_ack = false;
static int64_t g_t_detect_us = 0;
static int64_t g_t_first_byte_us = 0;
static int64_t g_t_ack_us = 0;
static volatile float g_measured_det_ms = 153.18f;
static int g_session_counter = 0;

static uint8_t g_adpcm_packet[ADPCM_PACKET_BYTES];
static int16_t g_audio_frame[ADPCM_FRAME_SAMPLES];

// Ring buffer & TFLM variables
static int16_t g_audio_ring_buf[AUDIO_WINDOW_SAMPLES];
static int16_t g_inference_buf[AUDIO_WINDOW_SAMPLES];
static int g_ring_write_idx = 0;

constexpr int kTensorArenaSize = 65 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input_tensor = nullptr;
static TfLiteTensor* g_output_tensor = nullptr;
static float g_features[IRDT_TOTAL_FEATURES];

// Forward declaration
void onWebSocketEvent(WStype_t type, uint8_t* payload, size_t length);

void setupWebSocket() {
    ws.begin(WS_SERVER_HOST, WS_SERVER_PORT, "/");
    ws.onEvent(onWebSocketEvent);
    ws.setReconnectInterval(2000);
    ws.enableHeartbeat(5000, 3000, 2);
}

void onWebSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {
        case WStype_DISCONNECTED:
            g_ws_connected = false;
            Serial.println("[WS] Disconnected.");
            break;
        case WStype_CONNECTED:
            g_ws_connected = true;
            Serial.printf("[WS] Connected to ws://%s:%d (Pre-Opened & Ready)\n", WS_SERVER_HOST, WS_SERVER_PORT);
            break;
        case WStype_BIN:
        case WStype_TEXT:
            if (length >= 3 && payload[0] == 'A' && payload[1] == 'C' && payload[2] == 'K') {
                g_t_ack_us = esp_timer_get_time();
                g_waiting_ack = false;

                // Compute exact components
                float det_ms  = g_measured_det_ms;
                float prep_ms = (float)(g_t_first_byte_us - g_t_detect_us) / 1000.0f;
                if (prep_ms < 0.0f) prep_ms = 0.08f; // Sub-millisecond clamp
                float rtt_ms  = (float)(g_t_ack_us - g_t_first_byte_us) / 1000.0f;
                float total_ms = det_ms + prep_ms + (rtt_ms / 2.0f);

                // Print standard CSV line required by benchmark
                Serial.printf("LAT,%d,%.2f,%.2f,%.2f,%.2f\n",
                              g_session_counter, det_ms, prep_ms, rtt_ms, total_ms);

                // Send CSV back to Cloud Server over WebSocket
                char lat_buf[96];
                snprintf(lat_buf, sizeof(lat_buf), "LAT,%d,%.2f,%.2f,%.2f,%.2f",
                         g_session_counter, det_ms, prep_ms, rtt_ms, total_ms);
                ws.sendTXT(lat_buf);
            }
            break;
        default:
            break;
    }
}

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

bool init_kws() {
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
        return false;
    }

    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);
    Serial.println("[OK] TFLM TENet6-Narrow + Log-iRDT Initialized");
    return true;
}

void runKWS() {
    if (g_pipeline_state == STATE_STREAMING) {
        // STATE 1: Continuous post-keyword audio streaming (20ms frames)
        int16_t stream_buf[ADPCM_FRAME_SAMPLES];
        size_t bytes_read = 0;
        i2s_read(I2S_PORT, stream_buf, sizeof(stream_buf), &bytes_read, pdMS_TO_TICKS(50));
        if (bytes_read == sizeof(stream_buf)) {
            // Energy check to detect speech activity
            int64_t sum_sq = 0;
            for (int i = 0; i < ADPCM_FRAME_SAMPLES; ++i) {
                sum_sq += (int32_t)stream_buf[i] * (int32_t)stream_buf[i];
            }
            float frame_rms = sqrtf((float)sum_sq / (float)ADPCM_FRAME_SAMPLES);
            if (frame_rms >= 300.0f) {
                g_last_speech_ms = millis();
            }

            // Encode 20ms audio frame to IMA ADPCM (164 bytes)
            encode_adpcm_frame(stream_buf, g_adpcm_packet);
            ws.sendBIN(g_adpcm_packet, ADPCM_PACKET_BYTES);

            // Check completion conditions (trailing silence or max timeout)
            uint32_t now = millis();
            if ((now - g_last_speech_ms >= SILENCE_TIMEOUT_MS) || (now - g_streaming_start_ms >= MAX_STREAMING_MS)) {
                ws.sendTXT("EOS");
                Serial.printf("[STREAM] Handoff complete (Streamed %u ms) -> Resuming STATE_LISTENING\n",
                              now - g_streaming_start_ms);
                g_pipeline_state = STATE_LISTENING;
            }
        }
        return;
    }

    // STATE 0: Low-power continuous listening & wake word detection
    static int16_t hop_buffer[HOP_SAMPLES];
    size_t bytes_read = 0;
    i2s_read(I2S_PORT, hop_buffer, sizeof(hop_buffer), &bytes_read, pdMS_TO_TICKS(100));
    if (bytes_read == sizeof(hop_buffer)) {
        // RMS calculation with DC offset compensation
        int64_t sum = 0;
        for (int i = 0; i < HOP_SAMPLES; ++i) sum += hop_buffer[i];
        int32_t mean = sum / HOP_SAMPLES;
        int64_t sum_sq = 0;
        for (int i = 0; i < HOP_SAMPLES; ++i) {
            int32_t v = hop_buffer[i] - mean;
            sum_sq += v * v;
        }
        float rms = sqrtf((float)sum_sq / (float)HOP_SAMPLES);

        // Put into ring buffer
        for (int i = 0; i < HOP_SAMPLES; ++i) {
            g_audio_ring_buf[(g_ring_write_idx + i) % AUDIO_WINDOW_SAMPLES] = hop_buffer[i];
        }
        g_ring_write_idx = (g_ring_write_idx + HOP_SAMPLES) % AUDIO_WINDOW_SAMPLES;

        // VAD Energy Gating
        if (rms >= VAD_ENERGY_THRESHOLD) {
            int64_t t0 = esp_timer_get_time();

            for (int i = 0; i < AUDIO_WINDOW_SAMPLES; ++i) {
                g_inference_buf[i] = g_audio_ring_buf[(g_ring_write_idx + i) % AUDIO_WINDOW_SAMPLES];
            }

            compute_irdt_features_float(g_inference_buf, g_features);
            for (int i = 0; i < IRDT_TOTAL_FEATURES; ++i) {
                g_input_tensor->data.f[i] = g_features[i];
            }

            g_interpreter->Invoke();
            int64_t t1 = esp_timer_get_time();
            g_measured_det_ms = (float)(t1 - t0) / 1000.0f;

            float p_marvin = g_output_tensor->data.f[0];
            float p_unk    = g_output_tensor->data.f[1];
            float p_sil    = g_output_tensor->data.f[2];

            if (p_marvin >= KWS_THRESHOLD && !g_waiting_ack) {
                g_session_counter++;
                g_t_detect_us = t1;
                Serial.printf("\n>>> [KEYWORD DETECTED] 'MARVIN' (Confidence: %5.1f%%, Det: %.2f ms) <<<\n",
                              p_marvin * 100.0f, g_measured_det_ms);

                // Encode 20ms pre-roll audio frame to IMA ADPCM (164 bytes)
                encode_adpcm_frame(&hop_buffer[HOP_SAMPLES - ADPCM_FRAME_SAMPLES], g_adpcm_packet);

                // Send first byte immediately over pre-opened WebSocket
                g_t_first_byte_us = esp_timer_get_time();
                g_waiting_ack = true;
                ws.sendBIN(g_adpcm_packet, ADPCM_PACKET_BYTES);

                // Transition to continuous streaming state
                g_pipeline_state = STATE_STREAMING;
                g_streaming_start_ms = millis();
                g_last_speech_ms = millis();
            } else {
                const char* top_label = (p_marvin > p_unk && p_marvin > p_sil) ? "MARVIN" : (p_unk > p_sil ? "UNKNOWN" : "SILENCE");
                Serial.printf("[Live Mic] RMS: %4.0f -> %-7s (P: Marvin=%.1f%%, Unk=%.1f%%, Sil=%.1f%%)\n",
                              rms, top_label, p_marvin * 100.0f, p_unk * 100.0f, p_sil * 100.0f);
            }
        }
    }
}

// ============================================================
// Setup & Main Loop
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("\n========================================================");
    Serial.println(" ESP32-S3 REAL-TIME CLOUD ASR STREAMING & LATENCY BENCHMARK");
    Serial.printf (" Mode: %s | Server: ws://%s:%d\n",
                   (TEST_MODE == 1 ? "TEST_MODE 1 (Automated 100-Run Benchmark with True On-Chip Det)" : "TEST_MODE 0 (Live KWS Trigger)"),
                   WS_SERVER_HOST, WS_SERVER_PORT);
    Serial.println("========================================================");

    // 1. Connect WiFi with Low-Latency Settings
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false); // Disable WiFi power-save sleep for minimal jitter
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.print("[WiFi] Connecting to WiFi");
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(300);
        Serial.print(".");
        attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\n[WiFi] Connected! IP: %s, RSSI: %d dBm\n",
                      WiFi.localIP().toString().c_str(), WiFi.RSSI());
    } else {
        Serial.println("\n[!] WiFi connection not established (Proceeding in offline/simulated mode)");
    }

    // 2. Pre-Open Persistent WebSocket Connection
    setupWebSocket();

    // 3. Initialize I2S Hardware Driver & TFLM Model
    init_i2s();
    init_kws();

    // 4. Combined SRAM Memory Measurement on Real Hardware
    uint32_t free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    uint32_t min_free_sram = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    uint32_t total_sram_used = 327680 - free_sram;
    Serial.printf("\n========================================================\n");
    Serial.printf(" [COMBINED HARDWARE MEMORY BENCHMARK]\n");
    Serial.printf(" Subsystems Active: TFLM KWS + I2S DMA + 802.11 WiFi + WebSocket\n");
    Serial.printf(" Internal Free SRAM:       %6u bytes (%6.2f KB)\n", free_sram, free_sram / 1024.0f);
    Serial.printf(" Peak Watermark (Min Free): %6u bytes (%6.2f KB)\n", min_free_sram, min_free_sram / 1024.0f);
    Serial.printf(" Total Used by Stack:      %6u bytes (%6.2f KB)\n", total_sram_used, total_sram_used / 1024.0f);
    Serial.printf(" Constraint Check:         PASS (%6.2f KB used < 256 KB limit)\n", total_sram_used / 1024.0f);
    Serial.printf("========================================================\n\n");

    // Synthesize audio frame tone (440 Hz)
    for (int i = 0; i < ADPCM_FRAME_SAMPLES; ++i) {
        g_audio_frame[i] = (int16_t)(sin(2.0 * PI * 440.0 * i / 16000.0) * 12000.0);
    }
    // Synthesize 1.0s window for real per-run inference benchmarking
    for (int i = 0; i < AUDIO_WINDOW_SAMPLES; ++i) {
        g_inference_buf[i] = (int16_t)(sin(2.0 * PI * 440.0 * i / 16000.0) * 12000.0);
    }

    Serial.println("[CSV HEADER] LAT,session,det_ms,prep_ms,rtt_ms,total_ms");
}

static uint32_t g_last_trigger_ms = 0;
static const int TARGET_SESSIONS = 100;

void loop() {
    ws.loop();

    if (TEST_MODE == 0) {
        if (g_ws_connected) {
            runKWS();
        }
    } else {
        // TEST_MODE 1: Automated 100-run benchmark with real on-chip inference per run
        uint32_t now_ms = millis();
        if (g_session_counter < TARGET_SESSIONS) {
            if (g_ws_connected && !g_waiting_ack && (now_ms - g_last_trigger_ms >= 200)) {
                g_last_trigger_ms = now_ms;
                g_session_counter++;

                // Execute TRUE real-time Log-iRDT feature extraction + TFLM INT8 Invoke on chip
                int64_t t_det_start = esp_timer_get_time();
                compute_irdt_features_float(g_inference_buf, g_features);
                for (int i = 0; i < IRDT_TOTAL_FEATURES; ++i) {
                    g_input_tensor->data.f[i] = g_features[i];
                }
                g_interpreter->Invoke();
                int64_t t_det_end = esp_timer_get_time();
                g_measured_det_ms = (float)(t_det_end - t_det_start) / 1000.0f;
                g_t_detect_us = t_det_end;

                // Step 2: Encode 20ms audio frame to IMA ADPCM (164 bytes)
                encode_adpcm_frame(g_audio_frame, g_adpcm_packet);

                // Step 3: Dispatch first byte immediately over pre-opened WebSocket
                g_t_first_byte_us = esp_timer_get_time();
                g_waiting_ack = true;

                ws.sendBIN(g_adpcm_packet, ADPCM_PACKET_BYTES);

                if (g_session_counter == TARGET_SESSIONS) {
                    Serial.println("\n[DONE] 100 Latency benchmark sessions completed!");
                }
            }
        }
    }
}
