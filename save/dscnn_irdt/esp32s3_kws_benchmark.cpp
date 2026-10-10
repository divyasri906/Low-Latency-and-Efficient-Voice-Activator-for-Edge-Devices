/**
 * ESP32-S3 On-Device Benchmark for DS-CNN with iRDT Feature Extractor.
 * 
 * Measures:
 * 1. Feature Extraction Latency (iRDT) with microsecond precision
 * 2. Neural Network Inference Latency (TFLM Invoke) with microsecond precision
 * 3. Total End-to-End Latency & Real-Time Factor (RTF)
 * 4. SRAM & Internal Heap Memory Usage
 * 5. On-Device Prediction Accuracy across embedded test samples (Marvin, Unknown, Silence)
 * 
 * Communication:
 * - Serial UART @ 115200 baud
 * - Formatted ASCII report & structured JSON telemetry
 */

#include <Arduino.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "dscnn_marvin_model_data.h"
#include "irdt_feature_extractor.h"
#include "test_samples_embedded.h"

// ============================================================
// Configuration Constants
// ============================================================
#define BENCHMARK_ITERATIONS 50
constexpr int kTensorArenaSize = 30 * 1024; // 30 KB
static uint8_t tensor_arena[kTensorArenaSize];

static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input_tensor = nullptr;
static TfLiteTensor* g_output_tensor = nullptr;

struct LatencyStats {
    float min_ms;
    float max_ms;
    float mean_ms;
    float median_ms;
};

// ============================================================
// Helper to compute stats from microsecond timings
// ============================================================
LatencyStats compute_stats(int64_t* times_us, int count) {
    // Bubble sort for median
    for (int i = 0; i < count - 1; ++i) {
        for (int j = 0; j < count - i - 1; ++j) {
            if (times_us[j] > times_us[j + 1]) {
                int64_t tmp = times_us[j];
                times_us[j] = times_us[j + 1];
                times_us[j + 1] = tmp;
            }
        }
    }
    int64_t sum = 0;
    for (int i = 0; i < count; ++i) {
        sum += times_us[i];
    }
    LatencyStats s;
    s.min_ms = (float)times_us[0] / 1000.0f;
    s.max_ms = (float)times_us[count - 1] / 1000.0f;
    s.mean_ms = ((float)sum / (float)count) / 1000.0f;
    s.median_ms = (float)times_us[count / 2] / 1000.0f;
    return s;
}

// ============================================================
// Run Single Inference and Return Probabilities
// ============================================================
void run_sample_inference(const int16_t* audio_pcm16, float* p_marvin, float* p_unknown, float* p_silence) {
    int8_t* in_data = g_input_tensor->data.int8;
    compute_irdt_features_int8(audio_pcm16, in_data);

    g_interpreter->Invoke();

    int8_t* out_data = g_output_tensor->data.int8;
    const float scale = g_output_tensor->params.scale;
    const int32_t zp = g_output_tensor->params.zero_point;

    *p_marvin = (out_data[0] - zp) * scale;
    *p_unknown = (out_data[1] - zp) * scale;
    *p_silence = (out_data[2] - zp) * scale;
}

// ============================================================
// Comprehensive Benchmark Routine
// ============================================================
void run_full_benchmark() {
    Serial.println("\n=======================================================================");
    Serial.println("   ESP32-S3 ON-CHIP BENCHMARK: DS-CNN INT8 WITH iRDT FEATURES");
    Serial.println("=======================================================================");

    int64_t feat_times_us[BENCHMARK_ITERATIONS];
    int64_t nn_times_us[BENCHMARK_ITERATIONS];
    int64_t e2e_times_us[BENCHMARK_ITERATIONS];

    int8_t* in_data = g_input_tensor->data.int8;

    Serial.printf("Running %d benchmark iterations...\n", BENCHMARK_ITERATIONS);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        compute_irdt_features_int8(g_test_sample_marvin, in_data);
        g_interpreter->Invoke();
    }

    // Benchmark loop
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        int64_t t0 = esp_timer_get_time();
        compute_irdt_features_int8(g_test_sample_marvin, in_data);
        int64_t t1 = esp_timer_get_time();

        g_interpreter->Invoke();
        int64_t t2 = esp_timer_get_time();

        feat_times_us[i] = (t1 - t0);
        nn_times_us[i] = (t2 - t1);
        e2e_times_us[i] = (t2 - t0);
    }

    LatencyStats feat_stats = compute_stats(feat_times_us, BENCHMARK_ITERATIONS);
    LatencyStats nn_stats = compute_stats(nn_times_us, BENCHMARK_ITERATIONS);
    LatencyStats e2e_stats = compute_stats(e2e_times_us, BENCHMARK_ITERATIONS);

    // Accuracy validation on the 3 embedded test clips
    float p_m_marvin, p_m_unk, p_m_sil;
    float p_u_marvin, p_u_unk, p_u_sil;
    float p_s_marvin, p_s_unk, p_s_sil;

    run_sample_inference(g_test_sample_marvin, &p_m_marvin, &p_m_unk, &p_m_sil);
    run_sample_inference(g_test_sample_unknown, &p_u_marvin, &p_u_unk, &p_u_sil);
    run_sample_inference(g_test_sample_silence, &p_s_marvin, &p_s_unk, &p_s_sil);

    uint32_t free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    uint32_t min_free_sram = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);

    // Print Formatted Report Table
    Serial.println("\n-----------------------------------------------------------------------");
    Serial.println("1. LATENCY BENCHMARK (over 50 sustained runs @ 240 MHz Xtensa LX7)");
    Serial.println("-----------------------------------------------------------------------");
    Serial.printf("  iRDT Feature Extraction:  Mean: %6.2f ms | Median: %6.2f ms | Min: %6.2f ms | Max: %6.2f ms\n",
                  feat_stats.mean_ms, feat_stats.median_ms, feat_stats.min_ms, feat_stats.max_ms);
    Serial.printf("  Neural Network Inference: Mean: %6.2f ms | Median: %6.2f ms | Min: %6.2f ms | Max: %6.2f ms\n",
                  nn_stats.mean_ms, nn_stats.median_ms, nn_stats.min_ms, nn_stats.max_ms);
    Serial.printf("  Total End-to-End Latency: Mean: %6.2f ms | Median: %6.2f ms | Min: %6.2f ms | Max: %6.2f ms\n",
                  e2e_stats.mean_ms, e2e_stats.median_ms, e2e_stats.min_ms, e2e_stats.max_ms);
    Serial.printf("  Real-Time Factor (RTF):   %6.4f  (Processes 1.0s audio in %.2f ms)\n",
                  e2e_stats.mean_ms / 1000.0f, e2e_stats.mean_ms);

    Serial.println("\n-----------------------------------------------------------------------");
    Serial.println("2. MEMORY UTILIZATION ON ESP32-S3");
    Serial.println("-----------------------------------------------------------------------");
    Serial.printf("  Flash Footprint (Model):  %d bytes (%.1f KB)\n", g_dscnn_model_data_len, g_dscnn_model_data_len / 1024.0f);
    Serial.printf("  Tensor Arena Size:        %d bytes (%.1f KB)\n", kTensorArenaSize, kTensorArenaSize / 1024.0f);
    Serial.printf("  Free Internal SRAM:       %u bytes (%.1f KB free)\n", free_sram, free_sram / 1024.0f);
    Serial.printf("  Min Ever Free SRAM:       %u bytes (%.1f KB)\n", min_free_sram, min_free_sram / 1024.0f);

    Serial.println("\n-----------------------------------------------------------------------");
    Serial.println("3. ON-DEVICE VALIDATION ON EMBEDDED TEST SAMPLES");
    Serial.println("-----------------------------------------------------------------------");
    Serial.printf("  [Test Clip 1: 'MARVIN']    -> P(Marvin)=%5.1f%% | P(Unknown)=%5.1f%% | P(Silence)=%5.1f%% -> %s\n",
                  p_m_marvin * 100.0f, p_m_unk * 100.0f, p_m_sil * 100.0f,
                  (p_m_marvin > 0.50f) ? "PASS (Correctly Detected)" : "FAIL");
    Serial.printf("  [Test Clip 2: 'UNKNOWN']   -> P(Marvin)=%5.1f%% | P(Unknown)=%5.1f%% | P(Silence)=%5.1f%% -> %s\n",
                  p_u_marvin * 100.0f, p_u_unk * 100.0f, p_u_sil * 100.0f,
                  (p_u_unk > 0.50f && p_u_marvin < 0.20f) ? "PASS (No False Alarm)" : "FAIL");
    Serial.printf("  [Test Clip 3: 'BACKGROUND']-> P(Marvin)=%5.1f%% | P(Unknown)=%5.1f%% | P(Silence)=%5.1f%% -> %s\n",
                  p_s_marvin * 100.0f, p_s_unk * 100.0f, p_s_sil * 100.0f,
                  (p_s_sil > 0.50f && p_s_marvin < 0.20f) ? "PASS (No False Alarm)" : "FAIL");

    Serial.println("=======================================================================\n");

    // Output JSON for host parsing
    Serial.println("---BENCHMARK_JSON_START---");
    Serial.printf("{\"board\":\"ESP32-S3\",\"cpu_freq_mhz\":%u,", getCpuFrequencyMhz());
    Serial.printf("\"feat_mean_ms\":%.2f,\"nn_mean_ms\":%.2f,\"e2e_mean_ms\":%.2f,",
                  feat_stats.mean_ms, nn_stats.mean_ms, e2e_stats.mean_ms);
    Serial.printf("\"rtf\":%.4f,\"model_flash_kb\":%.1f,\"tensor_arena_kb\":%.1f,\"free_sram_kb\":%.1f,",
                  e2e_stats.mean_ms / 1000.0f, g_dscnn_model_data_len / 1024.0f,
                  kTensorArenaSize / 1024.0f, free_sram / 1024.0f);
    Serial.printf("\"p_marvin\":%.3f,\"p_unknown\":%.3f,\"p_silence\":%.3f}\n",
                  p_m_marvin, p_u_unk, p_s_sil);
    Serial.println("---BENCHMARK_JSON_END---");
}

// ============================================================
// Setup
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(2000); // Wait for Serial Monitor connection

    g_model = tflite::GetModel(g_dscnn_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Schema mismatch: %d != %d\n", g_model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean();
    resolver.AddAdd();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();

    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors failed!");
        return;
    }

    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);

    // Automatically run benchmark on startup
    run_full_benchmark();
    Serial.println("Type 'b' and press Enter to rerun benchmark at any time.");
}

// ============================================================
// Loop (Listens for serial commands)
// ============================================================
void loop() {
    if (Serial.available()) {
        char c = Serial.read();
        if (c == 'b' || c == 'B') {
            run_full_benchmark();
        }
    }
    delay(100);
}
