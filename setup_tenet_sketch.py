"""
Setup, packaging, and header generation for TENet6-Narrow + iRDT on ESP32-S3.
"""

import os
import shutil
import wave
import numpy as np

# Create output directories
os.makedirs("save/tenet_narrow", exist_ok=True)
package_dir = r"C:\Users\sinth\Downloads\ESP32S3_TENet_Arduino_Package\ESP32S3_TENet_Benchmark"
os.makedirs(package_dir, exist_ok=True)
workspace_package = "save/tenet_narrow/ESP32S3_TENet_Benchmark"
os.makedirs(workspace_package, exist_ok=True)

# 1. Convert best_marvin_model_log_int8.tflite to C header
tflite_path = "best_marvin_model_log_int8.tflite"
with open(tflite_path, "rb") as f:
    model_bytes = f.read()

model_h_content = f"""// Auto-generated from {tflite_path}
// Model size: {len(model_bytes)} bytes
#ifndef TENET6_NARROW_MODEL_DATA_H_
#define TENET6_NARROW_MODEL_DATA_H_

#include <stdint.h>

const unsigned int g_tenet6_narrow_model_data_len = {len(model_bytes)};
alignas(16) const unsigned char g_tenet6_narrow_model_data[] = {{
"""

hex_lines = []
for i in range(0, len(model_bytes), 12):
    chunk = model_bytes[i:i+12]
    hex_lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")

model_h_content += "\n".join(hex_lines) + "\n};\n\n#endif  // TENET6_NARROW_MODEL_DATA_H_\n"

with open("save/tenet_narrow/tenet6_narrow_model_data.h", "w") as f:
    f.write(model_h_content)
with open(os.path.join(package_dir, "tenet6_narrow_model_data.h"), "w") as f:
    f.write(model_h_content)
with open(os.path.join(workspace_package, "tenet6_narrow_model_data.h"), "w") as f:
    f.write(model_h_content)

print(f"--> Exported tenet6_narrow_model_data.h ({len(model_bytes)} bytes)")

# 2. Generate C++ Log-iRDT Feature Extractor Engine
irdt_h_content = """#ifndef IRDT_FEATURE_EXTRACTOR_TENET_H
#define IRDT_FEATURE_EXTRACTOR_TENET_H

#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IRDT_SAMPLE_RATE     16000
#define IRDT_DESIRED_SAMPLES 16000
#define IRDT_W               64
#define IRDT_WIN_PER_SEGM    5
#define IRDT_NUM_CHANNELS    6
#define IRDT_NUM_WINDOWS     (IRDT_DESIRED_SAMPLES / IRDT_W)           // 250
#define IRDT_NUM_SEGMENTS    (IRDT_NUM_WINDOWS / IRDT_WIN_PER_SEGM)    // 50
#define IRDT_TOTAL_FEATURES  (IRDT_NUM_SEGMENTS * IRDT_NUM_CHANNELS)   // 300

static const int g_irdt_channels[IRDT_NUM_CHANNELS] = {1, 2, 4, 8, 16, 32};

/**
 * Computes Log-iRDT features directly from 16-bit PCM audio samples.
 * Produces float32 output tensor of shape [1, 50, 6] for TENet6-Narrow.
 * Formula: log2(1.0 + (sum / 32768.0))
 * 
 * @param audio_pcm16  Pointer to 16,000 int16_t raw audio samples.
 * @param out_float    Pointer to output float array of size 300 (50 x 6).
 */
static inline void compute_irdt_features_float(const int16_t* audio_pcm16, float* out_float) {
    static uint32_t spectrum[IRDT_NUM_WINDOWS][IRDT_NUM_CHANNELS];
    
    // Clear spectrum buffer
    for (int w = 0; w < IRDT_NUM_WINDOWS; ++w) {
        for (int k = 0; k < IRDT_NUM_CHANNELS; ++k) {
            spectrum[w][k] = 0;
        }
    }

    // Step 1: Integer Laplacian across raw windows
    const int limit = IRDT_W / 4; // 16
    for (int win = 0; win < IRDT_NUM_WINDOWS; ++win) {
        const int base = win * IRDT_W;
        for (int t = limit; t < (IRDT_W - limit); ++t) {
            const int n = base + t;
            const int mid_val = (int)audio_pcm16[n];

            for (int k = 0; k < IRDT_NUM_CHANNELS; ++k) {
                const int delay = g_irdt_channels[k];
                int lo = n - delay;
                int hi = n + delay;
                if (lo < 0) lo = 0;
                if (hi >= IRDT_DESIRED_SAMPLES) hi = IRDT_DESIRED_SAMPLES - 1;

                const int lap = (int)audio_pcm16[lo] + (int)audio_pcm16[hi] - (mid_val << 1);
                spectrum[win][k] += (uint32_t)abs(lap);
            }
        }
    }

    // Step 2: Frame aggregation & log2(1 + x)
    // Segments 0 to M-2 (0..48) are aggregated across win_per_segm (5) windows
    // Segment M-1 (49) is identically zero by design
    for (int s = 0; s < IRDT_NUM_SEGMENTS; ++s) {
        for (int k = 0; k < IRDT_NUM_CHANNELS; ++k) {
            float val = 0.0f;
            if (s < IRDT_NUM_SEGMENTS - 1) {
                uint32_t sum = 0;
                const int start_win = 1 + s * IRDT_WIN_PER_SEGM;
                for (int j = 0; j < IRDT_WIN_PER_SEGM; ++j) {
                    sum += spectrum[start_win + j][k];
                }
                val = ((float)sum) / 32768.0f;
                val = log2f(1.0f + val);
            }
            out_float[s * IRDT_NUM_CHANNELS + k] = val;
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif // IRDT_FEATURE_EXTRACTOR_TENET_H
"""

with open("save/tenet_narrow/irdt_feature_extractor.h", "w") as f:
    f.write(irdt_h_content)
with open(os.path.join(package_dir, "irdt_feature_extractor.h"), "w") as f:
    f.write(irdt_h_content)
with open(os.path.join(workspace_package, "irdt_feature_extractor.h"), "w") as f:
    f.write(irdt_h_content)

print("--> Generated irdt_feature_extractor.h")

# 3. Load Embedded Test Audio Clips
test_file = "configs/v1_irdt/test.txt"
data_root = r"C:\XboxGames\Dataset\speech_commands_v0.02"

with open(test_file) as f:
    lines = [l.strip() for l in f if l.strip()]

m_path = [l for l in lines if "marvin" in l][0]
u_path = [l for l in lines if "marvin" not in l and "_silence_" not in l][0]
s_path = [l for l in lines if "_silence_" in l][0]

def load_audio_int16(rel_path):
    full = os.path.join(data_root, rel_path)
    with wave.open(full, "rb") as wf:
        audio = np.frombuffer(wf.readframes(16000), dtype=np.int16)
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)))
    return audio[:16000]

audio_m = load_audio_int16(m_path)
audio_u = load_audio_int16(u_path)
audio_s = load_audio_int16(s_path)

def dump_c_array(arr, name):
    lines = [f"// Sample: {name} (16,000 int16 samples @ 16 kHz)",
             f"alignas(4) const int16_t {name}[16000] = {{"]
    for i in range(0, len(arr), 12):
        chunk = arr[i:i+12]
        lines.append("  " + ", ".join(f"{int(x)}" for x in chunk) + ",")
    lines.append("};\n")
    return "\n".join(lines)

test_samples_h = f"""#ifndef TEST_SAMPLES_EMBEDDED_H_
#define TEST_SAMPLES_EMBEDDED_H_

#include <stdint.h>

{dump_c_array(audio_m, "g_test_sample_marvin")}
{dump_c_array(audio_u, "g_test_sample_unknown")}
{dump_c_array(audio_s, "g_test_sample_silence")}

#endif // TEST_SAMPLES_EMBEDDED_H_
"""

with open("save/tenet_narrow/test_samples_embedded.h", "w") as f:
    f.write(test_samples_h)
with open(os.path.join(package_dir, "test_samples_embedded.h"), "w") as f:
    f.write(test_samples_h)
with open(os.path.join(workspace_package, "test_samples_embedded.h"), "w") as f:
    f.write(test_samples_h)

print("--> Generated test_samples_embedded.h")

# 4. Generate Arduino Sketch
sketch_code = """/**
 * ESP32-S3 Physical On-Chip Benchmark for TENet6-Narrow with iRDT.
 * 
 * Evaluates:
 * 1. Log-iRDT Feature Extraction Latency
 * 2. TENet6-Narrow Inference Latency (TFLM Invoke)
 * 3. Total End-to-End Latency & Real-Time Factor (RTF)
 * 4. SRAM & Internal Heap Memory Usage
 * 5. On-Device Prediction Accuracy across embedded test samples
 */

#include <TensorFlowLite_ESP32.h>
#include <Arduino.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tenet6_narrow_model_data.h"
#include "irdt_feature_extractor.h"
#include "test_samples_embedded.h"

// Configuration
#define BENCHMARK_ITERATIONS 5
#define WARMUP_ITERATIONS    5

// Tensor Arena in internal SRAM
constexpr int kTensorArenaSize = 120 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Global pointers
static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input_tensor = nullptr;
static TfLiteTensor* g_output_tensor = nullptr;

// Feature buffer
static float g_features[IRDT_TOTAL_FEATURES];

// Latency statistics helper
struct LatencyStats {
    float mean_ms;
    float median_ms;
    float min_ms;
    float max_ms;
};

static LatencyStats compute_stats(int64_t* times_us, int count) {
    for (int i = 0; i < count - 1; ++i) {
        for (int j = i + 1; j < count; ++j) {
            if (times_us[j] < times_us[i]) {
                int64_t tmp = times_us[i];
                times_us[i] = times_us[j];
                times_us[j] = tmp;
            }
        }
    }
    int64_t sum = 0;
    for (int i = 0; i < count; ++i) sum += times_us[i];
    LatencyStats s;
    s.mean_ms = (float)sum / (count * 1000.0f);
    s.median_ms = (float)times_us[count / 2] / 1000.0f;
    s.min_ms = (float)times_us[0] / 1000.0f;
    s.max_ms = (float)times_us[count - 1] / 1000.0f;
    return s;
}

static void run_sample_inference(const int16_t* sample, float* p_marvin, float* p_unk, float* p_sil) {
    compute_irdt_features_float(sample, g_features);
    for (int i = 0; i < IRDT_TOTAL_FEATURES; ++i) {
        g_input_tensor->data.f[i] = g_features[i];
    }
    g_interpreter->Invoke();
    *p_marvin = g_output_tensor->data.f[0];
    *p_unk    = g_output_tensor->data.f[1];
    *p_sil    = g_output_tensor->data.f[2];
}

void run_full_benchmark() {
    Serial.println("\\n=======================================================================");
    Serial.println("   ESP32-S3 ON-CHIP BENCHMARK: TENet6-Narrow INT8 WITH iRDT FEATURES   ");
    Serial.println("=======================================================================");
    Serial.printf("Running %d benchmark iterations...\\n", BENCHMARK_ITERATIONS);

    // Warmup
    Serial.printf("Starting Warmup (%d runs)...\\n", WARMUP_ITERATIONS);
    for (int w = 0; w < WARMUP_ITERATIONS; ++w) {
        Serial.printf("  Warmup %d: feat... ", w);
        compute_irdt_features_float(g_test_sample_marvin, g_features);
        for (int i = 0; i < IRDT_TOTAL_FEATURES; ++i) {
            g_input_tensor->data.f[i] = g_features[i];
        }
        Serial.print("invoking... ");
        TfLiteStatus s = g_interpreter->Invoke();
        Serial.printf("done (status %d)\\n", s);
        delay(2);
    }
    Serial.println("Warmup complete!");

    int64_t feat_times_us[BENCHMARK_ITERATIONS];
    int64_t nn_times_us[BENCHMARK_ITERATIONS];
    int64_t e2e_times_us[BENCHMARK_ITERATIONS];

    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        int64_t t0 = esp_timer_get_time();
        compute_irdt_features_float(g_test_sample_marvin, g_features);
        for (int k = 0; k < IRDT_TOTAL_FEATURES; ++k) {
            g_input_tensor->data.f[k] = g_features[k];
        }
        int64_t t1 = esp_timer_get_time();

        g_interpreter->Invoke();
        int64_t t2 = esp_timer_get_time();

        feat_times_us[i] = (t1 - t0);
        nn_times_us[i] = (t2 - t1);
        e2e_times_us[i] = (t2 - t0);

        Serial.print("  [Run "); Serial.print(i + 1); Serial.print("/"); Serial.print(BENCHMARK_ITERATIONS);
        Serial.print("] iRDT: "); Serial.print((float)feat_times_us[i] / 1000.0f);
        Serial.print(" ms | NN: "); Serial.print((float)nn_times_us[i] / 1000.0f);
        Serial.print(" ms | E2E: "); Serial.print((float)e2e_times_us[i] / 1000.0f);
        Serial.println(" ms");
        Serial.flush();
        delay(2);
    }

    LatencyStats feat_stats = compute_stats(feat_times_us, BENCHMARK_ITERATIONS);
    LatencyStats nn_stats = compute_stats(nn_times_us, BENCHMARK_ITERATIONS);
    LatencyStats e2e_stats = compute_stats(e2e_times_us, BENCHMARK_ITERATIONS);

    // Accuracy validation on embedded test clips
    float p_m_marvin, p_m_unk, p_m_sil;
    float p_u_marvin, p_u_unk, p_u_sil;
    float p_s_marvin, p_s_unk, p_s_sil;

    run_sample_inference(g_test_sample_marvin, &p_m_marvin, &p_m_unk, &p_m_sil);
    run_sample_inference(g_test_sample_unknown, &p_u_marvin, &p_u_unk, &p_u_sil);
    run_sample_inference(g_test_sample_silence, &p_s_marvin, &p_s_unk, &p_s_sil);

    uint32_t free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    uint32_t min_free_sram = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);

    // Print Formatted Report Table
    Serial.println("\\n-----------------------------------------------------------------------");
    Serial.println("1. LATENCY BENCHMARK (over sustained runs @ 240 MHz Xtensa LX7)");
    Serial.println("-----------------------------------------------------------------------");
    Serial.printf("  iRDT Feature Extraction:  Mean: %6.2f ms | Median: %6.2f ms | Min: %6.2f ms | Max: %6.2f ms\\n",
                  feat_stats.mean_ms, feat_stats.median_ms, feat_stats.min_ms, feat_stats.max_ms);
    Serial.printf("  Neural Network Inference: Mean: %6.2f ms | Median: %6.2f ms | Min: %6.2f ms | Max: %6.2f ms\\n",
                  nn_stats.mean_ms, nn_stats.median_ms, nn_stats.min_ms, nn_stats.max_ms);
    Serial.printf("  Total End-to-End Latency: Mean: %6.2f ms | Median: %6.2f ms | Min: %6.2f ms | Max: %6.2f ms\\n",
                  e2e_stats.mean_ms, e2e_stats.median_ms, e2e_stats.min_ms, e2e_stats.max_ms);
    Serial.printf("  Real-Time Factor (RTF):   %6.4f  (Processes 1.0s audio in %.2f ms)\\n",
                  e2e_stats.mean_ms / 1000.0f, e2e_stats.mean_ms);

    Serial.println("\\n-----------------------------------------------------------------------");
    Serial.println("2. MEMORY UTILIZATION ON ESP32-S3");
    Serial.println("-----------------------------------------------------------------------");
    Serial.printf("  Flash Footprint (Model):  %d bytes (%.1f KB)\\n", g_tenet6_narrow_model_data_len, g_tenet6_narrow_model_data_len / 1024.0f);
    Serial.printf("  Tensor Arena Size:        %d bytes (%.1f KB)\\n", kTensorArenaSize, kTensorArenaSize / 1024.0f);
    Serial.printf("  Free Internal SRAM:       %u bytes (%.1f KB free)\\n", free_sram, free_sram / 1024.0f);
    Serial.printf("  Min Ever Free SRAM:       %u bytes (%.1f KB)\\n", min_free_sram, min_free_sram / 1024.0f);

    Serial.println("\\n-----------------------------------------------------------------------");
    Serial.println("3. ON-DEVICE VALIDATION ON EMBEDDED TEST SAMPLES");
    Serial.println("-----------------------------------------------------------------------");
    Serial.printf("  [Test Clip 1: 'MARVIN']    -> P(Marvin)=%5.1f%% | P(Unknown)=%5.1f%% | P(Silence)=%5.1f%% -> %s\\n",
                  p_m_marvin * 100.0f, p_m_unk * 100.0f, p_m_sil * 100.0f,
                  (p_m_marvin > 0.50f) ? "PASS (Correctly Detected)" : "FAIL");
    Serial.printf("  [Test Clip 2: 'UNKNOWN']   -> P(Marvin)=%5.1f%% | P(Unknown)=%5.1f%% | P(Silence)=%5.1f%% -> %s\\n",
                  p_u_marvin * 100.0f, p_u_unk * 100.0f, p_u_sil * 100.0f,
                  (p_u_unk > 0.50f && p_u_marvin < 0.20f) ? "PASS (No False Alarm)" : "FAIL");
    Serial.printf("  [Test Clip 3: 'BACKGROUND']-> P(Marvin)=%5.1f%% | P(Unknown)=%5.1f%% | P(Silence)=%5.1f%% -> %s\\n",
                  p_s_marvin * 100.0f, p_s_unk * 100.0f, p_s_sil * 100.0f,
                  (p_s_sil > 0.50f && p_s_marvin < 0.20f) ? "PASS (No False Alarm)" : "FAIL");

    Serial.println("=======================================================================\\n");

    // Output JSON for host parsing
    Serial.println("---BENCHMARK_JSON_START---");
    Serial.printf("{\\"board\\":\\"ESP32-S3\\",\\"cpu_freq_mhz\\":%u,", getCpuFrequencyMhz());
    Serial.printf("\\"feat_mean_ms\\":%.2f,\\"nn_mean_ms\\":%.2f,\\"e2e_mean_ms\\":%.2f,",
                  feat_stats.mean_ms, nn_stats.mean_ms, e2e_stats.mean_ms);
    Serial.printf("\\"rtf\\":%.4f,\\"model_flash_kb\\":%.1f,\\"tensor_arena_kb\\":%.1f,\\"free_sram_kb\\":%.1f,",
                  e2e_stats.mean_ms / 1000.0f, g_tenet6_narrow_model_data_len / 1024.0f,
                  kTensorArenaSize / 1024.0f, free_sram / 1024.0f);
    Serial.printf("\\"p_marvin\\":%.3f,\\"p_unknown\\":%.3f,\\"p_silence\\":%.3f}\\n",
                  p_m_marvin, p_u_unk, p_s_sil);
    Serial.println("---BENCHMARK_JSON_END---");
}

void setup() {
    Serial.begin(115200);
    delay(2000);

    g_model = tflite::GetModel(g_tenet6_narrow_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Schema mismatch: %d != %d\\n", g_model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

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
        Serial.println("AllocateTensors failed!");
        return;
    }

    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);

    run_full_benchmark();
    Serial.println("Type 'b' and press Enter to rerun benchmark at any time.");
}

void loop() {
    if (Serial.available()) {
        char c = Serial.read();
        if ((c == 'b' || c == 'B') && g_input_tensor != nullptr) {
            run_full_benchmark();
        }
    }
    delay(100);
}
"""

with open(os.path.join(package_dir, "ESP32S3_TENet_Benchmark.ino"), "w", encoding="utf-8") as f:
    f.write(sketch_code)
with open(os.path.join(workspace_package, "ESP32S3_TENet_Benchmark.ino"), "w", encoding="utf-8") as f:
    f.write(sketch_code)

# Create ZIP archives
shutil.make_archive("ESP32S3_TENet_Arduino_Package", "zip", r"C:\Users\sinth\Downloads\ESP32S3_TENet_Arduino_Package")
shutil.copy("ESP32S3_TENet_Arduino_Package.zip", r"C:\Users\sinth\Downloads\ESP32S3_TENet_Arduino_Package.zip")

print("--> Successfully packaged ESP32S3_TENet_Arduino_Package.zip!")
