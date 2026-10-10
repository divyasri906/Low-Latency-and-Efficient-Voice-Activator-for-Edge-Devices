import os
import re

ino_path = r'C:\Users\sinth\Downloads\ESP32S3_KWS_Arduino_Package\ESP32S3_KWS_Benchmark\ESP32S3_KWS_Benchmark.ino'
with open(ino_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Set iterations to 20
text = re.sub(r'#define\s+BENCHMARK_ITERATIONS\s+\d+', '#define BENCHMARK_ITERATIONS 20', text)

# Replace the loop cleanly
loop_code = '''// Benchmark loop
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        int64_t t0 = esp_timer_get_time();
        compute_irdt_features_int8(g_test_sample_marvin, in_data);
        int64_t t1 = esp_timer_get_time();

        g_interpreter->Invoke();
        int64_t t2 = esp_timer_get_time();

        feat_times_us[i] = (t1 - t0);
        nn_times_us[i] = (t2 - t1);
        e2e_times_us[i] = (t2 - t0);

        if (i % 5 == 0 || i == BENCHMARK_ITERATIONS - 1) {
            Serial.printf("  [Run %2d/%2d] iRDT: %.2f ms | NN: %.2f ms | E2E: %.2f ms\\n", i + 1, BENCHMARK_ITERATIONS, (float)feat_times_us[i] / 1000.0f, (float)nn_times_us[i] / 1000.0f, (float)e2e_times_us[i] / 1000.0f);
            Serial.flush();
        }
        delay(2);
    }

    LatencyStats feat_stats'''

loop_regex = r'// Benchmark loop[\s\S]*?LatencyStats feat_stats'
text = re.sub(loop_regex, loop_code, text)

with open(ino_path, 'w', encoding='utf-8') as f:
    f.write(text)

save_ino = r'save\dscnn_irdt\ESP32S3_KWS_Arduino_Package\ESP32S3_KWS_Benchmark\ESP32S3_KWS_Benchmark.ino'
if os.path.exists(save_ino):
    with open(save_ino, 'w', encoding='utf-8') as f:
        f.write(text)

print('Sketch updated cleanly!')
