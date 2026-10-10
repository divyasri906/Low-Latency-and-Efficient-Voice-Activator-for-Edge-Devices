import os

def update_sketch(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 1. Update BENCHMARK_ITERATIONS to 20
    code = code.replace('#define BENCHMARK_ITERATIONS 5', '#define BENCHMARK_ITERATIONS 20')

    # 2. Update LatencyStats struct
    old_struct = '''struct LatencyStats {
    float min_ms;
    float max_ms;
    float mean_ms;
    float median_ms;
};'''
    new_struct = '''struct LatencyStats {
    float min_ms;
    float max_ms;
    float mean_ms;
    float median_ms;
    float std_ms;
};'''
    if old_struct in code:
        code = code.replace(old_struct, new_struct)

    # 3. Update compute_stats
    old_compute_end = '''    s.median_ms = (float)times_us[count / 2] / 1000.0f;
    return s;'''
    new_compute_end = '''    s.median_ms = (float)times_us[count / 2] / 1000.0f;
    float variance = 0.0f;
    for (int i = 0; i < count; ++i) {
        float diff = ((float)times_us[i] / 1000.0f) - s.mean_ms;
        variance += diff * diff;
    }
    s.std_ms = sqrtf(variance / count);
    return s;'''
    if old_compute_end in code:
        code = code.replace(old_compute_end, new_compute_end)

    # 4. Make sure every run is printed
    old_print_cond = 'if (i % 5 == 0 || i == BENCHMARK_ITERATIONS - 1) {'
    if old_print_cond in code:
        code = code.replace(old_print_cond, 'if (true) {')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"Successfully updated: {path}")

targets = [
    r'save/dscnn_irdt/ESP32S3_KWS_Arduino_Package/ESP32S3_KWS_Benchmark/ESP32S3_KWS_Benchmark.ino',
    r'C:\Users\sinth\Downloads\ESP32S3_KWS_Arduino_Package\ESP32S3_KWS_Benchmark\ESP32S3_KWS_Benchmark.ino',
    r'save/dscnn_mfcc/ESP32S3_MFCC_Arduino_Package/ESP32S3_MFCC_Benchmark/ESP32S3_MFCC_Benchmark.ino',
    r'C:\Users\sinth\Downloads\ESP32S3_MFCC_Arduino_Package\ESP32S3_MFCC_Benchmark\ESP32S3_MFCC_Benchmark.ino',
    r'save/tenet_narrow/ESP32S3_TENet_Benchmark/ESP32S3_TENet_Benchmark.ino',
    r'C:\Users\sinth\Downloads\ESP32S3_TENet_Arduino_Package\ESP32S3_TENet_Benchmark\ESP32S3_TENet_Benchmark.ino',
]

for t in targets:
    update_sketch(t)
