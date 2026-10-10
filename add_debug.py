ino_path = r'C:\Users\sinth\Downloads\ESP32S3_KWS_Arduino_Package\ESP32S3_KWS_Benchmark\ESP32S3_KWS_Benchmark.ino'
with open(ino_path, 'r', encoding='utf-8') as f:
    text = f.read()

warmup_old = """    // Warmup
    for (int i = 0; i < 5; ++i) {
        compute_irdt_features_int8(g_test_sample_marvin, in_data);
        g_interpreter->Invoke();
    }"""

warmup_new = """    // Warmup
    Serial.println("Starting Warmup (5 runs)...");
    for (int i = 0; i < 5; ++i) {
        Serial.printf("  Warmup %d: feat... ", i);
        compute_irdt_features_int8(g_test_sample_marvin, in_data);
        Serial.print("invoking... ");
        TfLiteStatus st = g_interpreter->Invoke();
        Serial.printf("done (status %d)\\n", (int)st);
    }
    Serial.println("Warmup complete!");"""

bench_old = """    // Benchmark loop
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {"""

bench_new = """    // Benchmark loop
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        if (i % 10 == 0) {
            Serial.printf("  Progress: iteration %d/%d...\\n", i, BENCHMARK_ITERATIONS);
        }"""

text = text.replace(warmup_old, warmup_new)
text = text.replace(bench_old, bench_new)

with open(ino_path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Patched with debug prints successfully!')
