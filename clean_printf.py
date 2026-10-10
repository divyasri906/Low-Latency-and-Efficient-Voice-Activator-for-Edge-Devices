for path in [
    r'C:\Users\sinth\Downloads\ESP32S3_KWS_Arduino_Package\ESP32S3_KWS_Benchmark\ESP32S3_KWS_Benchmark.ino',
    r'save\dscnn_irdt\ESP32S3_KWS_Arduino_Package\ESP32S3_KWS_Benchmark\ESP32S3_KWS_Benchmark.ino'
]:
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'Serial.printf("  [Run %2d/%2d] iRDT:' in line:
            new_lines.append('            Serial.print("  [Run "); Serial.print(i + 1); Serial.print("/"); Serial.print(BENCHMARK_ITERATIONS); Serial.print("] iRDT: "); Serial.print((float)feat_times_us[i] / 1000.0f); Serial.print(" ms | NN: "); Serial.print((float)nn_times_us[i] / 1000.0f); Serial.print(" ms | E2E: "); Serial.print((float)e2e_times_us[i] / 1000.0f); Serial.println(" ms");\n')
            # skip broken continuation line if present
            if i + 1 < len(lines) and '", i + 1, BENCHMARK_ITERATIONS' in lines[i + 1]:
                i += 1
        else:
            new_lines.append(line)
        i += 1

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Replaced with safe Serial.print calls in:", path)
