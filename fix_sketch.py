import os
import re
import shutil

target_dirs = [
    r"C:\Users\sinth\Downloads\ESP32S3_KWS_Arduino_Package",
    r"save\dscnn_irdt\ESP32S3_KWS_Arduino_Package"
]

for base_dir in target_dirs:
    for sketch_name in ["ESP32S3_KWS_Benchmark", "ESP32S3_KWS_Live_Mic"]:
        ino_path = os.path.join(base_dir, sketch_name, f"{sketch_name}.ino")
        if not os.path.exists(ino_path):
            continue
        with open(ino_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Update tensor arena size
        content = re.sub(r'kTensorArenaSize\s*=\s*\d+\s*\*\s*1024;', 'kTensorArenaSize = 80 * 1024;', content)

        # Update op resolver
        content = content.replace('MicroMutableOpResolver<6>', 'MicroMutableOpResolver<7>')
        if 'AddFullyConnected' not in content:
            content = content.replace('resolver.AddSoftmax();', 'resolver.AddFullyConnected();\n    resolver.AddSoftmax();')

        # Safety check in run_full_benchmark / run_inference
        if 'if (g_input_tensor == nullptr) return;' not in content:
            content = content.replace('void run_full_benchmark() {', 'void run_full_benchmark() {\n    if (g_input_tensor == nullptr) {\n        Serial.println("Error: Tensors not allocated.");\n        return;\n    }')

        with open(ino_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Patched {ino_path}!")

# Update ZIP package
shutil.make_archive('ESP32S3_KWS_Arduino_Package', 'zip', 'save/dscnn_irdt/ESP32S3_KWS_Arduino_Package')
shutil.copy('ESP32S3_KWS_Arduino_Package.zip', 'save/dscnn_irdt/ESP32S3_KWS_Arduino_Package.zip')
print("ZIP archives synchronized successfully!")
