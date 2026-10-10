import os
import zipfile

files_to_update = [
    r"C:\Users\sinth\Downloads\ESP32S3_KWS_Arduino_Package\ESP32S3_KWS_Benchmark\ESP32S3_KWS_Benchmark.ino",
    r"C:\Users\sinth\Downloads\ESP32S3_KWS_Arduino_Package\ESP32S3_KWS_Live_Mic\ESP32S3_KWS_Live_Mic.ino",
    r"save\dscnn_irdt\esp32s3_kws_benchmark.cpp",
    r"save\dscnn_irdt\esp32s3_kws_main.cpp"
]

target = '#include "tensorflow/lite/micro/all_ops_resolver.h"'
replacement = '#include <TensorFlowLite_ESP32.h>\n#include "tensorflow/lite/micro/all_ops_resolver.h"'

for p in files_to_update:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            code = f.read()
        if "#include <TensorFlowLite_ESP32.h>" not in code:
            code = code.replace(target, replacement)
            with open(p, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"Added <TensorFlowLite_ESP32.h> to: {p}")
        else:
            print(f"Already had header: {p}")

pkg_dir = r"C:\Users\sinth\Downloads\ESP32S3_KWS_Arduino_Package"
for zip_dest in [r"save\dscnn_irdt\ESP32S3_KWS_Arduino_Package.zip", "ESP32S3_KWS_Arduino_Package.zip"]:
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(pkg_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, pkg_dir)
                zf.write(full_path, rel_path)
    print(f"Repackaged: {zip_dest}")

print("DONE!")
