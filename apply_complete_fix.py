import os
import zipfile

def fix_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Add micro_error_reporter.h if missing
    if "micro_error_reporter.h" not in code:
        code = code.replace(
            '#include "tensorflow/lite/micro/micro_interpreter.h"',
            '#include "tensorflow/lite/micro/micro_error_reporter.h"\n#include "tensorflow/lite/micro/micro_interpreter.h"'
        )

    # 2. Add error_reporter to MicroInterpreter call
    old_call = """    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, tensor_arena, kTensorArenaSize);"""
    
    new_call = """    static tflite::MicroErrorReporter micro_error_reporter;
    static tflite::ErrorReporter* error_reporter = &micro_error_reporter;
    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);"""

    if old_call in code:
        code = code.replace(old_call, new_call)
    elif "error_reporter" not in code.split("static_interpreter(")[1].split(";")[0]:
        # Handle alternate whitespace
        import re
        code = re.sub(
            r'static\s+tflite::MicroInterpreter\s+static_interpreter\s*\(\s*g_model\s*,\s*resolver\s*,\s*tensor_arena\s*,\s*kTensorArenaSize\s*\);',
            new_call,
            code
        )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Fixed: {filepath}")

paths = [
    r"C:\Users\sinth\Downloads\ESP32S3_KWS_Arduino_Package\ESP32S3_KWS_Benchmark\ESP32S3_KWS_Benchmark.ino",
    r"C:\Users\sinth\Downloads\ESP32S3_KWS_Arduino_Package\ESP32S3_KWS_Live_Mic\ESP32S3_KWS_Live_Mic.ino",
    r"save\dscnn_irdt\esp32s3_kws_benchmark.cpp",
    r"save\dscnn_irdt\esp32s3_kws_main.cpp"
]

for p in paths:
    if os.path.exists(p):
        fix_file(p)

# Update zip files
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
