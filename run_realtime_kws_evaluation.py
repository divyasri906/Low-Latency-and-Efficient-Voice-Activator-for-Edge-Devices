import serial
import time
import json
import os

PORT = "COM19"
BAUD = 115200

def run_evaluation():
    print(f"Connecting to ESP32-S3 on {PORT} @ {BAUD} baud...")
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(1.0)
    ser.reset_input_buffer()

    conditions = [
        ("1", "SILENCE", 22),       # Run silence for ~2 telemetry reports
        ("2", "BACKGROUND", 25),    # Run background TV/speech for ~2 reports
        ("3", "MARVIN", 25),        # Run marvin spoken repeatedly for ~2 reports
    ]

    all_records = []
    current_json = None

    for cmd, name, duration in conditions:
        print(f"\n==================================================")
        print(f"--> SWITCHING CONDITION: {name} (Sending '{cmd}')")
        print(f"==================================================")
        ser.write(cmd.encode('utf-8'))
        ser.flush()

        t_start = time.time()
        in_json = False
        json_lines = []

        while time.time() - t_start < duration:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            if not line:
                continue

            print(line)

            if "---REALTIME_JSON_START---" in line:
                in_json = True
                json_lines = []
                continue
            elif "---REALTIME_JSON_END---" in line:
                in_json = False
                raw_json = "".join(json_lines).strip()
                try:
                    data = json.loads(raw_json)
                    data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    data["tested_mode"] = name
                    all_records.append(data)
                    print(f"\n[PARSED HARDWARE TELEMETRY] Mode={data['condition']} | Core0={data['core0_cpu_pct']}% | Core1={data['core1_cpu_pct']}% | SoC={data['chip_load_pct']}% | Target<10%={data['target_less_10pct']}\n")
                except Exception as e:
                    print("JSON decode error:", e, raw_json)
                continue

            if in_json:
                json_lines.append(line)

    ser.close()
    print("\nBenchmark Complete! Saving results...")
    
    os.makedirs("save", exist_ok=True)
    out_path = os.path.join("save", "esp32s3_realtime_energy_gated_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Saved {len(all_records)} hardware telemetry records to {out_path}")

if __name__ == "__main__":
    run_evaluation()
