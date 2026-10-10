"""
ESP32-S3 Serial Monitor & On-Chip Benchmark Collector for TENet6-Narrow + iRDT.
"""

import sys
import os
import time
import json
import serial
import serial.tools.list_ports


def find_esp32_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "COM19" in p.device or "CH343" in p.description or "USB" in p.description:
            return p.device
    if ports:
        return ports[0].device
    return None


def main():
    print("=" * 75)
    print(" ESP32-S3 TENet6-Narrow + iRDT ON-CHIP BENCHMARK MONITOR")
    print("=" * 75)

    port = find_esp32_port()
    if not port:
        print("[!] No COM port found!")
        sys.exit(1)

    print(f"--> Target Port: {port} @ 115200 baud")

    try:
        ser = serial.Serial(port, 115200, timeout=1)
    except Exception as e:
        print(f"[!] Error opening {port}: {e}")
        sys.exit(1)

    # Hardware reset
    print("--> Resetting ESP32-S3...")
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setRTS(False)
    time.sleep(0.5)

    print("--> Listening to ESP32-S3 Serial Output...\n")

    json_capturing = False
    json_lines = []
    benchmark_data = None
    start_time = time.time()
    max_wait = 60.0  # seconds

    try:
        while time.time() - start_time < max_wait:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="replace").rstrip()
                print(line, flush=True)

                if "---BENCHMARK_JSON_START---" in line:
                    json_capturing = True
                    json_lines = []
                    continue
                elif "---BENCHMARK_JSON_END---" in line:
                    json_capturing = False
                    try:
                        raw_json = "".join(json_lines)
                        benchmark_data = json.loads(raw_json)
                        print("\n[OK] Successfully captured on-device benchmark JSON!", flush=True)
                        break
                    except Exception as ex:
                        print(f"[!] Failed to parse benchmark JSON: {ex}", flush=True)
                    continue

                if json_capturing:
                    json_lines.append(line)

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[!] User stopped monitor.")
    finally:
        ser.close()

    if benchmark_data:
        os.makedirs("save/tenet_narrow", exist_ok=True)
        save_path = "save/tenet_narrow/esp32s3_on_chip_benchmark_results.json"
        with open(save_path, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        print(f"\n--> Benchmark data successfully saved to: {save_path}")

        print("\n" + "=" * 65)
        print(" ESP32-S3 TENet6-Narrow + iRDT ON-CHIP BENCHMARK SUMMARY")
        print("=" * 65)
        print(f" Board:                     {benchmark_data.get('board')} @ {benchmark_data.get('cpu_freq_mhz')} MHz")
        print(f" iRDT Feature Latency:      {benchmark_data.get('feat_mean_ms')} ms")
        print(f" Neural Network Latency:    {benchmark_data.get('nn_mean_ms')} ms")
        print(f" Total End-to-End Latency:  {benchmark_data.get('e2e_mean_ms')} ms")
        print(f" Real-Time Factor (RTF):    {benchmark_data.get('rtf')}")
        print(f" Model Flash Footprint:     {benchmark_data.get('model_flash_kb')} KB")
        print(f" Tensor Arena (RAM):        {benchmark_data.get('tensor_arena_kb')} KB")
        print(f" Free Internal SRAM:        {benchmark_data.get('free_sram_kb')} KB")
        print(f" Confidence on 'MARVIN':    {benchmark_data.get('p_marvin', 0)*100:.1f}%")
        print("=" * 65)
    else:
        print("\n[!] No benchmark JSON captured within timeout.")


if __name__ == "__main__":
    main()
