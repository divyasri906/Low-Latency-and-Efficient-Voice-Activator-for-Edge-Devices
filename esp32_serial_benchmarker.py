"""
ESP32-S3 Serial Monitor & On-Chip Benchmark Collector.

Features:
- Scans and detects connected ESP32-S3 COM ports
- Connects to Serial Monitor @ 115200 baud
- Resets ESP32-S3 to trigger the on-chip benchmark
- Live streams ESP32-S3 serial output to terminal
- Parses structured benchmark JSON telemetry
- Saves on-device metrics to save/dscnn_irdt/esp32s3_on_chip_benchmark_results.json
"""

import sys
import time
import json
import serial
import serial.tools.list_ports


def list_com_ports():
    ports = list(serial.tools.list_ports.comports())
    return ports


def main():
    print("=" * 75)
    print(" ESP32-S3 ON-CHIP SERIAL BENCHMARK MONITOR")
    print("=" * 75)

    ports = list_com_ports()
    if not ports:
        print("\n[!] No COM ports detected!")
        print("    Please connect your ESP32-S3 to your PC via a USB cable.")
        print("    Once plugged in, run this script again:\n")
        print("        python esp32_serial_benchmarker.py\n")
        sys.exit(1)

    print("\nDetected COM Ports:")
    for idx, p in enumerate(ports):
        print(f"  [{idx}] {p.device}: {p.description}")

    # Auto-select if only 1 port, or choose
    if len(ports) == 1:
        target_port = ports[0].device
        print(f"\n--> Auto-selecting port: {target_port}")
    else:
        target_port = ports[0].device
        print(f"\n--> Defaulting to first port: {target_port}")

    baud_rate = 115200
    print(f"--> Connecting to {target_port} at {baud_rate} baud...")

    try:
        ser = serial.Serial(target_port, baud_rate, timeout=1)
    except Exception as e:
        print(f"[!] Error opening {target_port}: {e}")
        sys.exit(1)

    # Trigger hardware reset via RTS/DTR
    print("--> Resetting ESP32-S3...")
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setRTS(False)
    time.sleep(0.5)

    print("--> Listening to ESP32-S3 Serial Output (Press Ctrl+C to stop)...\n")

    json_capturing = False
    json_lines = []
    benchmark_data = None

    start_time = time.time()
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="replace").rstrip()
                print(line)

                if "---BENCHMARK_JSON_START---" in line:
                    json_capturing = True
                    json_lines = []
                    continue
                elif "---BENCHMARK_JSON_END---" in line:
                    json_capturing = False
                    try:
                        raw_json = "".join(json_lines)
                        benchmark_data = json.loads(raw_json)
                        print("\n[✓] Successfully captured on-device benchmark JSON!")
                    except Exception as ex:
                        print(f"[!] Failed to parse benchmark JSON: {ex}")
                    continue

                if json_capturing:
                    json_lines.append(line)

            time.sleep(0.01)

            # If benchmark captured and completed, break after a short grace period
            if benchmark_data and (time.time() - start_time > 8.0):
                break

    except KeyboardInterrupt:
        print("\n[!] User interrupted serial monitor.")
    finally:
        ser.close()

    if benchmark_data:
        save_path = "save/dscnn_irdt/esp32s3_on_chip_benchmark_results.json"
        with open(save_path, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        print(f"\n--> Saved on-chip benchmark to {save_path}")

        print("\n" + "=" * 65)
        print(" ESP32-S3 ON-CHIP BENCHMARK SUMMARY")
        print("=" * 65)
        print(f" Board:                     {benchmark_data.get('board')} @ {benchmark_data.get('cpu_freq_mhz')} MHz")
        print(f" iRDT Feature Latency:      {benchmark_data.get('feat_mean_ms')} ms")
        print(f" Neural Network Latency:    {benchmark_data.get('nn_mean_ms')} ms")
        print(f" Total End-to-End Latency:  {benchmark_data.get('e2e_mean_ms')} ms")
        print(f" Real-Time Factor (RTF):    {benchmark_data.get('rtf')}")
        print(f" Model Flash Footprint:     {benchmark_data.get('model_flash_kb')} KB")
        print(f" Tensor Arena (RAM):        {benchmark_data.get('tensor_arena_kb')} KB")
        print(f" Free Internal SRAM:        {benchmark_data.get('free_sram_kb')} KB")
        print(f" Confidence on 'MARVIN':    {benchmark_data.get('p_marvin')*100:.1f}%")
        print("=" * 65)


if __name__ == "__main__":
    main()
