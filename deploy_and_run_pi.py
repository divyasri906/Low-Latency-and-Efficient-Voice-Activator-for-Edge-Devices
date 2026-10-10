import os
import sys
import time
import json
import paramiko

PI_HOST = "10.255.2.225"
PI_USER = "gspi5"
PI_PASS = "12345678"
REMOTE_DIR = "/home/gspi5/kws_benchmarks"
PYTHON_BIN = "/home/gspi5/kws_env/bin/python"

FILES_TO_UPLOAD = {
    "save/dscnn_mfcc/dscnn_mfcc_15k_int8.tflite": f"{REMOTE_DIR}/dscnn_mfcc_15k_int8.tflite",
    "save/dscnn_irdt/dscnn_irdt_15k_int8.tflite": f"{REMOTE_DIR}/dscnn_irdt_15k_int8.tflite",
    "best_marvin_model_log_int8.tflite": f"{REMOTE_DIR}/best_marvin_model_log_int8.tflite",
    "test_samples.npz": f"{REMOTE_DIR}/test_samples.npz",
    "pi_kws_features.c": f"{REMOTE_DIR}/pi_kws_features.c",
    "rpi_benchmark_runner.py": f"{REMOTE_DIR}/rpi_benchmark_runner.py",
}

def main():
    print("=" * 70)
    print(" DEPLOYING AND RUNNING KWS BENCHMARK SUITE ON RASPBERRY PI 5")
    print("=" * 70)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"--> Connecting to {PI_USER}@{PI_HOST}...")
    client.connect(PI_HOST, username=PI_USER, password=PI_PASS, timeout=15)
    print("--> Connected!")

    sftp = client.open_sftp()
    try:
        sftp.mkdir(REMOTE_DIR)
    except IOError:
        pass

    for local_path, remote_path in FILES_TO_UPLOAD.items():
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file {local_path} does not exist!")
        size_kb = os.path.getsize(local_path) / 1024.0
        print(f"--> Uploading {local_path} ({size_kb:.1f} KB) -> {remote_path}...")
        sftp.put(local_path, remote_path)
    sftp.close()
    print("--> All files successfully uploaded to Raspberry Pi 5!\n")

    # Execute benchmark
    cmd = f"cd {REMOTE_DIR} && {PYTHON_BIN} rpi_benchmark_runner.py"
    print(f"--> Executing remote command: {cmd}\n")
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)

    # Stream output live
    for line in iter(stdout.readline, ""):
        safe_line = line.encode("ascii", errors="replace").decode("ascii").rstrip()
        print(safe_line)

    err = stderr.read().decode("utf-8", errors="replace")
    if err.strip():
        print("\n--- REMOTE STDERR ---")
        print(err.encode("ascii", errors="replace").decode("ascii"))

    # Download results JSON
    sftp = client.open_sftp()
    remote_json = f"{REMOTE_DIR}/rpi5_kws_benchmark_results.json"
    local_json_dir = "save/rpi5"
    os.makedirs(local_json_dir, exist_ok=True)
    local_json = os.path.join(local_json_dir, "rpi5_kws_benchmark_results.json")
    print(f"\n--> Downloading results: {remote_json} -> {local_json}...")
    sftp.get(remote_json, local_json)
    sftp.close()
    client.close()

    if os.path.exists(local_json):
        print(f"--> [SUCCESS] Benchmark JSON saved locally to: {local_json}")
        with open(local_json, "r") as f:
            data = json.load(f)
        print("\nSummary of Benchmark Results:")
        print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
