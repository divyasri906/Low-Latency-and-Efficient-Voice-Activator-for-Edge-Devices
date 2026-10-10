#!/usr/bin/env python3
"""
Raspberry Pi 5 Hardware Benchmark Runner for Embedded KWS Models:
1. DS-CNN + MFCC (INT8)
2. DS-CNN + iRDT (INT8)
3. TENet6-Narrow + Log-iRDT (INT8 / Dynamic Range)

Measures:
- Feature Extraction Latency (us / ms)
- TFLite / LiteRT Neural Network Inference Latency (us / ms)
- End-to-End Latency & Real-Time Factor (RTF)
- CPU Core Utilization & Frequency
- RSS Memory & Peak RAM (KB / MB)
- On-Device Prediction Probabilities for (Marvin, Unknown, Silence)
"""

import os
import sys
import time
import json
import ctypes
import resource
import subprocess
import numpy as np
import psutil

# Import LiteRT / TFLite
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_cpu_info():
    info = {
        "model": "ARM Cortex-A76 (Raspberry Pi 5)",
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True)
    }
    try:
        freq = psutil.cpu_freq()
        info["freq_mhz"] = round(freq.current, 1) if freq else 2400.0
        info["max_freq_mhz"] = round(freq.max, 1) if freq else 2400.0
    except Exception:
        info["freq_mhz"] = 2400.0
        info["max_freq_mhz"] = 2400.0
    return info

def compile_c_engine():
    so_path = os.path.join(SCRIPT_DIR, "libkws_features.so")
    c_path = os.path.join(SCRIPT_DIR, "pi_kws_features.c")
    if not os.path.exists(so_path) or os.path.getmtime(c_path) > os.path.getmtime(so_path):
        print(f"--> Compiling {c_path} with gcc -O3...")
        cmd = ["gcc", "-O3", "-fPIC", "-shared", "-o", so_path, c_path, "-lm"]
        subprocess.check_call(cmd)
        print("--> libkws_features.so compiled successfully.")
    return so_path

class KWSFeatureExtractor:
    def __init__(self, so_path):
        self.lib = ctypes.CDLL(so_path)
        
        # void kws_extract_mfcc_int8(const int16_t* audio_pcm16, int8_t* out_int8)
        self.lib.kws_extract_mfcc_int8.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int8)
        ]
        self.lib.kws_extract_mfcc_int8.restype = None

        # void kws_extract_irdt_int8(const int16_t* audio_pcm16, int8_t* out_int8)
        self.lib.kws_extract_irdt_int8.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int8)
        ]
        self.lib.kws_extract_irdt_int8.restype = None

        # void kws_extract_log_irdt_float(const int16_t* audio_pcm16, float* out_float)
        self.lib.kws_extract_log_irdt_float.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.kws_extract_log_irdt_float.restype = None

    def extract_mfcc(self, pcm16):
        out = np.empty((49, 13), dtype=np.int8)
        p_in = pcm16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        p_out = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        self.lib.kws_extract_mfcc_int8(p_in, p_out)
        return out.reshape((1, 49, 13, 1))

    def extract_irdt(self, pcm16):
        out = np.empty((50, 6), dtype=np.int8)
        p_in = pcm16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        p_out = out.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        self.lib.kws_extract_irdt_int8(p_in, p_out)
        return out.reshape((1, 50, 6, 1))

    def extract_log_irdt(self, pcm16):
        out = np.empty((50, 6), dtype=np.float32)
        p_in = pcm16.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        p_out = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.kws_extract_log_irdt_float(p_in, p_out)
        return out.reshape((1, 50, 6))

def benchmark_model(name, model_path, feat_type, feat_extractor, test_samples, num_warmup=10, num_runs=100, num_threads=1):
    print(f"\n========================================================")
    print(f" BENCHMARKING: {name} (threads={num_threads})")
    print(f" Model File: {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")
    print(f"========================================================")

    # Initialize TFLite / LiteRT Interpreter
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    out_scale = output_details["quantization"][0]
    out_zero_point = output_details["quantization"][1]

    def run_inference(inp_tensor):
        interpreter.set_tensor(input_details["index"], inp_tensor)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_details["index"])
        if output_details["dtype"] == np.int8:
            probs = (raw_out.astype(np.float32) - out_zero_point) * out_scale
        else:
            probs = raw_out[0].astype(np.float32)
        return probs.flatten()

    def get_features(audio):
        if feat_type == "mfcc":
            return feat_extractor.extract_mfcc(audio)
        elif feat_type == "irdt":
            return feat_extractor.extract_irdt(audio)
        elif feat_type == "log_irdt":
            return feat_extractor.extract_log_irdt(audio)
        else:
            raise ValueError(f"Unknown feat_type {feat_type}")

    sample_marvin = test_samples["marvin"]
    sample_unknown = test_samples["unknown"]
    sample_silence = test_samples["silence"]

    # Warmup
    print(f"--> Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        feat = get_features(sample_marvin)
        _ = run_inference(feat)

    # Accuracy / Predictions on test samples
    # Output class order: 0 = 'marvin', 1 = '_unknown_', 2 = '_silence_'
    classes = ["marvin", "_unknown_", "_silence_"]
    probs_marvin = run_inference(get_features(sample_marvin))
    probs_unknown = run_inference(get_features(sample_unknown))
    probs_silence = run_inference(get_features(sample_silence))

    pred_marvin = classes[int(np.argmax(probs_marvin))]
    pred_unknown = classes[int(np.argmax(probs_unknown))]
    pred_silence = classes[int(np.argmax(probs_silence))]

    print(f"--> Test Predictions:")
    print(f"    Marvin audio  : {pred_marvin.upper()} (P_marvin = {probs_marvin[0]:.4f})")
    print(f"    Unknown audio : {pred_unknown.upper()} (P_unknown = {probs_unknown[1]:.4f})")
    print(f"    Silence audio : {pred_silence.upper()} (P_silence = {probs_silence[2]:.4f})")

    # Benchmarking runs
    print(f"--> Running {num_runs} timed iterations...")
    feat_times_us = []
    nn_times_us = []
    e2e_times_us = []

    # CPU utilization measurement
    proc = psutil.Process()
    cpu_start_times = proc.cpu_times()
    wall_start = time.perf_counter()

    for i in range(num_runs):
        # Audio sample cycling
        sample = sample_marvin if (i % 3 == 0) else (sample_unknown if i % 3 == 1 else sample_silence)

        t0 = time.perf_counter_ns()
        feat = get_features(sample)
        t1 = time.perf_counter_ns()
        _ = run_inference(feat)
        t2 = time.perf_counter_ns()

        f_us = (t1 - t0) / 1000.0
        n_us = (t2 - t1) / 1000.0
        e_us = (t2 - t0) / 1000.0

        feat_times_us.append(f_us)
        nn_times_us.append(n_us)
        e2e_times_us.append(e_us)

    wall_duration = time.perf_counter() - wall_start
    cpu_end_times = proc.cpu_times()
    cpu_user = cpu_end_times.user - cpu_start_times.user
    cpu_system = cpu_end_times.system - cpu_start_times.system
    cpu_util_pct = ((cpu_user + cpu_system) / wall_duration) * 100.0 if wall_duration > 0 else 0.0

    # Memory usage
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_kb = float(rusage.ru_maxrss)
    cur_rss_kb = float(proc.memory_info().rss) / 1024.0

    f_arr = np.array(feat_times_us)
    n_arr = np.array(nn_times_us)
    e_arr = np.array(e2e_times_us)

    res = {
        "model_name": name,
        "model_file": os.path.basename(model_path),
        "model_size_kb": round(os.path.getsize(model_path) / 1024.0, 2),
        "threads": num_threads,
        "runs": num_runs,
        "feat_mean_ms": round(float(np.mean(f_arr)) / 1000.0, 4),
        "feat_std_ms": round(float(np.std(f_arr)) / 1000.0, 4),
        "feat_median_ms": round(float(np.median(f_arr)) / 1000.0, 4),
        "feat_p95_ms": round(float(np.percentile(f_arr, 95)) / 1000.0, 4),
        "nn_mean_ms": round(float(np.mean(n_arr)) / 1000.0, 4),
        "nn_std_ms": round(float(np.std(n_arr)) / 1000.0, 4),
        "nn_median_ms": round(float(np.median(n_arr)) / 1000.0, 4),
        "nn_p95_ms": round(float(np.percentile(n_arr, 95)) / 1000.0, 4),
        "e2e_mean_ms": round(float(np.mean(e_arr)) / 1000.0, 4),
        "e2e_median_ms": round(float(np.median(e_arr)) / 1000.0, 4),
        "e2e_p95_ms": round(float(np.percentile(e_arr, 95)) / 1000.0, 4),
        "rtf": round(float(np.mean(e_arr)) / 1000000.0, 6),
        "realtime_speedup_x": round(1000000.0 / float(np.mean(e_arr)), 2),
        "cpu_util_pct": round(cpu_util_pct, 1),
        "peak_rss_kb": round(peak_rss_kb, 1),
        "cur_rss_kb": round(cur_rss_kb, 1),
        "p_marvin": round(float(probs_marvin[0]), 4),
        "p_unknown": round(float(probs_unknown[1]), 4),
        "p_silence": round(float(probs_silence[2]), 4),
        "pred_marvin": pred_marvin,
        "pred_unknown": pred_unknown,
        "pred_silence": pred_silence
    }

    print(f"--> Results:")
    print(f"    Feature Latency : {res['feat_mean_ms']:.4f} ms (p95: {res['feat_p95_ms']:.4f} ms)")
    print(f"    NN Inference    : {res['nn_mean_ms']:.4f} ms (p95: {res['nn_p95_ms']:.4f} ms)")
    print(f"    End-to-End Lat  : {res['e2e_mean_ms']:.4f} ms (p95: {res['e2e_p95_ms']:.4f} ms)")
    print(f"    RTF             : {res['rtf']:.6f} ({res['realtime_speedup_x']}x faster than real-time)")
    print(f"    CPU Util        : {res['cpu_util_pct']}% of 1 core")
    print(f"    Peak RSS RAM    : {res['peak_rss_kb'] / 1024.0:.2f} MB")
    return res

def main():
    print("=" * 65)
    print(" RASPBERRY PI 5 - EMBEDDED KWS ON-CHIP BENCHMARK SUITE")
    print("=" * 65)

    cpu = get_cpu_info()
    print(f"Target Hardware : {cpu['model']}")
    print(f"Physical Cores  : {cpu['cores']}, Threads: {cpu['threads']}")
    print(f"CPU Frequency   : {cpu['freq_mhz']} MHz (Max: {cpu['max_freq_mhz']} MHz)")
    print(f"OS / Kernel     : {os.uname().sysname} {os.uname().release} ({os.uname().machine})")

    so_path = compile_c_engine()
    extractor = KWSFeatureExtractor(so_path)

    samples_path = os.path.join(SCRIPT_DIR, "test_samples.npz")
    if not os.path.exists(samples_path):
        print(f"[!] Error: {samples_path} not found!")
        sys.exit(1)

    test_samples = np.load(samples_path)
    print(f"--> Loaded test samples: {list(test_samples.keys())}")

    models = [
        {
            "name": "DS-CNN MFCC",
            "path": os.path.join(SCRIPT_DIR, "dscnn_mfcc_15k_int8.tflite"),
            "feat_type": "mfcc"
        },
        {
            "name": "DS-CNN iRDT",
            "path": os.path.join(SCRIPT_DIR, "dscnn_irdt_15k_int8.tflite"),
            "feat_type": "irdt"
        },
        {
            "name": "TENet6-Narrow Log-iRDT",
            "path": os.path.join(SCRIPT_DIR, "best_marvin_model_log_int8.tflite"),
            "feat_type": "log_irdt"
        }
    ]

    all_results = {
        "board": "Raspberry Pi 5",
        "cpu_arch": "ARM Cortex-A76 @ 2.4 GHz (aarch64)",
        "kernel": f"{os.uname().sysname} {os.uname().release}",
        "cpu_info": cpu,
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "single_thread_results": {},
        "multi_thread_results": {}
    }

    # 1. Single Thread Benchmark (apples-to-apples with microcontrollers)
    print("\n" + "#" * 65)
    print(" PART 1: SINGLE-THREADED EXECUTION (1 Core)")
    print("#" * 65)
    for m in models:
        res = benchmark_model(m["name"], m["path"], m["feat_type"], extractor, test_samples, num_warmup=10, num_runs=100, num_threads=1)
        all_results["single_thread_results"][m["name"]] = res

    # 2. Multi-Thread Benchmark (4 Cores NEON)
    print("\n" + "#" * 65)
    print(" PART 2: MULTI-THREADED EXECUTION (4 Cores)")
    print("#" * 65)
    for m in models:
        res = benchmark_model(m["name"], m["path"], m["feat_type"], extractor, test_samples, num_warmup=10, num_runs=100, num_threads=4)
        all_results["multi_thread_results"][m["name"]] = res

    # Save JSON
    out_json = os.path.join(SCRIPT_DIR, "rpi5_kws_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Benchmark results successfully written to: {out_json}")

if __name__ == "__main__":
    main()
