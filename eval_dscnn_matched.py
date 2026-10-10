"""
Evaluate Parameter-Matched DS-CNN Models on Marvin KWS Test Split & Benchmark Latency.
Runs in clean eager mode for Keras 3.
"""

import glob
import json
import os
import time
import wave
import numpy as np
import librosa
import keras


CLASS_NAMES = ["marvin", "unknown", "background"]
DATA_ROOT = r"C:\XboxGames\Dataset\speech_commands_v0.02"
SPLIT_DIR = "configs/v1_irdt"


def load_wav_file(path):
    with wave.open(path, "rb") as w:
        buf = w.readframes(16000)
        audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
    return audio[:16000]


def parse_split(split_file):
    paths = []
    labels = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            full_p = os.path.join(DATA_ROOT, line)
            f_name = line.split("/")[0]
            if f_name == "marvin":
                l = 0
            elif f_name == "_silence_":
                l = 2
            else:
                l = 1
            paths.append(full_p)
            labels.append(l)
    return paths, np.array(labels, dtype=np.int64)


def benchmark_cpu_latency_dscnn(model, num_runs=500, warmup=50):
    rng = np.random.default_rng(42)
    raw_pcm = rng.integers(-20000, 20000, size=(16000,), dtype=np.int16)

    # Warmup
    for _ in range(warmup):
        s_float = raw_pcm.astype(np.float32) / 32768.0
        mfcc = librosa.feature.mfcc(y=s_float, sr=16000, n_mfcc=13, n_fft=512, hop_length=320).T[:49, :]
        mfcc_b = mfcc[np.newaxis, :, :, np.newaxis]
        _ = model(mfcc_b, training=False)

    # Stage 1: Ingestion
    t1 = []
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        s_float = raw_pcm.astype(np.float32) * (1.0 / 32768.0)
        t_e = time.perf_counter_ns()
        t1.append((t_e - t_s) / 1e6)

    # Stage 2: MFCC
    t2 = []
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        mfcc = librosa.feature.mfcc(y=s_float, sr=16000, n_mfcc=13, n_fft=512, hop_length=320).T[:49, :]
        t_e = time.perf_counter_ns()
        t2.append((t_e - t_s) / 1e6)
    sample_mfcc = mfcc[np.newaxis, :, :, np.newaxis]

    # Stage 3: NN Forward Pass
    t3 = []
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        probs = model(sample_mfcc, training=False).numpy()
        t_e = time.perf_counter_ns()
        t3.append((t_e - t_s) / 1e6)
    sample_out = probs[0]

    # Stage 4: Decision
    t4 = []
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        pred = int(np.argmax(sample_out))
        t_e = time.perf_counter_ns()
        t4.append((t_e - t_s) / 1e6)

    # Stage 5: Full Unified Loop
    t_e2e = []
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        s_f = raw_pcm.astype(np.float32) * (1.0 / 32768.0)
        m = librosa.feature.mfcc(y=s_f, sr=16000, n_mfcc=13, n_fft=512, hop_length=320).T[:49, :]
        m_b = m[np.newaxis, :, :, np.newaxis]
        p = model(m_b, training=False).numpy()
        pred = int(np.argmax(p[0]))
        t_e = time.perf_counter_ns()
        t_e2e.append((t_e - t_s) / 1e6)

    sum_isolated = np.mean(t1) + np.mean(t2) + np.mean(t3) + np.mean(t4)
    total_e2e = np.mean(t_e2e)
    inter_overhead = max(0.0, total_e2e - sum_isolated)

    return {
        "ingestion_ms": float(np.mean(t1)),
        "feature_extraction_ms": float(np.mean(t2)),
        "nn_forward_ms": float(np.mean(t3)),
        "decision_ms": float(np.mean(t4)),
        "sum_isolated_ms": float(sum_isolated),
        "inter_stage_overhead_ms": float(inter_overhead),
        "total_e2e_ms": float(total_e2e),
        "p95_e2e_ms": float(np.percentile(t_e2e, 95)),
        "rtf": float(total_e2e / 1000.0),
        "throughput_fps": float(1000.0 / total_e2e),
    }


def main():
    test_paths, test_labels = parse_split(os.path.join(SPLIT_DIR, "test.txt"))
    print(f"Extracting test MFCCs for {len(test_paths)} samples...")
    raw_feats = []
    for p in test_paths:
        audio = load_wav_file(p)
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13, n_fft=512, hop_length=320).T[:49, :]
        raw_feats.append(mfcc)
    X_test_raw = np.array(raw_feats, dtype=np.float32)[:, :, :, np.newaxis]

    model_files = sorted(glob.glob("model_training/models/dscnn_matched_15k_seed*.keras"))
    results = []

    for mf in model_files:
        seed_name = os.path.basename(mf).replace(".keras", "")
        meta_file = mf.replace(".keras", "_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta = json.load(f)
            t_mean = meta["train_mean"]
            t_std = meta["train_std"]
        else:
            t_mean = float(np.mean(X_test_raw))
            t_std = float(np.std(X_test_raw)) + 1e-7

        X_norm = (X_test_raw - t_mean) / t_std
        m = keras.models.load_model(mf)
        probs = m(X_norm, training=False).numpy()
        preds = np.argmax(probs, axis=1)

        acc = float(np.mean(preds == test_labels))
        marvin_recall = float(np.mean(preds[test_labels == 0] == 0))
        bg_rejection = float(np.mean(preds[test_labels == 2] == 2))
        unk_rejection = float(np.mean(preds[test_labels == 1] == 1))
        fa_rate = float(np.mean(preds[test_labels != 0] == 0))

        print(f"{seed_name}: Test Acc = {acc*100:.2f}%, Recall = {marvin_recall*100:.2f}%, FA = {fa_rate*100:.2f}%")

        results.append({
            "model_name": seed_name,
            "path": mf,
            "params": m.count_params(),
            "accuracy": acc,
            "marvin_recall": marvin_recall,
            "bg_rejection": bg_rejection,
            "unk_rejection": unk_rejection,
            "fa_rate": fa_rate,
            "predictions": preds.tolist(),
        })

    # Latency benchmark on the first model
    m_lat = keras.models.load_model(model_files[0])
    print("Profiling DS-CNN CPU latency (500 runs, batch size 1)...")
    lat_data = benchmark_cpu_latency_dscnn(m_lat, num_runs=500)
    print(f"DS-CNN End-to-End Latency: {lat_data['total_e2e_ms']:.3f} ms")

    output_payload = {
        "results": results,
        "latency": lat_data,
    }
    out_path = "model_training/models/dscnn_matched_test_results.json"
    with open(out_path, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Saved DS-CNN evaluation results to {out_path}")


if __name__ == "__main__":
    main()
