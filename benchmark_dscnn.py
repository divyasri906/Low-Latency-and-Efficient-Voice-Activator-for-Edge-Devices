"""
Benchmark MFCC + DS-CNN (Keras 3) Latency
Matches model_training/testing_code.ipynb exactly:
1. Audio loading / normalization (16000 samples)
2. librosa.feature.mfcc(y, sr=16000, n_mfcc=13, n_fft=512, hop_length=320)
3. Transpose & frame padding to (1, 49, 13, 1)
4. Normalization via normalization.json
5. DS-CNN forward pass (medium.keras)
6. Classification decision
"""

import json
import time
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf


def get_stats(times_ms):
    times_ms = np.array(times_ms)
    return {
        "mean": float(np.mean(times_ms)),
        "median": float(np.median(times_ms)),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99)),
        "min": float(np.min(times_ms)),
        "max": float(np.max(times_ms)),
        "std": float(np.std(times_ms)),
    }


def main():
    model_path = Path("model_training/models/medium.keras")
    norm_path = Path("model_training/models/normalization.json")

    with open(norm_path, "r") as f:
        norm = json.load(f)
    train_mean = float(norm["mean"])
    train_std = float(norm["std"])

    dscnn_model = tf.keras.models.load_model(model_path)
    total_params = dscnn_model.count_params()

    rng = np.random.default_rng(42)
    raw_pcm = rng.integers(-20000, 20000, size=(16000,), dtype=np.int16)
    audio_float = raw_pcm.astype(np.float32) / 32768.0

    # Warmup
    for _ in range(20):
        mfcc = librosa.feature.mfcc(y=audio_float, sr=16000, n_mfcc=13, n_fft=512, hop_length=320)
        mfcc = mfcc[:, :49] if mfcc.shape[1] >= 49 else np.pad(mfcc, ((0, 0), (0, 49 - mfcc.shape[1])))
        mfcc = np.expand_dims(mfcc.T[..., np.newaxis], axis=0)
        mfcc_norm = (mfcc - train_mean) / (train_std + 1e-8)
        _ = dscnn_model(mfcc_norm, training=False).numpy()

    num_runs = 100

    # 1. Ingestion
    t_ingest = []
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        a = raw_pcm.astype(np.float32) * (1.0 / 32768.0)
        t1 = time.perf_counter_ns()
        t_ingest.append((t1 - t0) / 1e6)

    # 2. MFCC extraction
    t_mfcc = []
    sample_mfcc = None
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        mfcc = librosa.feature.mfcc(y=a, sr=16000, n_mfcc=13, n_fft=512, hop_length=320)
        mfcc = mfcc[:, :49] if mfcc.shape[1] >= 49 else np.pad(mfcc, ((0, 0), (0, 49 - mfcc.shape[1])))
        mfcc = np.expand_dims(mfcc.T[..., np.newaxis], axis=0)
        t1 = time.perf_counter_ns()
        t_mfcc.append((t1 - t0) / 1e6)
        sample_mfcc = mfcc

    # 3. Normalization
    t_norm = []
    normed_mfcc = None
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        normed_mfcc = ((sample_mfcc - train_mean) / (train_std + 1e-8)).astype(np.float32)
        t1 = time.perf_counter_ns()
        t_norm.append((t1 - t0) / 1e6)

    # 4. Neural Network Forward Pass
    t_nn = []
    sample_preds = None
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        preds = dscnn_model(normed_mfcc, training=False).numpy()
        t1 = time.perf_counter_ns()
        t_nn.append((t1 - t0) / 1e6)
        sample_preds = preds

    # 5. Post-processing
    t_post = []
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        pred_class = int(np.argmax(sample_preds[0]))
        confidence = float(sample_preds[0][pred_class])
        t1 = time.perf_counter_ns()
        t_post.append((t1 - t0) / 1e6)

    # 6. Full End-to-End
    t_e2e = []
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        a = raw_pcm.astype(np.float32) * (1.0 / 32768.0)
        m = librosa.feature.mfcc(y=a, sr=16000, n_mfcc=13, n_fft=512, hop_length=320)
        m = m[:, :49] if m.shape[1] >= 49 else np.pad(m, ((0, 0), (0, 49 - m.shape[1])))
        m = np.expand_dims(m.T[..., np.newaxis], axis=0)
        m_norm = ((m - train_mean) / (train_std + 1e-8)).astype(np.float32)
        p = dscnn_model(m_norm, training=False).numpy()
        c = int(np.argmax(p[0]))
        t1 = time.perf_counter_ns()
        t_e2e.append((t1 - t0) / 1e6)

    stats = {
        "params": total_params,
        "ingest": get_stats(t_ingest),
        "mfcc": get_stats(t_mfcc),
        "norm": get_stats(t_norm),
        "nn": get_stats(t_nn),
        "post": get_stats(t_post),
        "e2e": get_stats(t_e2e),
    }

    with open("model_training/latency_results.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("=" * 80)
    print(" MFCC + DS-CNN (medium.keras) LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Parameters: {total_params:,}")
    print(f"1. Audio Normalization:       {stats['ingest']['mean']:.3f} ms")
    print(f"2. MFCC Extraction (librosa): {stats['mfcc']['mean']:.3f} ms  (Median: {stats['mfcc']['median']:.3f} ms)")
    print(f"3. Feature Normalization:     {stats['norm']['mean']:.3f} ms")
    print(f"4. DS-CNN Forward Pass (NN):  {stats['nn']['mean']:.3f} ms  (Median: {stats['nn']['median']:.3f} ms)")
    print(f"5. Softmax & Decision:        {stats['post']['mean']:.3f} ms")
    print("-" * 80)
    print(f"TOTAL END-TO-END LATENCY:     {stats['e2e']['mean']:.3f} ms  (Median: {stats['e2e']['median']:.3f} ms, P95: {stats['e2e']['p95']:.3f} ms)")
    print(f"Real-Time Factor (RTF):       {stats['e2e']['mean']/1000.0:.4f}x")
    print(f"Max Throughput:               {1000.0/stats['e2e']['mean']:.1f} inferences/sec")
    print("=" * 80)


if __name__ == "__main__":
    main()
