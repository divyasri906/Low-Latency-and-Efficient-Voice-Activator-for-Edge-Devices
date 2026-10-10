import time
import json
import numpy as np
import librosa
import keras

def profile():
    model = keras.models.load_model("model_training/models/dscnn_matched_15k_seed42.keras")
    rng = np.random.default_rng(42)
    raw_pcm = rng.integers(-20000, 20000, size=(16000,), dtype=np.int16)

    # Warmup
    for _ in range(50):
        s = raw_pcm.astype(np.float32) / 32768.0
        m = librosa.feature.mfcc(y=s, sr=16000, n_mfcc=13, n_fft=512, hop_length=320).T[:49, :]
        _ = model(m[np.newaxis, :, :, np.newaxis], training=False)

    num_runs = 500
    t1, t2, t3, t4, t_e2e = [], [], [], [], []

    # 1. Ingestion
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        s = raw_pcm.astype(np.float32) * (1.0 / 32768.0)
        t1.append((time.perf_counter_ns() - t0) / 1e6)

    # 2. MFCC
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        m = librosa.feature.mfcc(y=s, sr=16000, n_mfcc=13, n_fft=512, hop_length=320).T[:49, :]
        t2.append((time.perf_counter_ns() - t0) / 1e6)
    m_b = m[np.newaxis, :, :, np.newaxis]

    # 3. NN Forward
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        p = model(m_b, training=False).numpy()
        t3.append((time.perf_counter_ns() - t0) / 1e6)
    sample_out = p[0]

    # 4. Decision
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        pred = int(np.argmax(sample_out))
        t4.append((time.perf_counter_ns() - t0) / 1e6)

    # 5. Full End-to-End
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        s_f = raw_pcm.astype(np.float32) * (1.0 / 32768.0)
        mf = librosa.feature.mfcc(y=s_f, sr=16000, n_mfcc=13, n_fft=512, hop_length=320).T[:49, :]
        mb = mf[np.newaxis, :, :, np.newaxis]
        probs = model(mb, training=False).numpy()
        pred = int(np.argmax(probs[0]))
        t_e2e.append((time.perf_counter_ns() - t0) / 1e6)

    sum_iso = np.mean(t1) + np.mean(t2) + np.mean(t3) + np.mean(t4)
    total = np.mean(t_e2e)
    overhead = max(0.0, total - sum_iso)

    lat_data = {
        "ingestion_ms": float(np.mean(t1)),
        "feature_extraction_ms": float(np.mean(t2)),
        "nn_forward_ms": float(np.mean(t3)),
        "decision_ms": float(np.mean(t4)),
        "sum_isolated_ms": float(sum_iso),
        "inter_stage_overhead_ms": float(overhead),
        "total_e2e_ms": float(total),
        "p95_e2e_ms": float(np.percentile(t_e2e, 95)),
        "rtf": float(total / 1000.0),
        "throughput_fps": float(1000.0 / total),
    }

    # Update json
    res_path = "model_training/models/dscnn_matched_test_results.json"
    with open(res_path, "r") as f:
        data = json.load(f)
    data["latency"] = lat_data
    with open(res_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Clean DS-CNN Latency Profiling (500 runs):")
    print(f"  Ingestion:  {lat_data['ingestion_ms']:.4f} ms")
    print(f"  MFCC Feat:  {lat_data['feature_extraction_ms']:.4f} ms")
    print(f"  NN Forward: {lat_data['nn_forward_ms']:.4f} ms")
    print(f"  Decision:   {lat_data['decision_ms']:.4f} ms")
    print(f"  Sum Iso:    {lat_data['sum_isolated_ms']:.4f} ms")
    print(f"  Overhead:   {lat_data['inter_stage_overhead_ms']:.4f} ms")
    print(f"  Total E2E:  {lat_data['total_e2e_ms']:.4f} ms (p95: {lat_data['p95_e2e_ms']:.4f} ms)")
    print(f"  Throughput: {lat_data['throughput_fps']:.1f} FPS | RTF: {lat_data['rtf']:.5f}")

if __name__ == "__main__":
    profile()
