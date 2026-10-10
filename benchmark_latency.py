"""
Comprehensive End-to-End Latency & Stage-by-Stage Profiler for TENet-iRDT Marvin KWS

Measures:
1. Audio Ingestion / Normalization
2. iRDTv Feature Extraction (N=16000 -> (83, 6))
3. TENet6-Narrow Neural Network Forward Pass (TensorFlow session)
4. Post-processing (Softmax & Argmax Classification Decision)
5. Complete End-to-End Latency (Single 1-second audio clip -> Classification output)
6. Comparison with MFCC extraction
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import config
import models
from irdt.irdtv import irdtv_optimized


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark KWS Model Latency")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="save/TENet6NarrowIRDTModel/best/TENet6NarrowIRDTModel-best",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100,
        help="Number of timed benchmark runs (default: 100)",
    )
    parser.add_argument(
        "--warmup_runs",
        type=int,
        default=20,
        help="Number of warmup runs before timing (default: 20)",
    )
    return parser.parse_args()


def get_stats(times_ms):
    times_ms = np.array(times_ms)
    return {
        "mean": float(np.mean(times_ms)),
        "std": float(np.std(times_ms)),
        "median": float(np.median(times_ms)),
        "p90": float(np.percentile(times_ms, 90)),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99)),
        "min": float(np.min(times_ms)),
        "max": float(np.max(times_ms)),
    }


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def main():
    cli_args = parse_args()

    print("=" * 80)
    print(" TENet6-Narrow + iRDT Keyword Spotting: End-to-End Latency Profiler")
    print(f" Checkpoint:  {cli_args.checkpoint_path}")
    print(f" Benchmark:   {cli_args.num_runs} iterations (after {cli_args.warmup_runs} warmup runs)")
    print(f" Audio Input: 1.0 second (16,000 samples @ 16 kHz, single-stream batch=1)")
    print("=" * 80)

    # 1. Build TF Graph for TENet6-Narrow Inference
    tf.compat.v1.reset_default_graph()

    model_args = argparse.Namespace(
        arch="TENet6NarrowIRDTModel",
        num_classes=3,
        width_multiplier=1.0,
        dropout=0.0,
        scope_name="",
        kernel_list=None,
        weight_decay=4e-5,
        desired_samples=16000,
        sample_rate=16000,
        irdt_w=64,
        irdt_chan=[1, 2, 4, 8, 16, 32],
        irdt_win_per_segm=3,
        irdt_biti=24,
        irdt_log=False,
    )

    # Unified model graph from raw audio
    raw_audio_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 16000, 1], name="raw_audio")
    feat_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 83, 6], name="irdt_features")

    model = models.__dict__[model_args.arch](model_args)
    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        model.build(wavs=raw_audio_ph, labels=tf.constant([0], dtype=tf.int64), is_training=False)
        logits_from_raw = model.logits
        logits_from_feat = model.build_inference(feat_ph, is_training=False)

    session = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])

    saver = tf.compat.v1.train.Saver(var_list=model.model_variables)
    if os.path.exists(cli_args.checkpoint_path + ".index"):
        saver.restore(session, cli_args.checkpoint_path)
        print(f"[OK] Restored weights from: {cli_args.checkpoint_path}\n")
    else:
        print(f"[WARN] Checkpoint not found at {cli_args.checkpoint_path}. Benchmarking graph with initialized weights.\n")

    # Generate synthetic 1-second 16-bit PCM audio (as int16 from microphone ADC)
    rng = np.random.default_rng(42)
    raw_pcm_int16 = rng.integers(-20000, 20000, size=(16000,), dtype=np.int16)

    # -------------------------------------------------------------------------
    # Warmup Phase
    # -------------------------------------------------------------------------
    print("Warming up execution pipeline and CPU caches...")
    for _ in range(cli_args.warmup_runs):
        audio_float = raw_pcm_int16.astype(np.float32) / 32768.0
        _, feat = irdtv_optimized(audio_float, model_args.irdt_chan, model_args.irdt_w, model_args.irdt_win_per_segm, model_args.irdt_biti, model_args.irdt_log)
        feat_batch = feat[np.newaxis, :, :].astype(np.float32)
        logits_val = session.run(logits_from_feat, feed_dict={feat_ph: feat_batch})
        probs = softmax(logits_val[0])
        pred_class = np.argmax(probs)

        # Warmup unified graph
        raw_tensor = audio_float[np.newaxis, :, np.newaxis]
        _ = session.run(logits_from_raw, feed_dict={raw_audio_ph: raw_tensor})

    print(f"Warmup completed ({cli_args.warmup_runs} passes).\n")

    # -------------------------------------------------------------------------
    # Benchmark Stage 1: Audio Ingestion / Normalization
    # -------------------------------------------------------------------------
    t_stage1 = []
    for _ in range(cli_args.num_runs):
        t0 = time.perf_counter_ns()
        audio_norm = raw_pcm_int16.astype(np.float32) * (1.0 / 32768.0)
        t1 = time.perf_counter_ns()
        t_stage1.append((t1 - t0) / 1e6)

    # -------------------------------------------------------------------------
    # Benchmark Stage 2: iRDT Feature Extraction
    # -------------------------------------------------------------------------
    t_stage2 = []
    sample_feat = None
    for _ in range(cli_args.num_runs):
        t0 = time.perf_counter_ns()
        _, feat = irdtv_optimized(
            audio_norm,
            model_args.irdt_chan,
            model_args.irdt_w,
            model_args.irdt_win_per_segm,
            model_args.irdt_biti,
            model_args.irdt_log,
        )
        t1 = time.perf_counter_ns()
        t_stage2.append((t1 - t0) / 1e6)
        sample_feat = feat

    feat_tensor = sample_feat[np.newaxis, :, :].astype(np.float32)

    # -------------------------------------------------------------------------
    # Benchmark Stage 3: TENet6-Narrow Neural Network Forward Pass
    # -------------------------------------------------------------------------
    t_stage3 = []
    sample_logits = None
    for _ in range(cli_args.num_runs):
        t0 = time.perf_counter_ns()
        logits_val = session.run(logits_from_feat, feed_dict={feat_ph: feat_tensor})
        t1 = time.perf_counter_ns()
        t_stage3.append((t1 - t0) / 1e6)
        sample_logits = logits_val

    # -------------------------------------------------------------------------
    # Benchmark Stage 4: Post-Processing & Classification Decision
    # -------------------------------------------------------------------------
    t_stage4 = []
    for _ in range(cli_args.num_runs):
        t0 = time.perf_counter_ns()
        probs = softmax(sample_logits[0])
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        is_marvin = (pred_idx == 0) and (confidence >= 0.60)
        t1 = time.perf_counter_ns()
        t_stage4.append((t1 - t0) / 1e6)

    # -------------------------------------------------------------------------
    # Benchmark Stage 5: Full End-to-End Pipeline (Python/NumPy + TF Model)
    # -------------------------------------------------------------------------
    t_e2e_modular = []
    for _ in range(cli_args.num_runs):
        t0 = time.perf_counter_ns()
        # 1. Ingestion
        s_audio = raw_pcm_int16.astype(np.float32) * (1.0 / 32768.0)
        # 2. iRDTv
        _, s_feat = irdtv_optimized(s_audio, model_args.irdt_chan, model_args.irdt_w, model_args.irdt_win_per_segm, model_args.irdt_biti, model_args.irdt_log)
        s_feat_b = s_feat[np.newaxis, :, :].astype(np.float32)
        # 3. Model Forward Pass
        s_logits = session.run(logits_from_feat, feed_dict={feat_ph: s_feat_b})
        # 4. Decision
        s_probs = softmax(s_logits[0])
        s_pred = int(np.argmax(s_probs))
        t1 = time.perf_counter_ns()
        t_e2e_modular.append((t1 - t0) / 1e6)

    # -------------------------------------------------------------------------
    # Benchmark Stage 6: Unified TF Graph (raw audio placeholder -> logits)
    # -------------------------------------------------------------------------
    t_e2e_tf = []
    raw_input_batch = (raw_pcm_int16.astype(np.float32) / 32768.0)[np.newaxis, :, np.newaxis]
    for _ in range(cli_args.num_runs):
        t0 = time.perf_counter_ns()
        res = session.run(logits_from_raw, feed_dict={raw_audio_ph: raw_input_batch})
        _ = np.argmax(res[0])
        t1 = time.perf_counter_ns()
        t_e2e_tf.append((t1 - t0) / 1e6)

    session.close()

    # -------------------------------------------------------------------------
    # Compute Statistics
    # -------------------------------------------------------------------------
    s1 = get_stats(t_stage1)
    s2 = get_stats(t_stage2)
    s3 = get_stats(t_stage3)
    s4 = get_stats(t_stage4)
    s_e2e = get_stats(t_e2e_modular)
    s_tf = get_stats(t_e2e_tf)

    total_mean = s_e2e["mean"]

    # -------------------------------------------------------------------------
    # Formatted Latency Report
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(" LATENCY BENCHMARK RESULTS (Single 1.0s Audio Clip, batch_size=1)")
    print("=" * 80)
    header = f"{'Pipeline Stage / Component':<32} | {'Mean':>8} | {'Median':>8} | {'P95':>8} | {'P99':>8} | {'% Total':>8}"
    print(header)
    print("-" * 80)

    p1 = (s1['mean'] / total_mean) * 100
    p2 = (s2['mean'] / total_mean) * 100
    p3 = (s3['mean'] / total_mean) * 100
    p4 = (s4['mean'] / total_mean) * 100

    print(f"{'1. Audio Ingestion / Normalization':<32} | {s1['mean']:7.3f}ms | {s1['median']:7.3f}ms | {s1['p95']:7.3f}ms | {s1['p99']:7.3f}ms | {p1:6.1f}%")
    print(f"{'2. iRDT Feature Extraction (83, 6)':<32} | {s2['mean']:7.3f}ms | {s2['median']:7.3f}ms | {s2['p95']:7.3f}ms | {s2['p99']:7.3f}ms | {p2:6.1f}%")
    print(f"{'3. TENet6-Narrow Forward Pass (NN)':<32} | {s3['mean']:7.3f}ms | {s3['median']:7.3f}ms | {s3['p95']:7.3f}ms | {s3['p99']:7.3f}ms | {p3:6.1f}%")
    print(f"{'4. Softmax / Classification Decision':<32} | {s4['mean']:7.3f}ms | {s4['median']:7.3f}ms | {s4['p95']:7.3f}ms | {s4['p99']:7.3f}ms | {p4:6.1f}%")
    print("-" * 80)
    print(f"{'TOTAL END-TO-END LATENCY':<32} | {s_e2e['mean']:7.3f}ms | {s_e2e['median']:7.3f}ms | {s_e2e['p95']:7.3f}ms | {s_e2e['p99']:7.3f}ms | {'100.0%':>8}")
    print(f"  (Min: {s_e2e['min']:.3f} ms, Max: {s_e2e['max']:.3f} ms, Std: {s_e2e['std']:.3f} ms)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Edge Real-time Metrics
    # -------------------------------------------------------------------------
    rtf = s_e2e["mean"] / 1000.0
    throughput = 1000.0 / s_e2e["mean"]

    print("\n" + "=" * 80)
    print(" REAL-TIME STREAMING & THROUGHPUT METRICS")
    print("=" * 80)
    print(f" Real-Time Factor (RTF):   {rtf:.4f}x  (A value << 1.0 means much faster than real-time)")
    print(f" Execution Speedup:        {1.0 / rtf:.1f}x real-time (takes only {s_e2e['mean']:.2f}ms to process 1,000ms audio)")
    print(f" Maximum Throughput:       {throughput:.1f} inferences/sec on CPU (single thread)")
    print(f" CPU Headroom:             {(1.0 - rtf) * 100:.1f}% idle (available for other system tasks)")
    print("=" * 80)


if __name__ == "__main__":
    main()
