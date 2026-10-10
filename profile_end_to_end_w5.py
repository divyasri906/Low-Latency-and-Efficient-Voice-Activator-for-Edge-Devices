"""
Comprehensive End-to-End Latency, MACs, Parameters, and Memory Profiler
for TENet6-Narrow + iRDT (W=5, M=50) Marvin KWS Model & 3-Seed Ensemble.
"""

import time
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import models
from irdt.irdtv import irdtv_optimized


def calculate_layer_macs(M=50, m_channels=6, num_classes=3):
    """
    Computes exact multiply-accumulate operations (MACs) layer by layer
    for TENet6-Narrow with M temporal frames and m_channels input channels.

    Architecture (from audio_nets/tenet.py):
      - inputs: [1, M, 1, m_channels]
      - conv0: conv2d kernel=[3,1], stride=1, out=16. MACs = M * 1 * (3 * 1 * m_channels) * 16
      - block0 (2 layers):
          - layer 0: stride=2 (M -> M/2 = 25).
              down: 1x1 conv, out=16. MACs = (M/2) * (1 * 1 * 16) * 16
              pw0: 1x1 conv, out=48 (ratio=3). MACs = M * 16 * 48
              dw0: 9x1 conv, stride=2, out=48. MACs = (M/2) * (9 * 1) * 48
              pw1: 1x1 conv, out=16. MACs = (M/2) * 48 * 16
          - layer 1: stride=1 (M/2 = 25).
              pw0: 1x1 conv, out=48. MACs = (M/2) * 16 * 48
              dw1: 9x1 conv, stride=1, out=48. MACs = (M/2) * 9 * 48
              pw1: 1x1 conv, out=16. MACs = (M/2) * 48 * 16
      - block1 (2 layers):
          - layer 0: stride=2 (25 -> 13).
              down: 1x1 conv, out=16. MACs = 13 * 16 * 16
              pw0: 1x1 conv, out=48. MACs = 25 * 16 * 48
              dw0: 9x1 conv, stride=2, out=48. MACs = 13 * 9 * 48
              pw1: 1x1 conv, out=16. MACs = 13 * 48 * 16
          - layer 1: stride=1 (13).
              pw0: 1x1 conv, out=48. MACs = 13 * 16 * 48
              dw1: 9x1 conv, stride=1, out=48. MACs = 13 * 9 * 48
              pw1: 1x1 conv, out=16. MACs = 13 * 48 * 16
      - block2 (2 layers):
          - layer 0: stride=2 (13 -> 7).
              down: 1x1 conv, out=16. MACs = 7 * 16 * 16
              pw0: 1x1 conv, out=48. MACs = 13 * 16 * 48
              dw0: 9x1 conv, stride=2, out=48. MACs = 7 * 9 * 48
              pw1: 1x1 conv, out=16. MACs = 7 * 48 * 16
          - layer 1: stride=1 (7).
              pw0: 1x1 conv, out=48. MACs = 7 * 16 * 48
              dw1: 9x1 conv, stride=1, out=48. MACs = 7 * 9 * 48
              pw1: 1x1 conv, out=16. MACs = 7 * 48 * 16
      - avg_pool: global pool over [7, 1] -> [1, 16]
      - fc: 1x1 conv, 16 -> num_classes. MACs = 16 * num_classes
    """
    macs = {}
    # conv0
    macs["conv0"] = M * (3 * m_channels) * 16

    # Block 0: in=M, out=M0
    M0 = int(np.ceil(M / 2)) # 25
    macs["b0_l0_down"] = M0 * 16 * 16
    macs["b0_l0_pw0"] = M * 16 * 48
    macs["b0_l0_dw0"] = M0 * 9 * 48
    macs["b0_l0_pw1"] = M0 * 48 * 16
    macs["b0_l1_pw0"] = M0 * 16 * 48
    macs["b0_l1_dw1"] = M0 * 9 * 48
    macs["b0_l1_pw1"] = M0 * 48 * 16

    # Block 1: in=M0, out=M1
    M1 = int(np.ceil(M0 / 2)) # 13
    macs["b1_l0_down"] = M1 * 16 * 16
    macs["b1_l0_pw0"] = M0 * 16 * 48
    macs["b1_l0_dw0"] = M1 * 9 * 48
    macs["b1_l0_pw1"] = M1 * 48 * 16
    macs["b1_l1_pw0"] = M1 * 16 * 48
    macs["b1_l1_dw1"] = M1 * 9 * 48
    macs["b1_l1_pw1"] = M1 * 48 * 16

    # Block 2: in=M1, out=M2
    M2 = int(np.ceil(M1 / 2)) # 7
    macs["b2_l0_down"] = M2 * 16 * 16
    macs["b2_l0_pw0"] = M1 * 16 * 48
    macs["b2_l0_dw0"] = M2 * 9 * 48
    macs["b2_l0_pw1"] = M2 * 48 * 16
    macs["b2_l1_pw0"] = M2 * 16 * 48
    macs["b2_l1_dw1"] = M2 * 9 * 48
    macs["b2_l1_pw1"] = M2 * 48 * 16

    # Classifier
    macs["fc"] = 16 * num_classes

    total_macs = sum(macs.values())

    # iRDT feature extraction MACs/Ops:
    # N=16000 audio samples, 6 delay channels.
    # Delay subtraction: 6 channels * (N - delay) operations approx 6 * 16000 = 96,000 ops.
    # Histogram sorting / threshold binning: 50 frames * (w=64 samples per segment) * 6 channels = ~30,000 operations.
    irdt_approx_ops = 16000 * m_channels + M * 64 * m_channels

    # Peak activation memory (largest intermediate tensor buffer during inference):
    # Largest tensor is typically the pw expansion in block 0: [1, M, 1, 48]
    peak_activation_elements = M * 1 * 48 + M * 1 * 16  # input + expanded buffer
    peak_ram_bytes_fp32 = peak_activation_elements * 4
    peak_ram_bytes_int8 = peak_activation_elements * 1

    return {
        "nn_macs": total_macs,
        "nn_mops": total_macs * 2,  # 1 MAC = 2 FLOPs/Ops (mul + add)
        "irdt_ops": irdt_approx_ops,
        "total_ops": (total_macs * 2) + irdt_approx_ops,
        "peak_ram_bytes_fp32": peak_ram_bytes_fp32,
        "peak_ram_bytes_int8": peak_ram_bytes_int8,
        "mac_breakdown": macs,
    }


def benchmark_latency(num_runs=100, warmup=20):
    # 1. Setup Models
    g = tf.Graph()
    with g.as_default():
        feat_ph = tf.compat.v1.placeholder(tf.float32, [1, 50, 6])
        model_args = argparse.Namespace(
            arch="TENet6NarrowIRDTModel", num_classes=3, width_multiplier=1.0, dropout=0.0,
            scope_name="", kernel_list=None, weight_decay=4e-5, desired_samples=16000,
            sample_rate=16000, irdt_w=64, irdt_chan=[1, 2, 4, 8, 16, 32],
            irdt_win_per_segm=5, irdt_biti=24, irdt_log=False, mod="eval", dataset_name="eval",
            window_size_samples=480, window_stride_samples=160, spectrogram_length=98,
        )

        model = models.__dict__[model_args.arch](model_args)
        logits_op = model.build_inference(feat_ph, is_training=False)
        probs_op = tf.nn.softmax(logits_op)

        sess = tf.compat.v1.Session()
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "save/ensemble_w5/seed_101/TENet6NarrowIRDTModel-3000")

    # Generate test 1-second audio
    rng = np.random.default_rng(42)
    raw_pcm_int16 = rng.integers(-20000, 20000, size=(16000,), dtype=np.int16)

    # Warmup
    for _ in range(warmup):
        audio_float = raw_pcm_int16.astype(np.float32) / 32768.0
        _, feat = irdtv_optimized(audio_float, [1, 2, 4, 8, 16, 32], 64, 5, 24, False)
        feat_batch = feat[np.newaxis, :, :].astype(np.float32)
        _ = sess.run(probs_op, feed_dict={feat_ph: feat_batch})

    # 1. Normalization Latency
    t_norm = []
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        audio_norm = raw_pcm_int16.astype(np.float32) * (1.0 / 32768.0)
        t1 = time.perf_counter_ns()
        t_norm.append((t1 - t0) / 1e6)

    # 2. iRDT Feature Extraction Latency
    t_irdt = []
    sample_feat = None
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        _, feat = irdtv_optimized(audio_norm, [1, 2, 4, 8, 16, 32], 64, 5, 24, False)
        t1 = time.perf_counter_ns()
        t_irdt.append((t1 - t0) / 1e6)
        sample_feat = feat

    feat_tensor = sample_feat[np.newaxis, :, :].astype(np.float32)

    # 3. Single NN Forward Pass Latency
    t_single_nn = []
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        p = sess.run(probs_op, feed_dict={feat_ph: feat_tensor})
        t1 = time.perf_counter_ns()
        t_single_nn.append((t1 - t0) / 1e6)

    # 4. Total End-to-End Single Model Latency (Audio -> Normalization -> iRDT -> NN -> Softmax)
    t_e2e_single = []
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        a = raw_pcm_int16.astype(np.float32) * (1.0 / 32768.0)
        _, f = irdtv_optimized(a, [1, 2, 4, 8, 16, 32], 64, 5, 24, False)
        fb = f[np.newaxis, :, :].astype(np.float32)
        p = sess.run(probs_op, feed_dict={feat_ph: fb})
        decision = (p[0, 0] >= 0.95)
        t1 = time.perf_counter_ns()
        t_e2e_single.append((t1 - t0) / 1e6)

    sess.close()

    # Now load 3 sessions to benchmark the complete 3-Model Ensemble Latency
    g_ens = tf.Graph()
    with g_ens.as_default():
        feat_ph_ens = tf.compat.v1.placeholder(tf.float32, [1, 50, 6])
        model_ens = models.__dict__["TENet6NarrowIRDTModel"](model_args)
        logits_ens = model_ens.build_inference(feat_ph_ens, is_training=False)
        probs_ens_op = tf.nn.softmax(logits_ens)

        sess_42 = tf.compat.v1.Session()
        sess_101 = tf.compat.v1.Session()
        sess_2024 = tf.compat.v1.Session()

        saver_ens = tf.compat.v1.train.Saver()
        saver_ens.restore(sess_42, "save/ensemble_w5/seed_42/TENet6NarrowIRDTModel-3000")
        saver_ens.restore(sess_101, "save/ensemble_w5/seed_101/TENet6NarrowIRDTModel-3000")
        saver_ens.restore(sess_2024, "save/ensemble_w5/seed_2024/TENet6NarrowIRDTModel-3000")

    # 5. Full End-to-End 3-Model Ensemble Latency
    t_e2e_ens = []
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        a = raw_pcm_int16.astype(np.float32) * (1.0 / 32768.0)
        # Note: Feature extraction is performed ONCE for all 3 models!
        _, f = irdtv_optimized(a, [1, 2, 4, 8, 16, 32], 64, 5, 24, False)
        fb = f[np.newaxis, :, :].astype(np.float32)
        p1 = sess_42.run(probs_ens_op, feed_dict={feat_ph_ens: fb})
        p2 = sess_101.run(probs_ens_op, feed_dict={feat_ph_ens: fb})
        p3 = sess_2024.run(probs_ens_op, feed_dict={feat_ph_ens: fb})
        p_avg = (p1[0] + p2[0] + p3[0]) / 3.0
        decision = (p_avg[0] >= 0.95)
        t1 = time.perf_counter_ns()
        t_e2e_ens.append((t1 - t0) / 1e6)

    sess_42.close()
    sess_101.close()
    sess_2024.close()

    return {
        "norm_ms": np.mean(t_norm),
        "irdt_ms": np.mean(t_irdt),
        "single_nn_ms": np.mean(t_single_nn),
        "e2e_single_ms": np.mean(t_e2e_single),
        "e2e_ens_ms": np.mean(t_e2e_ens),
        "e2e_ens_median_ms": np.median(t_e2e_ens),
        "e2e_ens_p95_ms": np.percentile(t_e2e_ens, 95),
        "e2e_single_median_ms": np.median(t_e2e_single),
        "e2e_single_p95_ms": np.percentile(t_e2e_single, 95),
    }


def main():
    print("=" * 80)
    print(" COMPUTING COMPUTATIONAL COMPLEXITY & BENCHMARKING END-TO-END LATENCY")
    print(" Model: TENet6-Narrow + iRDT (W=5, M=50 frames, m=6 delay channels)")
    print("=" * 80)

    # Complexity for W=5 (M=50)
    c5 = calculate_layer_macs(M=50, m_channels=6)
    # Complexity for baseline W=3 (M=83)
    c3 = calculate_layer_macs(M=83, m_channels=6)

    print("\n--- 1. THEORETICAL COMPUTATIONAL COMPLEXITY (W=5 vs W=3) ---")
    print(f"  Configuration               | W=5 (M=50 frames)   | W=3 Baseline (M=83) | Reduction")
    print(f"  ----------------------------+---------------------+---------------------+-----------")
    print(f"  Temporal Frames (M)         | {50:5d} frames        | {83:5d} frames        | -39.76%")
    print(f"  Single Model Parameters     | {15120:7,d} weights     | {15120:7,d} weights     | 0.00% (Identical)")
    print(f"  Single Model MACs           | {c5['nn_macs']:7,d} MACs       | {c3['nn_macs']:7,d} MACs       | {((c5['nn_macs']-c3['nn_macs'])/c3['nn_macs'])*100:+.2f}%")
    print(f"  Single Model FLOPs (MOPs)   | {c5['nn_mops']/1e6:7.2f} MFLOPs      | {c3['nn_mops']/1e6:7.2f} MFLOPs      | {((c5['nn_mops']-c3['nn_mops'])/c3['nn_mops'])*100:+.2f}%")
    print(f"  3-Model Ensemble Parameters | {15120*3:7,d} weights     | {15120*3:7,d} weights     | -")
    print(f"  3-Model Ensemble MACs       | {c5['nn_macs']*3:7,d} MACs       | {c3['nn_macs']*3:7,d} MACs       | {((c5['nn_macs']-c3['nn_macs'])/c3['nn_macs'])*100:+.2f}%")
    print(f"  Peak Activation RAM (INT8)  | {c5['peak_ram_bytes_int8']:7,d} Bytes      | {c3['peak_ram_bytes_int8']:7,d} Bytes      | -39.76%")
    print(f"  Peak Activation RAM (FP32)  | {c5['peak_ram_bytes_fp32']:7,d} Bytes      | {c3['peak_ram_bytes_fp32']:7,d} Bytes      | -39.76%")
    print(f"  Model Weights Flash (INT8)  | {15120/1024:7.2f} KB (x3={45360/1024:.1f}K) | {15120/1024:7.2f} KB          | -")
    print(f"  Model Weights Flash (FP32)  | {15120*4/1024:7.2f} KB (x3={45360*4/1024:.1f}K)| {15120*4/1024:7.2f} KB         | -")

    print("\n--- 2. EMPIRICAL WALL-CLOCK LATENCY BENCHMARK (100 runs on local CPU) ---")
    print("  Measuring stage-by-stage execution time on 1.0 second audio input (batch=1)...")
    lat = benchmark_latency(num_runs=100, warmup=20)

    print(f"\n  Stage-by-Stage Breakdown:")
    print(f"    1. Audio Ingestion / Normalization:   {lat['norm_ms']:.4f} ms")
    print(f"    2. iRDT Feature Extraction (W=5):    {lat['irdt_ms']:.4f} ms (Run ONCE even for ensemble!)")
    print(f"    3. Single NN Forward Pass:           {lat['single_nn_ms']:.4f} ms")
    print(f"  -------------------------------------------------------------")
    print(f"  SINGLE MODEL TOTAL END-TO-END LATENCY:  {lat['e2e_single_ms']:.2f} ms (Median: {lat['e2e_single_median_ms']:.2f} ms, P95: {lat['e2e_single_p95_ms']:.2f} ms)")
    print(f"  3-MODEL ENSEMBLE END-TO-END LATENCY:    {lat['e2e_ens_ms']:.2f} ms (Median: {lat['e2e_ens_median_ms']:.2f} ms, P95: {lat['e2e_ens_p95_ms']:.2f} ms)")
    print(f"  Real-time Factor (RTF, 1000ms audio):  {lat['e2e_ens_ms']/1000.0:.4f}x ({(1000.0/lat['e2e_ens_ms']):.1f}x faster than real-time)")


if __name__ == "__main__":
    main()
