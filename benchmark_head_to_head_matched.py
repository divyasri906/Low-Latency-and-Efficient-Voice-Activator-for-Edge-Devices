"""
Head-to-Head Latency, MACs, Accuracy & Statistical Significance Benchmark.

Compares:
- Model A: 2D MFCC + DS-CNN (15,121 parameters, matched)
- Model B: 1D iRDT + TENet6-Narrow (15,120 parameters, base or fused)

Features:
- Multi-seed accuracy reporting (mean +/- std).
- Paired McNemar's statistical significance test (chi^2 and p-value).
- Theoretical MACs / FLOPs calculation for both graphs.
- High-precision CPU latency profiling (batch_size=1, 500 runs):
  * Ingestion & Normalization
  * Feature Extraction (librosa MFCC vs integer iRDT)
  * Neural Network Forward Pass
  * Post-Processing Decision
  * Itemized Inter-stage Python Overhead
  * Total End-to-End Latency, RTF, Throughput.
"""

import argparse
import glob
import json
import os
import time
import wave
import numpy as np
from scipy import stats
import tensorflow as tf
import keras

tf.compat.v1.disable_eager_execution()

import config
import models
from irdt.irdtv import irdtv_optimized
import librosa


CLASS_NAMES = ["Marvin", "Unknown Speech", "Background"]
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


def compute_mcnemar_test(preds_a, preds_b, true_labels):
    """
    Computes McNemar's test with continuity correction on paired test predictions.
    a: both correct
    b: Model B correct, Model A wrong
    c: Model A correct, Model B wrong
    d: both wrong
    """
    correct_a = (preds_a == true_labels)
    correct_b = (preds_b == true_labels)

    a = int(np.sum(correct_a & correct_b))
    b = int(np.sum(~correct_a & correct_b))
    c = int(np.sum(correct_a & ~correct_b))
    d = int(np.sum(~correct_a & ~correct_b))

    denom = b + c
    if denom == 0:
        chi2 = 0.0
        p_val = 1.0
    else:
        chi2 = float((abs(b - c) - 1.0) ** 2 / denom)
        p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))

    return {
        "contingency_table": {"both_correct_a": a, "model_b_only": b, "model_a_only": c, "both_wrong_d": d},
        "chi2": chi2,
        "p_value": p_val,
        "statistically_significant": bool(p_val < 0.05),
    }


def calculate_theoretical_macs():
    """
    Theoretical MACs calculation:
    DS-CNN (49x13 MFCC, 15,121 params):
      - stem conv 3x3: 49 * 13 * (3 * 3 * 1 * 34) = 194,922
      - b1 dw 3x3: 49 * 13 * (3 * 3 * 34) = 194,922
      - b1 pw 1x1: 49 * 13 * (34 * 42) = 909,636
      - b2 dw 3x3: 49 * 13 * (3 * 3 * 42) = 240,786
      - b2 pw 1x1: 49 * 13 * (42 * 46) = 1,230,444
      - b3 dw 3x3: 49 * 13 * (3 * 3 * 46) = 263,718
      - b3 pw 1x1: 49 * 13 * (46 * 50) = 1,465,100
      - b4 dw 3x3: 49 * 13 * (3 * 3 * 50) = 286,650
      - b4 pw 1x1: 49 * 13 * (50 * 52) = 1,656,200
      - fc1: 52 * 54 = 2,808
      - output: 54 * 3 = 162
      Total DS-CNN MACs: ~6.45 Million MACs

    TENet6-Narrow (83x6 iRDT, 15,120 params):
      - stem 9x1: 83 * (9 * 6 * 16) = 71,712
      - block0 (stride 2 -> 42):
        pw: 83 * (16 * 48) = 63,744
        dw 9x1: 42 * (9 * 48) = 18,144
        pw: 42 * (48 * 16) = 32,256
      - block1 (stride 2 -> 21):
        pw: 42 * (16 * 48) = 32,256
        dw 9x1: 21 * (9 * 48) = 9,072
        pw: 21 * (48 * 16) = 16,128
      - block2 (stride 2 -> 11):
        pw: 21 * (16 * 48) = 16,128
        dw 9x1: 11 * (9 * 48) = 4,752
        pw: 11 * (48 * 16) = 8,448
      - shortcut convs: ~15,000
      - output: 16 * 3 = 48
      Total TENet6-Narrow MACs: ~0.29 Million MACs (22x fewer MACs!)
    """
    return {
        "dscnn_macs": 6_445_000,
        "tenet_macs": 288_000,
        "macs_ratio": 6_445_000 / 288_000,  # ~22.38x fewer MACs for TENet
    }


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
    sample_mfcc = None
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        mfcc = librosa.feature.mfcc(y=s_float, sr=16000, n_mfcc=13, n_fft=512, hop_length=320).T[:49, :]
        t_e = time.perf_counter_ns()
        t2.append((t_e - t_s) / 1e6)
    sample_mfcc = mfcc[np.newaxis, :, :, np.newaxis]

    # Stage 3: NN Forward Pass
    t3 = []
    sample_out = None
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
    timing_delta = total_e2e - sum_isolated

    return {
        "ingestion_ms": float(np.mean(t1)),
        "feature_extraction_ms": float(np.mean(t2)),
        "nn_forward_ms": float(np.mean(t3)),
        "decision_ms": float(np.mean(t4)),
        "sum_isolated_ms": float(sum_isolated),
        "timing_delta_ms": float(timing_delta),
        "total_e2e_ms": float(total_e2e),
        "p95_e2e_ms": float(np.percentile(t_e2e, 95)),
        "rtf": float(total_e2e / 1000.0),
        "throughput_fps": float(1000.0 / total_e2e),
    }


def benchmark_cpu_latency_tenet(sess, logits_op, feat_ph, num_runs=500, warmup=50):
    rng = np.random.default_rng(42)
    raw_pcm = rng.integers(-20000, 20000, size=(16000,), dtype=np.int16)
    chan = [1, 2, 4, 8, 16, 32]

    # Warmup
    for _ in range(warmup):
        s_float = raw_pcm.astype(np.float32) / 32768.0
        _, feat = irdtv_optimized(s_float, chan, 64, 3, 24, False)
        feat_b = feat[np.newaxis, :, :].astype(np.float32)
        _ = sess.run(logits_op, feed_dict={feat_ph: feat_b})

    # Stage 1: Ingestion
    t1 = []
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        s_float = raw_pcm.astype(np.float32) * (1.0 / 32768.0)
        t_e = time.perf_counter_ns()
        t1.append((t_e - t_s) / 1e6)

    # Stage 2: iRDT
    t2 = []
    sample_feat = None
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        _, feat = irdtv_optimized(s_float, chan, 64, 3, 24, False)
        t_e = time.perf_counter_ns()
        t2.append((t_e - t_s) / 1e6)
    sample_feat = feat[np.newaxis, :, :].astype(np.float32)

    # Stage 3: NN Forward Pass
    t3 = []
    sample_logits = None
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        l_val = sess.run(logits_op, feed_dict={feat_ph: sample_feat})
        t_e = time.perf_counter_ns()
        t3.append((t_e - t_s) / 1e6)
    sample_logits = l_val[0]

    # Stage 4: Decision
    t4 = []
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        exp_l = np.exp(sample_logits - np.max(sample_logits))
        probs = exp_l / np.sum(exp_l)
        pred = int(np.argmax(probs))
        t_e = time.perf_counter_ns()
        t4.append((t_e - t_s) / 1e6)

    # Stage 5: Full Unified Loop
    t_e2e = []
    for _ in range(num_runs):
        t_s = time.perf_counter_ns()
        s_f = raw_pcm.astype(np.float32) * (1.0 / 32768.0)
        _, f_v = irdtv_optimized(s_f, chan, 64, 3, 24, False)
        f_b = f_v[np.newaxis, :, :].astype(np.float32)
        l = sess.run(logits_op, feed_dict={feat_ph: f_b})
        p = np.exp(l[0] - np.max(l[0]))
        pred = int(np.argmax(p))
        t_e = time.perf_counter_ns()
        t_e2e.append((t_e - t_s) / 1e6)

    sum_isolated = np.mean(t1) + np.mean(t2) + np.mean(t3) + np.mean(t4)
    total_e2e = np.mean(t_e2e)
    timing_delta = total_e2e - sum_isolated

    return {
        "ingestion_ms": float(np.mean(t1)),
        "feature_extraction_ms": float(np.mean(t2)),
        "nn_forward_ms": float(np.mean(t3)),
        "decision_ms": float(np.mean(t4)),
        "sum_isolated_ms": float(sum_isolated),
        "timing_delta_ms": float(timing_delta),
        "total_e2e_ms": float(total_e2e),
        "p95_e2e_ms": float(np.percentile(t_e2e, 95)),
        "rtf": float(total_e2e / 1000.0),
        "throughput_fps": float(1000.0 / total_e2e),
    }


def evaluate_dscnn_models(test_paths, test_labels):
    print("Evaluating Matched DS-CNN models on test set...")
    # 1. Extract unaugmented test features
    features = []
    for p in test_paths:
        audio = load_wav_file(p)
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13, n_fft=512, hop_length=320).T[:49, :]
        features.append(mfcc)
    X_test_raw = np.array(features, dtype=np.float32)[:, :, :, np.newaxis]

    # Find models
    model_files = sorted(glob.glob("model_training/models/dscnn_matched_15k_seed*.keras"))
    if not model_files:
        # Fallback to medium.keras if matched models not yet finished
        model_files = ["model_training/models/medium.keras"]

    results = []
    all_seed_preds = []

    for mf in model_files:
        seed_name = os.path.basename(mf).replace(".keras", "")
        # Load metadata for normalization if exists
        meta_file = mf.replace(".keras", "_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta = json.load(f)
            t_mean = meta["train_mean"]
            t_std = meta["train_std"]
        else:
            norm_path = "model_training/models/normalization.json"
            if os.path.exists(norm_path):
                with open(norm_path, "r") as f:
                    norm = json.load(f)
                t_mean = norm["mean"]
                t_std = norm["std"]
            else:
                t_mean = float(np.mean(X_test_raw))
                t_std = float(np.std(X_test_raw)) + 1e-7

        X_norm = (X_test_raw - t_mean) / t_std
        m = keras.models.load_model(mf)
        probs = m.predict(X_norm, batch_size=100, verbose=0)
        preds = np.argmax(probs, axis=1)

        acc = float(np.mean(preds == test_labels))
        marvin_recall = float(np.mean(preds[test_labels == 0] == 0))
        bg_rejection = float(np.mean(preds[test_labels == 2] == 2))
        unk_rejection = float(np.mean(preds[test_labels == 1] == 1))
        fa_rate = float(np.mean(preds[test_labels != 0] == 0))

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
        all_seed_preds.append(preds)

    return results, X_norm, m


def evaluate_tenet_models(test_paths, test_labels):
    print("Evaluating TENet / MTENet Fused models on test set...")
    chan = [1, 2, 4, 8, 16, 32]
    features = []
    for p in test_paths:
        audio = load_wav_file(p)
        _, feat = irdtv_optimized(audio, chan, 64, 3, 24, False)
        features.append(feat)
    X_feat = np.array(features, dtype=np.float32)

    # Search for multi-seed checkpoints, fused checkpoints, or base checkpoint
    ckpt_candidates = []
    multi_seed_ckpts = sorted(glob.glob("save/TENet6Narrow_MultiSeed/seed_*/best/TENet6NarrowIRDTModel-best.index"))
    for f in multi_seed_ckpts:
        ckpt_candidates.append(f[:-6])
    if not ckpt_candidates:
        fused_ckpts = sorted(glob.glob("save/MTENet_Fusion/seed_*/fused_tenet/TENet6NarrowIRDTModel-fused.index"))
        for f in fused_ckpts:
            ckpt_candidates.append(f[:-6])
    if not ckpt_candidates:
        ckpt_candidates.append("save/TENet6NarrowIRDTModel/best/TENet6NarrowIRDTModel-best")

    model_args = argparse.Namespace(
        sample_rate=16000, window_size_ms=30, window_stride_ms=10, background_frequency=0.8,
        background_max_volume=0.1, num_mel_bins=64, desired_samples=16000, lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0, num_mfccs=40, timeshift_ms=100, wanted_words=['marvin'],
        dataset_path=DATA_ROOT, split_dir=SPLIT_DIR, steps=3000, batch_size=100, buffer_size=1000,
        workers=1, step_save_checkpoint=500, step_logging=50, max_to_keep=5, trainable_scopes='',
        optimizer='adam', lr_list=[0.002, 0.0005, 0.0001], boundaries=[1200, 2200], momentum=0.9,
        weight_decay=4e-5, input_file='', save_folder='save/TENet6NarrowIRDTModel', checkpoint_path=None,
        arch='TENet6NarrowIRDTModel', dropout=0.0, width_multiplier=1.0, num_classes=3, scope_name='',
        kernel_list=None, irdt_w=64, irdt_chan=chan, irdt_win_per_segm=3, irdt_biti=24, irdt_log=False,
        mod='test', dataset_name='test', window_size_samples=480, window_stride_samples=160, spectrogram_length=98
    )

    results = []
    for ckpt in ckpt_candidates:
        if "TENet6Narrow_MultiSeed" in ckpt:
            tag = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
        elif "MTENet_Fusion" in ckpt:
            tag = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
        else:
            tag = "best_base"
        g = tf.Graph()
        with g.as_default():
            sess = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
            feat_ph = tf.compat.v1.placeholder(tf.float32, [None, 83, 6], name="feat_ph")
            model = models.__dict__[model_args.arch](model_args)
            logits_op = model.build_inference(feat_ph, is_training=True)
            
            sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
            saver = tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.global_variables() if v.name.startswith("TENet")])
            saver.restore(sess, ckpt)

            logits_val = sess.run(logits_op, feed_dict={feat_ph: X_feat})
            preds = np.argmax(logits_val, axis=1)

            acc = float(np.mean(preds == test_labels))
            marvin_recall = float(np.mean(preds[test_labels == 0] == 0))
            bg_rejection = float(np.mean(preds[test_labels == 2] == 2))
            unk_rejection = float(np.mean(preds[test_labels == 1] == 1))
            fa_rate = float(np.mean(preds[test_labels != 0] == 0))

            results.append({
                "model_name": f"TENet6Narrow_{tag}",
                "path": ckpt,
                "params": 15120,
                "accuracy": acc,
                "marvin_recall": marvin_recall,
                "bg_rejection": bg_rejection,
                "unk_rejection": unk_rejection,
                "fa_rate": fa_rate,
                "predictions": preds.tolist(),
            })
            sess.close()

    return results, ckpt_candidates[0]


def main():
    print("=" * 75)
    print(" HEAD-TO-HEAD MATCHED-PARAMETER BENCHMARK (15,120 PARAMS)")
    print("=" * 75)

    test_paths, test_labels = parse_split(os.path.join(SPLIT_DIR, "test.txt"))
    print(f"Loaded test set: {len(test_paths)} samples (Marvin: {np.sum(test_labels==0)}, Unknown: {np.sum(test_labels==1)}, Noise: {np.sum(test_labels==2)})\n")

    # 1. Theoretical MACs
    macs_data = calculate_theoretical_macs()
    print(f"Theoretical Compute Complexity:")
    print(f"  • DS-CNN (Matched, 15,121 params): {macs_data['dscnn_macs']/1e6:.2f} M MACs")
    print(f"  • TENet-6 (Matched, 15,120 params): {macs_data['tenet_macs']/1e6:.2f} M MACs")
    print(f"  • Compute Advantage: TENet requires {macs_data['macs_ratio']:.1f}x FEWER operations\n")

    # 2. Evaluate Models
    dscnn_json_path = "model_training/models/dscnn_matched_test_results.json"
    with open(dscnn_json_path, "r") as f:
        dscnn_payload = json.load(f)
    dscnn_results = dscnn_payload["results"]
    lat_dscnn = dscnn_payload["latency"]
    print(f"Loaded {len(dscnn_results)} DS-CNN matched model test evaluations.")

    tenet_results, last_tenet_ckpt = evaluate_tenet_models(test_paths, test_labels)

    # 3. Aggregate Statistics (mean +/- std)
    dscnn_accs = [r["accuracy"] for r in dscnn_results]
    tenet_accs = [r["accuracy"] for r in tenet_results]
    dscnn_recalls = [r["marvin_recall"] for r in dscnn_results]
    tenet_recalls = [r["marvin_recall"] for r in tenet_results]

    print("\n" + "=" * 75)
    print(" ACCURACY & RECALL COMPARISON (MATCHED PARAMETERS)")
    print("=" * 75)
    print(f"DS-CNN (Matched, 15,121 params, {len(dscnn_results)} runs):")
    print(f"  • Test Accuracy:   {np.mean(dscnn_accs)*100:.2f}% +/- {np.std(dscnn_accs)*100:.2f}% (Best: {np.max(dscnn_accs)*100:.2f}%)")
    print(f"  • Marvin Recall:   {np.mean(dscnn_recalls)*100:.2f}% +/- {np.std(dscnn_recalls)*100:.2f}%")
    print(f"TENet-6 (Matched, 15,120 params, {len(tenet_results)} runs):")
    print(f"  • Test Accuracy:   {np.mean(tenet_accs)*100:.2f}% +/- {np.std(tenet_accs)*100:.2f}% (Best: {np.max(tenet_accs)*100:.2f}%)")
    print(f"  • Marvin Recall:   {np.mean(tenet_recalls)*100:.2f}% +/- {np.std(tenet_recalls)*100:.2f}%")

    # 4. Paired McNemar Test (best DS-CNN vs best TENet)
    best_dscnn = max(dscnn_results, key=lambda x: x["accuracy"])
    best_tenet = max(tenet_results, key=lambda x: x["accuracy"])
    mcnemar = compute_mcnemar_test(
        np.array(best_dscnn["predictions"]),
        np.array(best_tenet["predictions"]),
        test_labels
    )
    print("\n" + "=" * 75)
    print(" PAIRED MCNEMAR STATISTICAL SIGNIFICANCE TEST")
    print("=" * 75)
    tbl = mcnemar["contingency_table"]
    print(f"Contingency Matrix (N={len(test_labels)}):")
    print(f"  • Both Correct (a):               {tbl['both_correct_a']}")
    print(f"  • TENet-6 Correct, DS-CNN Wrong (b): {tbl['model_b_only']}")
    print(f"  • DS-CNN Correct, TENet-6 Wrong (c): {tbl['model_a_only']}")
    print(f"  • Both Wrong (d):                 {tbl['both_wrong_d']}")
    print(f"McNemar Chi-Square (continuity corrected): chi^2 = {mcnemar['chi2']:.4f}")
    print(f"P-Value:                                   p = {mcnemar['p_value']:.4e}")
    print(f"Statistically Significant (alpha=0.05):    {mcnemar['statistically_significant']}")

    # 5. Latency Profiling
    print("\n" + "=" * 75)
    print(" HIGH-PRECISION CPU LATENCY BENCHMARK (500 Runs, Batch Size 1)")
    print("=" * 75)
    print("DS-CNN Latency Profile (500 Runs):")
    print(f"  • Ingestion:             {lat_dscnn['ingestion_ms']:.4f} ms")
    print(f"  • MFCC Feature Extr.:    {lat_dscnn['feature_extraction_ms']:.4f} ms")
    print(f"  • NN Forward Pass:       {lat_dscnn['nn_forward_ms']:.4f} ms")
    print(f"  • Decision:              {lat_dscnn['decision_ms']:.4f} ms")
    print(f"  • Inter-stage Overhead:  {lat_dscnn['inter_stage_overhead_ms']:.4f} ms")
    print(f"  • Total End-to-End:      {lat_dscnn['total_e2e_ms']:.4f} ms (Throughput: {lat_dscnn['throughput_fps']:.1f} fps, RTF: {lat_dscnn['rtf']:.5f})")

    print("Benchmarking TENet-6 Latency...")
    g_bench = tf.Graph()
    with g_bench.as_default():
        sess_b = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
        feat_ph_b = tf.compat.v1.placeholder(tf.float32, [1, 83, 6], name="feat_ph_b")
        model_args_b = argparse.Namespace(
            sample_rate=16000, window_size_ms=30, window_stride_ms=10, background_frequency=0.8,
            background_max_volume=0.1, num_mel_bins=64, desired_samples=16000, lower_edge_hertz=80.0,
            upper_edge_hertz=7600.0, num_mfccs=40, timeshift_ms=100, wanted_words=['marvin'],
            dataset_path=DATA_ROOT, split_dir=SPLIT_DIR, steps=3000, batch_size=1, buffer_size=1000,
            workers=1, step_save_checkpoint=500, step_logging=50, max_to_keep=5, trainable_scopes='',
            optimizer='adam', lr_list=[0.002, 0.0005, 0.0001], boundaries=[1200, 2200], momentum=0.9,
            weight_decay=4e-5, input_file='', save_folder='save/TENet6NarrowIRDTModel', checkpoint_path=None,
            arch='TENet6NarrowIRDTModel', dropout=0.0, width_multiplier=1.0, num_classes=3, scope_name='',
            kernel_list=None, irdt_w=64, irdt_chan=[1, 2, 4, 8, 16, 32], irdt_win_per_segm=3, irdt_biti=24,
            irdt_log=False, mod='test', dataset_name='test', window_size_samples=480, window_stride_samples=160,
            spectrogram_length=98
        )
        m_b = models.__dict__[model_args_b.arch](model_args_b)
        logits_op_b = m_b.build_inference(feat_ph_b, is_training=False)
        sess_b.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        saver_b = tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.global_variables() if v.name.startswith("TENet")])
        saver_b.restore(sess_b, last_tenet_ckpt)
        lat_tenet = benchmark_cpu_latency_tenet(sess_b, logits_op_b, feat_ph_b, num_runs=500)
        sess_b.close()

    print(f"  TENet-6 Total End-to-End: {lat_tenet['total_e2e_ms']:.3f} ms (Throughput: {lat_tenet['throughput_fps']:.1f} fps)")
    speedup = lat_dscnn['total_e2e_ms'] / lat_tenet['total_e2e_ms']
    print(f"  TENet Speedup: {speedup:.2f}x FASTER at matched 15,120 parameter capacity!")

    # Save benchmark JSON
    benchmark_report = {
        "parameters": {"dscnn": 15121, "tenet": 15120},
        "macs": macs_data,
        "accuracy": {
            "dscnn_mean": float(np.mean(dscnn_accs)),
            "dscnn_std": float(np.std(dscnn_accs)),
            "tenet_mean": float(np.mean(tenet_accs)),
            "tenet_std": float(np.std(tenet_accs)),
        },
        "mcnemar": mcnemar,
        "latency_dscnn": lat_dscnn,
        "latency_tenet": lat_tenet,
        "speedup": float(speedup),
    }

    out_file = "save/head_to_head_matched_benchmark_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nSaved complete benchmark report to: {out_file}")


if __name__ == "__main__":
    main()

