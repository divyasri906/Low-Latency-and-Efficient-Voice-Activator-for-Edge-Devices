"""
Train and Benchmark Scaled TENet Models:
- Width 1.00: ~15k params, ~62 KB (Baseline Seed 101)
- Width 1.25: ~23k params, ~90 KB (~20k target)
- Width 1.50: ~31k params, ~122 KB (~30k target)
- Width 1.75: ~41k params, ~160 KB (~150 KB target)

Measures:
- Trainable & Total Parameters
- Model File Size (FP32 weights KB & Checkpoint .data KB)
- Accuracy (Multi-class Test Accuracy & Binary KWS Accuracy)
- Marvin Recall (Sensitivity)
- Marvin Precision
- Time Latency (ms per 1.0s audio inference)
- CPU Utilization (%) during inference
"""

import os
import sys
import time
import json
import psutil
import argparse
import subprocess
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import models
from evaluate_optimized_w5 import load_split_data


def train_model(save_dir, width_multiplier, steps=3000, seed=101,
                dataset_path=r"C:\XboxGames\Dataset\speech_commands_v0.02",
                split_dir="configs/v1_irdt"):
    ckpt_target = os.path.join(save_dir, f"TENet6NarrowIRDTModel-{steps}.index")
    if os.path.exists(ckpt_target):
        print(f"--> [EXISTS] Checkpoint already found at {save_dir}. Skipping training.")
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*75}")
    print(f" TRAINING: TENet6-Narrow (width_multiplier={width_multiplier})")
    print(f" Target: {save_dir} | Steps: {steps} | Seed: {seed}")
    print(f"{'='*75}\n")

    cmd = [
        sys.executable, "-u", "-m", "main",
        "--dataset_path", dataset_path,
        "--split_dir", split_dir,
        "--arch", "TENet6NarrowIRDTModel",
        "--num_classes", "3",
        "--wanted_words", "marvin",
        "--irdt_w", "64",
        "--irdt_chan", "1,2,4,8,16,32",
        "--irdt_win_per_segm", "5",
        "--steps", str(steps),
        "--step_save_checkpoint", "500",
        "--step_logging", "100",
        "--save_folder", save_dir,
        "--lr_list", "0.002,0.0005,0.0001",
        "--boundaries", "1200,2200",
        "--dropout", "0.2",
        "--seed", str(seed),
        "--width_multiplier", str(width_multiplier),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed with code {proc.returncode}")
    print(f"--> [DONE] Training completed in {elapsed:.1f}s ({elapsed/60:.2f} mins).\n")


def evaluate_model(ckpt_path, width_multiplier, X_test, y_test, num_latency_runs=200):
    g = tf.Graph()
    with g.as_default():
        feat_ph = tf.compat.v1.placeholder(tf.float32, [None, 50, 6])
        model_args = argparse.Namespace(
            sample_rate=16000, window_size_ms=30, window_stride_ms=10,
            background_frequency=0.8, background_max_volume=0.1, num_mel_bins=64,
            desired_samples=16000, lower_edge_hertz=80.0, upper_edge_hertz=7600.0,
            num_mfccs=40, timeshift_ms=100, wanted_words=["marvin"],
            dataset_path="", split_dir="", steps=3000,
            batch_size=1, buffer_size=1000, workers=1,
            step_save_checkpoint=500, step_logging=50, max_to_keep=5,
            trainable_scopes="", optimizer="adam", lr_list=[0.002, 0.0005, 0.0001],
            boundaries=[1200, 2200], momentum=0.9, weight_decay=4e-5,
            input_file="", save_folder="", checkpoint_path=None,
            arch="TENet6NarrowIRDTModel", dropout=0.0, width_multiplier=width_multiplier,
            num_classes=3, scope_name="", kernel_list=None, irdt_w=64,
            irdt_chan=[1, 2, 4, 8, 16, 32], irdt_win_per_segm=5, irdt_biti=24,
            irdt_log=False, mod="eval", dataset_name="eval",
            window_size_samples=480, window_stride_samples=160, spectrogram_length=98,
        )

        model = models.TENet6NarrowIRDTModel(model_args)
        logits_op = model.build_inference(feat_ph, is_training=False)
        probs_tensor = tf.nn.softmax(logits_op)

        all_vars = tf.compat.v1.global_variables()
        train_vars = tf.compat.v1.trainable_variables()
        total_params = sum(v.shape.num_elements() for v in all_vars)
        train_params = sum(v.shape.num_elements() for v in train_vars)

        sess = tf.compat.v1.Session()
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, ckpt_path)

        # Batch prediction on test set (400 samples)
        probs = sess.run(probs_tensor, feed_dict={feat_ph: X_test})
        preds = np.argmax(probs, axis=1)

        # Multi-class accuracy
        test_acc = np.mean(preds == y_test) * 100.0

        # Binary KWS Metric (Marvin vs Non-Marvin)
        # Class 0: Marvin, Class 1: Unknown, Class 2: Silence
        y_true_binary = (y_test == 0)
        y_pred_binary = (preds == 0)

        tp = int(np.sum((y_true_binary == True) & (y_pred_binary == True)))
        fn = int(np.sum((y_true_binary == True) & (y_pred_binary == False)))
        fp = int(np.sum((y_true_binary == False) & (y_pred_binary == True)))
        tn = int(np.sum((y_true_binary == False) & (y_pred_binary == False)))

        marvin_recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        marvin_precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        binary_kws_acc = ((tp + tn) / len(y_test)) * 100.0
        fp_rate = (fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0

        # Benchmarking Latency and CPU Utilization
        single_sample = X_test[:1]  # shape (1, 50, 6)

        # Warmup
        for _ in range(20):
            _ = sess.run(probs_tensor, feed_dict={feat_ph: single_sample})

        # Measure Latency (ms)
        latencies_ms = []
        for _ in range(num_latency_runs):
            t_start = time.perf_counter()
            _ = sess.run(probs_tensor, feed_dict={feat_ph: single_sample})
            t_end = time.perf_counter()
            latencies_ms.append((t_end - t_start) * 1000.0)

        median_latency = float(np.median(latencies_ms))
        mean_latency = float(np.mean(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))

        # Measure CPU Utilization (%) during active inference loop
        proc = psutil.Process()
        proc.cpu_percent(interval=None)  # initial probe
        loop_start = time.time()
        count = 0
        while time.time() - loop_start < 1.0:  # sustained 1-second load
            _ = sess.run(probs_tensor, feed_dict={feat_ph: single_sample})
            count += 1
        cpu_util = proc.cpu_percent(interval=None)

        sess.close()

    # Determine file size of checkpoint .data
    data_file = ckpt_path + ".data-00000-of-00001"
    if os.path.exists(data_file):
        model_size_kb = os.path.getsize(data_file) / 1024.0
    else:
        model_size_kb = (total_params * 4) / 1024.0

    raw_weights_kb = (train_params * 4) / 1024.0

    return {
        "width_multiplier": width_multiplier,
        "trainable_params": train_params,
        "total_params": total_params,
        "raw_weights_kb": round(raw_weights_kb, 2),
        "model_file_size_kb": round(model_size_kb, 2),
        "test_acc": round(test_acc, 2),
        "binary_kws_acc": round(binary_kws_acc, 2),
        "recall": round(marvin_recall, 2),
        "precision": round(marvin_precision, 2),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "median_latency_ms": round(median_latency, 2),
        "mean_latency_ms": round(mean_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "cpu_util_pct": round(cpu_util, 1)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=r"C:\XboxGames\Dataset\speech_commands_v0.02")
    parser.add_argument("--split_dir", default="configs/v1_irdt")
    parser.add_argument("--base_save_dir", default="save/tenet_scale_w5")
    args = parser.parse_args()

    # Configuration definitions:
    # 1. wm=1.00: 14.4k trainable, 15.8k total -> ~62 KB
    # 2. wm=1.25: 21.1k trainable, 22.9k total -> ~90 KB
    # 3. wm=1.50: 29.1k trainable, 31.3k total -> ~122 KB
    # 4. wm=1.75: 38.3k trainable, 40.9k total -> ~160 KB (150 KB target)
    models_to_train = [
        {"name": "TENet6-Narrow-1.00x", "wm": 1.00,
         "ckpt": "save/ensemble_w5/seed_101/TENet6NarrowIRDTModel-3000", "train": False},
        {"name": "TENet6-Narrow-1.25x (~20k)", "wm": 1.25,
         "ckpt": os.path.join(args.base_save_dir, "wm_125", "TENet6NarrowIRDTModel-3000"), "train": True},
        {"name": "TENet6-Narrow-1.50x (~30k)", "wm": 1.50,
         "ckpt": os.path.join(args.base_save_dir, "wm_150", "TENet6NarrowIRDTModel-3000"), "train": True},
        {"name": "TENet6-Narrow-1.75x (~150KB)", "wm": 1.75,
         "ckpt": os.path.join(args.base_save_dir, "wm_175", "TENet6NarrowIRDTModel-3000"), "train": True},
    ]

    # Train scaled models
    for m in models_to_train:
        if m["train"]:
            save_dir = os.path.dirname(m["ckpt"])
            train_model(save_dir, m["wm"], steps=3000, seed=101,
                        dataset_path=args.dataset_path, split_dir=args.split_dir)

    # Load test data
    test_file = os.path.join(args.split_dir, "test.txt")
    print(f"\nLoading test dataset features from {test_file}...")
    X_test, y_test, words_test, paths_test = load_split_data(args.dataset_path, test_file, w=5)

    print(f"\nEvaluating all 4 models on {len(X_test)} test samples...")
    results = []
    for m in models_to_train:
        print(f"--> Evaluating {m['name']} ({m['ckpt']})...")
        res = evaluate_model(m["ckpt"], m["wm"], X_test, y_test)
        res["model_name"] = m["name"]
        results.append(res)

    # Print Clean Comparison Table
    print("\n" + "=" * 115)
    print(" SCALED TENET MODEL PARAMETER & PERFORMANCE BENCHMARK")
    print("=" * 115)
    header = (f"{'Model Name':<28} | {'Params (Train/Tot)':<18} | {'Size (KB)':<10} | "
              f"{'Test Acc':<9} | {'Recall':<7} | {'Precision':<9} | {'Latency':<8} | {'CPU Util':<8}")
    print(header)
    print("-" * 115)
    for r in results:
        param_str = f"{r['trainable_params']:,} / {r['total_params']:,}"
        size_str = f"{r['model_file_size_kb']} KB"
        acc_str = f"{r['test_acc']:.2f}%"
        rec_str = f"{r['recall']:.1f}%"
        prec_str = f"{r['precision']:.1f}%"
        lat_str = f"{r['median_latency_ms']:.2f} ms"
        cpu_str = f"{r['cpu_util_pct']:.1f}%"
        print(f"{r['model_name']:<28} | {param_str:<18} | {size_str:<10} | "
              f"{acc_str:<9} | {rec_str:<7} | {prec_str:<9} | {lat_str:<8} | {cpu_str:<8}")
    print("=" * 115)

    # Save to JSON
    out_json = "scaled_tenet_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed benchmark metrics to {out_json}")


if __name__ == "__main__":
    main()
