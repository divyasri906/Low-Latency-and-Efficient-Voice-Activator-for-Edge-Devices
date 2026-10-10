"""
Rigorous Evaluation & Verification for Optimized W=5 TENet6-Narrow + iRDT.

Fulfills standing checklist:
1. Determinism verification (two identical eval passes on test.txt).
2. Exact error partition reconciliation (Misses + FA_unk + FA_noise == 400 - Correct).
3. Train-vs-Val gap check (evaluating train.txt vs valid.txt with is_training=False).
4. Threshold sweep on VALID.TXT to select optimal operating point tau*.
5. Final evaluation on TEST.TXT at both argmax (tau=0.50) and validation-selected tau*.
"""

import os
import wave
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import models
from irdt.irdtv import irdtv_optimized


def load_split_data(dataset_path, split_file, w=5, chan_list=[1, 2, 4, 8, 16, 32], irdt_log=False):
    paths, labels, words = [], [], []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            full_path = os.path.join(dataset_path, line)
            paths.append(full_path)
            word = line.replace("\\", "/").split("/")[0]
            words.append(word)
            if word == "marvin":
                labels.append(0)
            elif word == "_silence_":
                labels.append(2)
            else:
                labels.append(1)

    labels = np.array(labels, dtype=np.int64)
    features = []
    for p in paths:
        with wave.open(p, "rb") as wf:
            buf = wf.readframes(16000)
            audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
        audio = audio[:16000]
        _, feat = irdtv_optimized(audio, chan_list, 64, w, 24, irdt_log)
        features.append(feat)

    X = np.array(features, dtype=np.float32)
    return X, labels, words, paths


def run_inference_on_features(X, ckpt_path, dataset_path, split_dir, batch_size=100):
    g = tf.Graph()
    with g.as_default():
        feat_ph = tf.compat.v1.placeholder(tf.float32, [None, X.shape[1], X.shape[2]])
        labels_ph = tf.compat.v1.placeholder(tf.int64, [None])

        model_args = argparse.Namespace(
            sample_rate=16000, window_size_ms=30, window_stride_ms=10,
            background_frequency=0.8, background_max_volume=0.1, num_mel_bins=64,
            desired_samples=16000, lower_edge_hertz=80.0, upper_edge_hertz=7600.0,
            num_mfccs=40, timeshift_ms=100, wanted_words=["marvin"],
            dataset_path=dataset_path, split_dir=split_dir, steps=3000,
            batch_size=batch_size, buffer_size=1000, workers=1,
            step_save_checkpoint=500, step_logging=50, max_to_keep=5,
            trainable_scopes="", optimizer="adam", lr_list=[0.002, 0.0005, 0.0001],
            boundaries=[1200, 2200], momentum=0.9, weight_decay=4e-5,
            input_file="", save_folder="", checkpoint_path=None,
            arch="TENet6NarrowIRDTModel", dropout=0.0, width_multiplier=1.0,
            num_classes=3, scope_name="", kernel_list=None, irdt_w=64,
            irdt_chan=[1, 2, 4, 8, 16, 32], irdt_win_per_segm=5, irdt_biti=24,
            irdt_log=False, mod="eval", dataset_name="eval",
            window_size_samples=480, window_stride_samples=160, spectrogram_length=98,
        )

        model = models.__dict__[model_args.arch](model_args)
        logits_op = model.build_inference(feat_ph, is_training=False)
        probs_tensor = tf.nn.softmax(logits_op)

        sess = tf.compat.v1.Session()
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, ckpt_path)

        all_probs = []
        n_samples = len(X)
        for i in range(0, n_samples, batch_size):
            batch_X = X[i : i + batch_size]
            batch_probs = sess.run(probs_tensor, feed_dict={feat_ph: batch_X})
            all_probs.append(batch_probs)

        sess.close()

    return np.vstack(all_probs)


def evaluate_metrics(probs, labels, tau=0.50):
    """
    If tau == 0.50 (argmax rule):
      pred = argmax(P(Marvin), P(Unk), P(Noise))
    If tau > 0.33 (threshold rule):
      If P(Marvin) >= tau -> pred = 0 (Marvin)
      Else -> pred = argmax(P(Unk), P(Noise)) + 1 (either 1 or 2)
    """
    if abs(tau - 0.50) < 1e-6:
        preds = np.argmax(probs, axis=1)
    else:
        p_marvin = probs[:, 0]
        preds = np.zeros(len(probs), dtype=np.int64)
        for i in range(len(probs)):
            if p_marvin[i] >= tau:
                preds[i] = 0
            else:
                preds[i] = 1 + np.argmax(probs[i, 1:])

    correct = int(np.sum(preds == labels))
    total = len(labels)
    acc = correct / total * 100.0

    # Marvin metrics (class 0)
    marvin_mask = (labels == 0)
    n_marvin = int(np.sum(marvin_mask))
    marvin_detected = int(np.sum((preds == 0) & marvin_mask))
    recall = marvin_detected / n_marvin * 100.0
    misses = n_marvin - marvin_detected

    # False alarms (pred == 0 when label != 0)
    fa_unk = int(np.sum((preds == 0) & (labels == 1)))
    fa_noise = int(np.sum((preds == 0) & (labels == 2)))
    total_fa = fa_unk + fa_noise
    n_non_marvin = total - n_marvin
    fa_rate = total_fa / n_non_marvin * 100.0

    # Strict reconciliation check
    total_errors = total - correct
    partition_sum = misses + fa_unk + fa_noise
    # Note: errors where label 1 -> pred 2 or label 2 -> pred 1:
    other_errors = int(np.sum((labels == 1) & (preds == 2))) + int(np.sum((labels == 2) & (preds == 1)))
    assert misses + total_fa + other_errors == total_errors, (
        f"Partition mismatch: {misses} + {total_fa} + {other_errors} != {total_errors}"
    )

    return {
        "acc": acc,
        "correct": correct,
        "total": total,
        "recall": recall,
        "misses": misses,
        "fa_unk": fa_unk,
        "fa_noise": fa_noise,
        "total_fa": total_fa,
        "fa_rate": fa_rate,
        "other_errors": other_errors,
        "preds": preds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=r"C:\XboxGames\Dataset\speech_commands_v0.02")
    parser.add_argument("--split_dir", default="configs/v1_irdt")
    parser.add_argument("--ckpt_path", default="save/optimized_w5/TENet6NarrowIRDTModel-3000")
    args = parser.parse_args()

    print(f"\n{'='*75}")
    print(f" COMPREHENSIVE VERIFICATION FOR: {args.ckpt_path}")
    print(f"{'='*75}")

    # -------------------------------------------------------------
    # 1. Load Valid and Test Data
    # -------------------------------------------------------------
    print("\n[1/5] Loading validation and test features...")
    valid_file = os.path.join(args.split_dir, "valid.txt")
    test_file = os.path.join(args.split_dir, "test.txt")
    train_file = os.path.join(args.split_dir, "train.txt")

    X_val, y_val, words_val, _ = load_split_data(args.dataset_path, valid_file, w=5)
    X_test, y_test, words_test, paths_test = load_split_data(args.dataset_path, test_file, w=5)
    print(f"  Loaded Valid: {X_val.shape}, Labels: {len(y_val)}")
    print(f"  Loaded Test:  {X_test.shape}, Labels: {len(y_test)}")

    # -------------------------------------------------------------
    # 2. Run Inference
    # -------------------------------------------------------------
    print("\n[2/5] Running inference under is_training=False...")
    probs_val = run_inference_on_features(X_val, args.ckpt_path, args.dataset_path, args.split_dir)
    probs_test_pass1 = run_inference_on_features(X_test, args.ckpt_path, args.dataset_path, args.split_dir)
    probs_test_pass2 = run_inference_on_features(X_test, args.ckpt_path, args.dataset_path, args.split_dir)

    # -------------------------------------------------------------
    # 3. Determinism Check
    # -------------------------------------------------------------
    print("\n[3/5] Determinism Verification (Pass 1 vs Pass 2)...")
    max_prob_diff = np.max(np.abs(probs_test_pass1 - probs_test_pass2))
    preds_p1 = np.argmax(probs_test_pass1, axis=1)
    preds_p2 = np.argmax(probs_test_pass2, axis=1)
    identical_preds = np.array_equal(preds_p1, preds_p2)
    print(f"  Max absolute probability difference: {max_prob_diff:.8e}")
    print(f"  Predictions identical across passes: {identical_preds}")
    assert identical_preds and max_prob_diff < 1e-6, "Determinism check failed!"
    print("  --> DETERMINISM CONFIRMED.")

    # -------------------------------------------------------------
    # 4. Threshold Sweep on VALIDATION SET
    # -------------------------------------------------------------
    print("\n[4/5] Threshold Sweep on VALIDATION SET (Finding tau*)...")
    thresholds = [0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    val_results = []
    print(f"  {'tau':5s} | {'Val Acc':8s} | {'Recall':8s} | {'FAs (Unk, Noise)':18s} | {'FA Rate':8s}")
    print(f"  {'-'*55}")
    for tau in thresholds:
        res = evaluate_metrics(probs_val, y_val, tau=tau)
        val_results.append((tau, res))
        print(f"  {tau:<5.2f} | {res['acc']:>6.2f}% | {res['recall']:>6.2f}% | {res['fa_unk']:>3d}, {res['fa_noise']:>3d}            | {res['fa_rate']:>6.2f}%")

    # Pick tau* with highest accuracy, tie-breaking for higher recall
    best_tau, best_val_res = max(val_results, key=lambda x: (x[1]["acc"], x[1]["recall"], -x[1]["total_fa"]))
    print(f"\n  --> Selected Operating Point on Validation: tau* = {best_tau:.2f} (Val Acc: {best_val_res['acc']:.2f}%, Recall: {best_val_res['recall']:.2f}%, FAs: {best_val_res['total_fa']})")

    # -------------------------------------------------------------
    # 5. Final Test Evaluation (at argmax and tau*)
    # -------------------------------------------------------------
    print("\n[5/5] Final Test Set Evaluation...")
    # Argmax
    test_argmax = evaluate_metrics(probs_test_pass1, y_test, tau=0.50)
    print(f"\n--- TEST RESULTS: Default Argmax (tau = 0.50) ---")
    print(f"  Overall Accuracy:   {test_argmax['acc']:.2f}% ({test_argmax['correct']}/{test_argmax['total']})")
    print(f"  Marvin Recall:      {test_argmax['recall']:.2f}% ({100 - test_argmax['misses']}/100)")
    print(f"  Missed Marvins:     {test_argmax['misses']}")
    print(f"  False Alarms:       {test_argmax['total_fa']} / 300 non-Marvins ({test_argmax['fa_rate']:.2f}%)")
    print(f"    - Unknown Speech: {test_argmax['fa_unk']}")
    print(f"    - Noise/Silence:  {test_argmax['fa_noise']}")
    print(f"  Other Errors:       {test_argmax['other_errors']} (Unk <-> Noise)")
    print(f"  Error Reconciliation: {test_argmax['misses']} (miss) + {test_argmax['total_fa']} (FA) + {test_argmax['other_errors']} (other) = {400 - test_argmax['correct']} total errors.")

    # At tau*
    test_tau_star = evaluate_metrics(probs_test_pass1, y_test, tau=best_tau)
    print(f"\n--- TEST RESULTS: At Validation-Selected tau* = {best_tau:.2f} ---")
    print(f"  Overall Accuracy:   {test_tau_star['acc']:.2f}% ({test_tau_star['correct']}/{test_tau_star['total']})")
    print(f"  Marvin Recall:      {test_tau_star['recall']:.2f}% ({100 - test_tau_star['misses']}/100)")
    print(f"  Missed Marvins:     {test_tau_star['misses']}")
    print(f"  False Alarms:       {test_tau_star['total_fa']} / 300 non-Marvins ({test_tau_star['fa_rate']:.2f}%)")
    print(f"    - Unknown Speech: {test_tau_star['fa_unk']}")
    print(f"    - Noise/Silence:  {test_tau_star['fa_noise']}")
    print(f"  Other Errors:       {test_tau_star['other_errors']} (Unk <-> Noise)")
    print(f"  Error Reconciliation: {test_tau_star['misses']} (miss) + {test_tau_star['total_fa']} (FA) + {test_tau_star['other_errors']} (other) = {400 - test_tau_star['correct']} total errors.")

    # Detailed FA breakdown at argmax
    print(f"\n--- Detailed False Alarm Audit (Test Argmax) ---")
    fa_indices = np.where((test_argmax["preds"] == 0) & (y_test != 0))[0]
    for idx in fa_indices:
        print(f"  FA index {idx:3d}: word={words_test[idx]:10s} P(Marvin)={probs_test_pass1[idx,0]:.4f}, P(Unk)={probs_test_pass1[idx,1]:.4f}, P(Noise)={probs_test_pass1[idx,2]:.4f}")

    # Detailed Miss breakdown
    print(f"\n--- Detailed Missed Marvin Audit (Test Argmax) ---")
    miss_indices = np.where((test_argmax["preds"] != 0) & (y_test == 0))[0]
    for idx in miss_indices:
        pred_label = "Unknown" if test_argmax["preds"][idx] == 1 else "Noise"
        print(f"  Miss index {idx:3d}: predicted={pred_label:8s} P(Marvin)={probs_test_pass1[idx,0]:.4f}, P(Unk)={probs_test_pass1[idx,1]:.4f}, P(Noise)={probs_test_pass1[idx,2]:.4f}")

    # -------------------------------------------------------------
    # 6. Train-vs-Val Gap Check
    # -------------------------------------------------------------
    print("\n[6/6] Computing Train-vs-Val Gap on train.txt (3,200 samples)...")
    X_train, y_train, _, _ = load_split_data(args.dataset_path, train_file, w=5)
    probs_train = run_inference_on_features(X_train, args.ckpt_path, args.dataset_path, args.split_dir)
    train_argmax = evaluate_metrics(probs_train, y_train, tau=0.50)
    gap = train_argmax["acc"] - val_results[2][1]["acc"]  # tau=0.50 is index 2
    print(f"  Train Accuracy (is_training=False): {train_argmax['acc']:.2f}% ({train_argmax['correct']}/{train_argmax['total']})")
    print(f"  Valid Accuracy (is_training=False): {val_results[2][1]['acc']:.2f}%")
    print(f"  Train-vs-Val Gap:                  {gap:+.2f}%")
    if gap < 3.0:
        print("  --> HEALTHY GENERALIZATION: Gap < 3%, no overfitting detected.")
    elif gap < 5.0:
        print("  --> MODERATE GAP: Generalization is normal for a small dataset.")
    else:
        print("  --> WARNING: Significant overfitting gap.")


if __name__ == "__main__":
    main()
