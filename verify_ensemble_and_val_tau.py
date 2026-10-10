"""
Independent Audit & Validation-Tuned Threshold Confirmation for 3-Seed Ensemble.

1. Confirms the ensemble is NOT just Seed 101:
   - Full pairwise agreement analysis (Seed 42, 101, 2024, Ensemble).
   - Exact identification of samples where Ensemble differs from Seed 101 alone.
2. Complete tau sweep on VALID.TXT:
   - Evaluates Soft Probability Ensemble across tau in [0.35, 0.98].
   - Selects optimal validation operating points:
     (a) Acc-maximizing tau*
     (b) Strict Zero/Near-Zero FA tau*
3. Unbiased Confirmation on TEST.TXT using the validation-selected tau*:
   - Evaluates test performance at both validation operating points.
   - Verifies whether near-0% FA with low-90s recall holds up out-of-sample.
   - Verifies exact integer error reconciliation.
"""

import os
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from evaluate_optimized_w5 import load_split_data, evaluate_metrics
from run_ensemble_training_and_eval import get_model_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=r"C:\XboxGames\Dataset\speech_commands_v0.02")
    parser.add_argument("--split_dir", default="configs/v1_irdt")
    parser.add_argument("--base_save_dir", default="save/ensemble_w5")
    args = parser.parse_args()

    ckpt_42 = os.path.join(args.base_save_dir, "seed_42", "TENet6NarrowIRDTModel-3000")
    ckpt_101 = os.path.join(args.base_save_dir, "seed_101", "TENet6NarrowIRDTModel-3000")
    ckpt_2024 = os.path.join(args.base_save_dir, "seed_2024", "TENet6NarrowIRDTModel-3000")

    print(f"\n{'='*75}")
    print(f" 3-SEED ENSEMBLE AUDIT & VALIDATION-CALIBRATED CONFIRMATION")
    print(f"{'='*75}\n")

    # 1. Load Data
    valid_file = os.path.join(args.split_dir, "valid.txt")
    test_file = os.path.join(args.split_dir, "test.txt")
    print("[1/4] Loading features for valid.txt and test.txt...")
    X_val, y_val, words_val, paths_val = load_split_data(args.dataset_path, valid_file, w=5)
    X_test, y_test, words_test, paths_test = load_split_data(args.dataset_path, test_file, w=5)

    # 2. Get probabilities for all 3 models on both splits
    print("[2/4] Running inference for all 3 checkpoints under is_training=False...")
    print("  Evaluating on VALID split...")
    p_val_42 = get_model_probs(X_val, ckpt_42, args.dataset_path, args.split_dir)
    p_val_101 = get_model_probs(X_val, ckpt_101, args.dataset_path, args.split_dir)
    p_val_2024 = get_model_probs(X_val, ckpt_2024, args.dataset_path, args.split_dir)
    p_val_ens = (p_val_42 + p_val_101 + p_val_2024) / 3.0

    print("  Evaluating on TEST split...")
    p_test_42 = get_model_probs(X_test, ckpt_42, args.dataset_path, args.split_dir)
    p_test_101 = get_model_probs(X_test, ckpt_101, args.dataset_path, args.split_dir)
    p_test_2024 = get_model_probs(X_test, ckpt_2024, args.dataset_path, args.split_dir)
    p_test_ens = (p_test_42 + p_test_101 + p_test_2024) / 3.0

    # 3. Check 1: Is the ensemble secretly just Seed 101?
    print(f"\n{'='*75}")
    print(f" CHECK 1: INDEPENDENCE AUDIT (Is Ensemble secretly just Seed 101?)")
    print(f"{'='*75}")
    preds_test_42 = np.argmax(p_test_42, axis=1)
    preds_test_101 = np.argmax(p_test_101, axis=1)
    preds_test_2024 = np.argmax(p_test_2024, axis=1)
    preds_test_ens = np.argmax(p_test_ens, axis=1)

    is_identical_101 = np.array_equal(preds_test_ens, preds_test_101)
    agree_42_101 = np.mean(preds_test_42 == preds_test_101) * 100.0
    agree_101_2024 = np.mean(preds_test_101 == preds_test_2024) * 100.0
    agree_42_2024 = np.mean(preds_test_42 == preds_test_2024) * 100.0
    agree_ens_101 = np.mean(preds_test_ens == preds_test_101) * 100.0
    agree_ens_42 = np.mean(preds_test_ens == preds_test_42) * 100.0
    agree_ens_2024 = np.mean(preds_test_ens == preds_test_2024) * 100.0

    print(f"  Are Ensemble and Seed 101 predictions identical? {is_identical_101}")
    print(f"  Pairwise Agreement Matrix (Test Set, 400 samples):")
    print(f"    - Seed 42 vs Seed 101:   {agree_42_101:.2f}% ({np.sum(preds_test_42 == preds_test_101)}/400)")
    print(f"    - Seed 101 vs Seed 2024: {agree_101_2024:.2f}% ({np.sum(preds_test_101 == preds_test_2024)}/400)")
    print(f"    - Seed 42 vs Seed 2024:  {agree_42_2024:.2f}% ({np.sum(preds_test_42 == preds_test_2024)}/400)")
    print(f"    - Ensemble vs Seed 101:  {agree_ens_101:.2f}% ({np.sum(preds_test_ens == preds_test_101)}/400)")
    print(f"    - Ensemble vs Seed 42:   {agree_ens_42:.2f}% ({np.sum(preds_test_ens == preds_test_42)}/400)")
    print(f"    - Ensemble vs Seed 2024: {agree_ens_2024:.2f}% ({np.sum(preds_test_ens == preds_test_2024)}/400)")

    diff_indices = np.where(preds_test_ens != preds_test_101)[0]
    print(f"\n  Samples where Ensemble differs from Seed 101 alone ({len(diff_indices)} instances):")
    for idx in diff_indices:
        print(f"    Index {idx:3d}: Word={words_test[idx]:10s} TrueLabel={y_test[idx]} | S101_pred={preds_test_101[idx]} (P={p_test_101[idx, preds_test_101[idx]]:.4f}) | Ensemble_pred={preds_test_ens[idx]} (P={p_test_ens[idx, preds_test_ens[idx]]:.4f}) | S42={preds_test_42[idx]}, S2024={preds_test_2024[idx]}")

    # 4. Check 2: Threshold sweep on VALID.TXT
    print(f"\n{'='*75}")
    print(f" CHECK 2: THRESHOLD SWEEP ON VALID.TXT (400 Samples)")
    print(f"{'='*75}")
    thresholds = [0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98]
    val_metrics = []
    print(f"  {'tau':5s} | {'Val Acc':8s} | {'Recall':8s} | {'FAs (Unk, Noise)':18s} | {'FA Rate':8s} | {'Misses':6s}")
    print(f"  {'-'*65}")
    for tau in thresholds:
        m = evaluate_metrics(p_val_ens, y_val, tau=tau)
        val_metrics.append((tau, m))
        print(f"  {tau:<5.2f} | {m['acc']:>6.2f}% | {m['recall']:>6.2f}% | {m['fa_unk']:>3d}, {m['fa_noise']:>3d}            | {m['fa_rate']:>6.2f}% | {m['misses']:>6d}")

    # Select operating points on validation set:
    # Operating Point A: Highest Accuracy on Valid
    tau_acc_star, val_m_acc = max(val_metrics, key=lambda x: (x[1]["acc"], x[1]["recall"], -x[1]["total_fa"]))
    # Operating Point B: Zero or Lowest False Alarm Rate on Valid with >= 85% recall
    zero_fa_candidates = [x for x in val_metrics if x[1]["total_fa"] == 0]
    if zero_fa_candidates:
        tau_zero_fa, val_m_zero = max(zero_fa_candidates, key=lambda x: x[1]["recall"])
    else:
        tau_zero_fa, val_m_zero = min(val_metrics, key=lambda x: (x[1]["total_fa"], -x[1]["recall"]))

    print(f"\n  Validation-Selected Operating Points:")
    print(f"    [A] Max-Accuracy Operating Point:       tau* = {tau_acc_star:.2f} (Val Acc: {val_m_acc['acc']:.2f}%, Recall: {val_m_acc['recall']:.2f}%, FAs: {val_m_acc['total_fa']}, FA Rate: {val_m_acc['fa_rate']:.2f}%)")
    print(f"    [B] Strict Zero/Min-FA Operating Point: tau* = {tau_zero_fa:.2f} (Val Acc: {val_m_zero['acc']:.2f}%, Recall: {val_m_zero['recall']:.2f}%, FAs: {val_m_zero['total_fa']}, FA Rate: {val_m_zero['fa_rate']:.2f}%)")

    # 5. Check 3: Confirm once on TEST.TXT using validation-selected thresholds
    print(f"\n{'='*75}")
    print(f" CHECK 3: UNBIASED EVALUATION ON TEST.TXT USING VALID-SELECTED TAU*")
    print(f"{'='*75}")

    # Evaluate at Argmax (tau=0.50)
    test_argmax = evaluate_metrics(p_test_ens, y_test, tau=0.50)
    # Evaluate at Operating Point A
    test_acc_star = evaluate_metrics(p_test_ens, y_test, tau=tau_acc_star)
    # Evaluate at Operating Point B
    test_zero_fa = evaluate_metrics(p_test_ens, y_test, tau=tau_zero_fa)

    print(f"\n  --- OPERATING POINT A: Validation Max-Accuracy (tau = {tau_acc_star:.2f}) ---")
    print(f"    Test Accuracy:        {test_acc_star['acc']:.2f}% ({test_acc_star['correct']}/400)")
    print(f"    Marvin Recall:        {test_acc_star['recall']:.2f}% ({100 - test_acc_star['misses']}/100)")
    print(f"    Missed Marvins:       {test_acc_star['misses']}")
    print(f"    False Alarms:         {test_acc_star['total_fa']} / 300 non-Marvins ({test_acc_star['fa_rate']:.2f}%)")
    print(f"      - Unknown Speech:   {test_acc_star['fa_unk']}")
    print(f"      - Background Noise: {test_acc_star['fa_noise']}")
    print(f"    Error Reconciliation: {test_acc_star['misses']} (miss) + {test_acc_star['total_fa']} (FA) + {test_acc_star['other_errors']} (other) = {400 - test_acc_star['correct']} errors. (Exact match: {400 - test_acc_star['correct'] == test_acc_star['misses'] + test_acc_star['total_fa'] + test_acc_star['other_errors']})")

    print(f"\n  --- OPERATING POINT B: Validation Strict Zero/Min-FA (tau = {tau_zero_fa:.2f}) ---")
    print(f"    Test Accuracy:        {test_zero_fa['acc']:.2f}% ({test_zero_fa['correct']}/400)")
    print(f"    Marvin Recall:        {test_zero_fa['recall']:.2f}% ({100 - test_zero_fa['misses']}/100)")
    print(f"    Missed Marvins:       {test_zero_fa['misses']}")
    print(f"    False Alarms:         {test_zero_fa['total_fa']} / 300 non-Marvins ({test_zero_fa['fa_rate']:.2f}%)")
    print(f"      - Unknown Speech:   {test_zero_fa['fa_unk']}")
    print(f"      - Background Noise: {test_zero_fa['fa_noise']}")
    print(f"    Error Reconciliation: {test_zero_fa['misses']} (miss) + {test_zero_fa['total_fa']} (FA) + {test_zero_fa['other_errors']} (other) = {400 - test_zero_fa['correct']} errors. (Exact match: {400 - test_zero_fa['correct'] == test_zero_fa['misses'] + test_zero_fa['total_fa'] + test_zero_fa['other_errors']})")

    print(f"\n  --- SUMMARY OF GENERALIZATION ACROSS SPLITS ---")
    print(f"  Metric              | Valid (tau={tau_zero_fa:.2f}) | Test (tau={tau_zero_fa:.2f})  | Difference")
    print(f"  --------------------+-----------------------+----------------------+-----------")
    print(f"  Overall Accuracy    | {val_m_zero['acc']:>6.2f}%                | {test_zero_fa['acc']:>6.2f}%               | {test_zero_fa['acc'] - val_m_zero['acc']:>+6.2f}%")
    print(f"  Marvin Recall       | {val_m_zero['recall']:>6.2f}%                | {test_zero_fa['recall']:>6.2f}%               | {test_zero_fa['recall'] - val_m_zero['recall']:>+6.2f}%")
    print(f"  False Alarm Rate    | {val_m_zero['fa_rate']:>6.2f}%                | {test_zero_fa['fa_rate']:>6.2f}%               | {test_zero_fa['fa_rate'] - val_m_zero['fa_rate']:>+6.2f}%")
    print(f"  Noise False Alarms  | {val_m_zero['fa_noise']:>6d}                | {test_zero_fa['fa_noise']:>6d}               | {test_zero_fa['fa_noise'] - val_m_zero['fa_noise']:>+6d}")


if __name__ == "__main__":
    main()
