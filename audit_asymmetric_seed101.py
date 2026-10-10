"""
Head-to-head Audit: Baseline Seed 101 vs Asymmetric Loss Seed 101.

Measures:
1. False Alarm Cluster Suppression:
   Tracks dog, happy, learn, left, right, down to see if their P(Marvin) drops below 0.60.
2. Marvin Under-Confidence Check:
   Examines whether true Marvin confidence (>= 0.95 group) is preserved or dragged down.
3. Test Set Threshold Sweep & Accuracy/Recall/FA table.
4. Error Reconciliation Check (Misses + FAs = Errors).
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
    parser.add_argument("--baseline_ckpt", default="save/ensemble_w5/seed_101/TENet6NarrowIRDTModel-3000")
    parser.add_argument("--asymm_ckpt", default="save/asymmetric_w5/seed_101/TENet6NarrowIRDTModel-3000")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" HEAD-TO-HEAD AUDIT: BASELINE SEED 101 vs ASYMMETRIC LOSS SEED 101")
    print("=" * 80 + "\n")

    test_file = os.path.join(args.split_dir, "test.txt")
    valid_file = os.path.join(args.split_dir, "valid.txt")

    print("[1/3] Loading test and validation features...")
    X_test, y_test, words_test, paths_test = load_split_data(args.dataset_path, test_file, w=5)
    X_val, y_val, words_val, _ = load_split_data(args.dataset_path, valid_file, w=5)

    print("[2/3] Running inference for Baseline and Asymmetric models...")
    p_base_test = get_model_probs(X_test, args.baseline_ckpt, args.dataset_path, args.split_dir)
    p_asymm_test = get_model_probs(X_test, args.asymm_ckpt, args.dataset_path, args.split_dir)
    p_asymm_val = get_model_probs(X_val, args.asymm_ckpt, args.dataset_path, args.split_dir)

    # -------------------------------------------------------------
    # 1. FALSE ALARM CLUSTER SUPPRESSION AUDIT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 1. FALSE ALARM CLUSTER SUPPRESSION AUDIT (Top Non-Marvins in Test Set)")
    print("=" * 80)
    non_marvin_idx = np.where(y_test != 0)[0]
    p_nm_base = p_base_test[non_marvin_idx, 0]
    p_nm_asymm = p_asymm_test[non_marvin_idx, 0]

    # Sort by baseline probability to see how the highest FAs moved
    sort_order = np.argsort(p_nm_base)[::-1]
    print(f" {'Rank':4s} | {'Idx':3s} | {'Word':10s} | {'P_base(Marvin)':15s} | {'P_asymm(Marvin)':16s} | {'Delta':8s} | {'Suppressed?':12s}")
    print(" " + "-" * 78)
    for rank, i in enumerate(sort_order[:15]):
        real_idx = non_marvin_idx[i]
        pb = p_nm_base[i]
        pa = p_nm_asymm[i]
        delta = pa - pb
        suppressed = "YES (dropped)" if pa < pb else "NO (increased)"
        if pa < 0.50 and pb >= 0.50:
            suppressed = "FIXED (<0.50)"
        w = words_test[real_idx]
        print(f"  {rank+1:<4d} | {real_idx:<3d} | {w:<10s} | {pb:>14.4f}  | {pa:>15.4f}  | {delta:>+7.4f} | {suppressed}")

    # Top overall in asymmetric model
    sort_asymm = np.argsort(p_nm_asymm)[::-1]
    print(f"\n  Top 5 Highest False Alarm scores in Asymmetric Model:")
    for rank, i in enumerate(sort_asymm[:5]):
        real_idx = non_marvin_idx[i]
        print(f"    Rank {rank+1}: Index {real_idx:3d} ({words_test[real_idx]:10s}): P_asymm = {p_nm_asymm[i]:.4f} (was {p_nm_base[i]:.4f} in baseline)")

    # -------------------------------------------------------------
    # 2. MARVIN UNDER-CONFIDENCE CHECK
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 2. MARVIN UNDER-CONFIDENCE CHECK (Did True Marvins Get Dragged Down?)")
    print("=" * 80)
    marvin_idx = np.where(y_test == 0)[0]
    p_m_base = p_base_test[marvin_idx, 0]
    p_m_asymm = p_asymm_test[marvin_idx, 0]

    base_ge_95 = int(np.sum(p_m_base >= 0.95))
    asymm_ge_95 = int(np.sum(p_m_asymm >= 0.95))
    base_ge_80 = int(np.sum(p_m_base >= 0.80))
    asymm_ge_80 = int(np.sum(p_m_asymm >= 0.80))
    base_ge_50 = int(np.sum(p_m_base >= 0.50))
    asymm_ge_50 = int(np.sum(p_m_asymm >= 0.50))

    print(f"  Confidence Tiers (out of 100 true Marvins):")
    print(f"    - P(Marvin) >= 0.95: Baseline = {base_ge_95:2d} | Asymmetric = {asymm_ge_95:2d} (Change: {asymm_ge_95 - base_ge_95:+d})")
    print(f"    - P(Marvin) >= 0.80: Baseline = {base_ge_80:2d} | Asymmetric = {asymm_ge_80:2d} (Change: {asymm_ge_80 - base_ge_80:+d})")
    print(f"    - P(Marvin) >= 0.50: Baseline = {base_ge_50:2d} | Asymmetric = {asymm_ge_50:2d} (Change: {asymm_ge_50 - base_ge_50:+d})")
    print(f"    - Mean Marvin Conf:  Baseline = {np.mean(p_m_base)*100:.2f}% | Asymmetric = {np.mean(p_m_asymm)*100:.2f}%")

    if asymm_ge_95 < base_ge_95 - 5:
        print("  --> WARNING: Noticeable under-confidence detected on true Marvins! lambda_fp may be too strong.")
    else:
        print("  --> HEALTHY: True Marvin confidence remains robust.")

    # -------------------------------------------------------------
    # 3. THRESHOLD SWEEP COMPARISON ON TEST.TXT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 3. TEST SET PERFORMANCE ACROSS THRESHOLDS (Baseline vs Asymmetric)")
    print("=" * 80)
    thresholds = [0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    print(f"  {'tau':5s} | {'Base Acc':8s} | {'Asym Acc':8s} | {'Base Rec':8s} | {'Asym Rec':8s} | {'Base FAs':8s} | {'Asym FAs':8s}")
    print("  " + "-" * 70)
    for tau in thresholds:
        mb = evaluate_metrics(p_base_test, y_test, tau=tau)
        ma = evaluate_metrics(p_asymm_test, y_test, tau=tau)
        print(f"  {tau:<5.2f} | {mb['acc']:>6.2f}%  | {ma['acc']:>6.2f}%  | {mb['recall']:>6.2f}%  | {ma['recall']:>6.2f}%  | {mb['total_fa']:>4d}/300  | {ma['total_fa']:>4d}/300")

    # Final error reconciliation check on Asymmetric model at argmax
    m_asymm_argmax = evaluate_metrics(p_asymm_test, y_test, tau=0.50)
    reconciled = (m_asymm_argmax["misses"] + m_asymm_argmax["total_fa"] + m_asymm_argmax["other_errors"] == 400 - m_asymm_argmax["correct"])
    print(f"\n  Error Reconciliation (Asymmetric argmax): {m_asymm_argmax['misses']} (miss) + {m_asymm_argmax['total_fa']} (FA) + {m_asymm_argmax['other_errors']} (other) = {400 - m_asymm_argmax['correct']} errors. (Exact match: {reconciled})")


if __name__ == "__main__":
    main()
