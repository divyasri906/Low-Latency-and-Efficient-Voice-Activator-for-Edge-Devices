"""
End-to-end Multi-Seed Ensemble Runner for TENet6-Narrow + iRDT (W=5).

1. Populates seed_42 from verified save/optimized_w5 checkpoint.
2. Trains seed_101 (3000 steps, lr_list=[0.002, 0.0005, 0.0001], boundaries=[1200, 2200], dropout=0.2).
3. Trains seed_2024 (same hyperparameters).
4. Runs ensemble evaluation on test.txt (400 samples):
   - Individual Seed Metrics (Seed 42, Seed 101, Seed 2024)
   - Soft Probability Averaging (P_ens = mean(P_i))
   - Majority Hard Voting (Marvin if >= 2 models agree)
   - Detailed false alarm reduction audit
"""

import os
import sys
import glob
import shutil
import subprocess
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import models
from irdt.irdtv import irdtv_optimized
from evaluate_optimized_w5 import load_split_data, evaluate_metrics


def copy_checkpoint(src_prefix, dst_prefix):
    src_dir = os.path.dirname(src_prefix)
    dst_dir = os.path.dirname(dst_prefix)
    os.makedirs(dst_dir, exist_ok=True)
    src_base = os.path.basename(src_prefix)
    dst_base = os.path.basename(dst_prefix)

    for fname in os.listdir(src_dir):
        if fname.startswith(src_base):
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname.replace(src_base, dst_base)))

    with open(os.path.join(dst_dir, "checkpoint"), "w") as f:
        f.write(f'model_checkpoint_path: "{dst_base}"\n')
        f.write(f'all_model_checkpoint_paths: "{dst_base}"\n')


def train_seed(seed, save_dir, dataset_path, split_dir, steps=3000):
    ckpt_target = os.path.join(save_dir, f"TENet6NarrowIRDTModel-{steps}.index")
    if os.path.exists(ckpt_target):
        print(f"--> Checkpoint for Seed {seed} already exists at {save_dir}. Skipping training.")
        return

    print(f"\n{'='*70}")
    print(f" [TRAINING] TENet6-Narrow (W=5, M=50) | SEED {seed}")
    print(f" Target folder: {save_dir}")
    print(f"{'='*70}\n")

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
    ]

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed for seed {seed} with return code {proc.returncode}")
    print(f"\n--> Successfully trained Seed {seed}.\n")


def get_model_probs(X, ckpt_path, dataset_path, split_dir, batch_size=100):
    g = tf.Graph()
    with g.as_default():
        feat_ph = tf.compat.v1.placeholder(tf.float32, [None, X.shape[1], X.shape[2]])
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
        for i in range(0, len(X), batch_size):
            batch_X = X[i : i + batch_size]
            probs_val = sess.run(probs_tensor, feed_dict={feat_ph: batch_X})
            all_probs.append(probs_val)

        sess.close()
    return np.vstack(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=r"C:\XboxGames\Dataset\speech_commands_v0.02")
    parser.add_argument("--split_dir", default="configs/v1_irdt")
    parser.add_argument("--base_save_dir", default="save/ensemble_w5")
    args = parser.parse_args()

    os.makedirs(args.base_save_dir, exist_ok=True)
    seed_42_dir = os.path.join(args.base_save_dir, "seed_42")
    seed_101_dir = os.path.join(args.base_save_dir, "seed_101")
    seed_2024_dir = os.path.join(args.base_save_dir, "seed_2024")

    # Step 1: Copy existing verified seed 42
    if not os.path.exists(os.path.join(seed_42_dir, "TENet6NarrowIRDTModel-3000.index")):
        print("--> Populating seed_42 from save/optimized_w5...")
        copy_checkpoint("save/optimized_w5/TENet6NarrowIRDTModel-3000", os.path.join(seed_42_dir, "TENet6NarrowIRDTModel-3000"))

    # Step 2: Train Seed 101
    train_seed(101, seed_101_dir, args.dataset_path, args.split_dir)

    # Step 3: Train Seed 2024
    train_seed(2024, seed_2024_dir, args.dataset_path, args.split_dir)

    # Step 4: Run Multi-Seed Ensemble Evaluation
    print(f"\n{'='*75}")
    print(f" EVALUATING 3-SEED ENSEMBLE ON TEST SET (400 samples)")
    print(f"{'='*75}\n")

    test_file = os.path.join(args.split_dir, "test.txt")
    X_test, y_test, words_test, _ = load_split_data(args.dataset_path, test_file, w=5)

    ckpt_42 = os.path.join(seed_42_dir, "TENet6NarrowIRDTModel-3000")
    ckpt_101 = os.path.join(seed_101_dir, "TENet6NarrowIRDTModel-3000")
    ckpt_2024 = os.path.join(seed_2024_dir, "TENet6NarrowIRDTModel-3000")

    print("[1/4] Running inference for Seed 42...")
    probs_42 = get_model_probs(X_test, ckpt_42, args.dataset_path, args.split_dir)
    print("[2/4] Running inference for Seed 101...")
    probs_101 = get_model_probs(X_test, ckpt_101, args.dataset_path, args.split_dir)
    print("[3/4] Running inference for Seed 2024...")
    probs_2024 = get_model_probs(X_test, ckpt_2024, args.dataset_path, args.split_dir)

    # Evaluate individual models
    res_42 = evaluate_metrics(probs_42, y_test, tau=0.50)
    res_101 = evaluate_metrics(probs_101, y_test, tau=0.50)
    res_2024 = evaluate_metrics(probs_2024, y_test, tau=0.50)

    # Soft Voting Ensemble: Mean Probability
    probs_soft_ens = (probs_42 + probs_101 + probs_2024) / 3.0
    res_soft_ens = evaluate_metrics(probs_soft_ens, y_test, tau=0.50)

    # Majority Hard Voting Ensemble: Marvin if >= 2 seeds predict Marvin
    preds_42 = res_42["preds"]
    preds_101 = res_101["preds"]
    preds_2024 = res_2024["preds"]

    # Marvin indicator (pred == 0)
    marvin_votes = (preds_42 == 0).astype(int) + (preds_101 == 0).astype(int) + (preds_2024 == 0).astype(int)
    # Background indicator (pred == 2)
    noise_votes = (preds_42 == 2).astype(int) + (preds_101 == 2).astype(int) + (preds_2024 == 2).astype(int)

    preds_hard_ens = np.ones(len(y_test), dtype=np.int64)  # default Unknown (1)
    for i in range(len(y_test)):
        if marvin_votes[i] >= 2:
            preds_hard_ens[i] = 0
        elif noise_votes[i] >= 2:
            preds_hard_ens[i] = 2
        else:
            # Fall back to argmax of soft probabilities
            preds_hard_ens[i] = np.argmax(probs_soft_ens[i])

    correct_hard = int(np.sum(preds_hard_ens == y_test))
    acc_hard = correct_hard / len(y_test) * 100.0
    marvin_detected_hard = int(np.sum((preds_hard_ens == 0) & (y_test == 0)))
    recall_hard = marvin_detected_hard / 100.0 * 100.0
    misses_hard = 100 - marvin_detected_hard
    fa_unk_hard = int(np.sum((preds_hard_ens == 0) & (y_test == 1)))
    fa_noise_hard = int(np.sum((preds_hard_ens == 0) & (y_test == 2)))
    total_fa_hard = fa_unk_hard + fa_noise_hard
    fa_rate_hard = total_fa_hard / 300.0 * 100.0
    other_errors_hard = int(np.sum((y_test == 1) & (preds_hard_ens == 2))) + int(np.sum((y_test == 2) & (preds_hard_ens == 1)))

    print(f"\n{'='*75}")
    print(f" MULTI-SEED ENSEMBLE RESULTS SUMMARY (TEST SET: 400 SAMPLES)")
    print(f"{'='*75}")
    print(f"{'Model / Ensemble':25s} | {'Test Acc':8s} | {'Recall':8s} | {'FAs (Unk, Noise)':18s} | {'FA Rate':8s} | {'Misses':6s}")
    print(f"{'-'*75}")
    print(f"{'Seed 42 (Individual)':25s} | {res_42['acc']:>6.2f}% | {res_42['recall']:>6.2f}% | {res_42['fa_unk']:>3d}, {res_42['fa_noise']:>3d}            | {res_42['fa_rate']:>6.2f}% | {res_42['misses']:>6d}")
    print(f"{'Seed 101 (Individual)':25s} | {res_101['acc']:>6.2f}% | {res_101['recall']:>6.2f}% | {res_101['fa_unk']:>3d}, {res_101['fa_noise']:>3d}            | {res_101['fa_rate']:>6.2f}% | {res_101['misses']:>6d}")
    print(f"{'Seed 2024 (Individual)':25s} | {res_2024['acc']:>6.2f}% | {res_2024['recall']:>6.2f}% | {res_2024['fa_unk']:>3d}, {res_2024['fa_noise']:>3d}            | {res_2024['fa_rate']:>6.2f}% | {res_2024['misses']:>6d}")
    print(f"{'-'*75}")
    print(f"{'Soft Prob Ensemble (Mean)':25s} | {res_soft_ens['acc']:>6.2f}% | {res_soft_ens['recall']:>6.2f}% | {res_soft_ens['fa_unk']:>3d}, {res_soft_ens['fa_noise']:>3d}            | {res_soft_ens['fa_rate']:>6.2f}% | {res_soft_ens['misses']:>6d}")
    print(f"{'Majority Voting (>=2/3)':25s} | {acc_hard:>6.2f}% | {recall_hard:>6.2f}% | {fa_unk_hard:>3d}, {fa_noise_hard:>3d}            | {fa_rate_hard:>6.2f}% | {misses_hard:>6d}")
    print(f"{'='*75}")

    # Reconciliations
    print(f"\nError Reconciliation Checks:")
    print(f"  Soft Ensemble: {res_soft_ens['misses']} (misses) + {res_soft_ens['total_fa']} (FAs) + {res_soft_ens['other_errors']} (other) = {400 - res_soft_ens['correct']} errors. (Matches 400 - {res_soft_ens['correct']})")
    print(f"  Hard Ensemble: {misses_hard} (misses) + {total_fa_hard} (FAs) + {other_errors_hard} (other) = {400 - correct_hard} errors. (Matches 400 - {correct_hard})")

    # Audit of False Alarms in Majority Ensemble
    print(f"\n--- Majority Voting (>=2/3) False Alarm Audit ---")
    fa_hard_indices = np.where((preds_hard_ens == 0) & (y_test != 0))[0]
    print(f"  Total False Alarms surviving 2-of-3 voting: {len(fa_hard_indices)}")
    for idx in fa_hard_indices:
        v42 = int(preds_42[idx] == 0)
        v101 = int(preds_101[idx] == 0)
        v2024 = int(preds_2024[idx] == 0)
        print(f"    FA index {idx:3d}: word={words_test[idx]:10s} Votes=[S42:{v42}, S101:{v101}, S2024:{v2024}] P_soft(Marvin)={probs_soft_ens[idx,0]:.4f}")

    # Threshold tuning on Soft Probability Ensemble
    print(f"\n--- Soft Probability Ensemble Threshold Sweep ---")
    print(f"  {'tau':5s} | {'Test Acc':8s} | {'Recall':8s} | {'FAs (Unk, Noise)':18s} | {'FA Rate':8s}")
    print(f"  {'-'*55}")
    for tau in [0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        r = evaluate_metrics(probs_soft_ens, y_test, tau=tau)
        print(f"  {tau:<5.2f} | {r['acc']:>6.2f}% | {r['recall']:>6.2f}% | {r['fa_unk']:>3d}, {r['fa_noise']:>3d}            | {r['fa_rate']:>6.2f}%")


if __name__ == "__main__":
    main()
