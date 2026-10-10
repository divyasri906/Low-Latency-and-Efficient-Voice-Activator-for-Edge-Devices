"""
Train Multi-Branch MTENet6-Narrow (kernel_list=[3, 9]) and Fuse to 15,120 Params.

Features:
- Multi-branch training with kernel_list=[3, 9] (17,872 training params).
- Automated linear re-parameterization via tenet_fusion.py to 15,120 deployed params.
- Pre/post fusion sanity check on validation set (verifying 0 flipped predictions).
- Clean checkpoint scoping (saving only TENet6Narrow/ variables).
- Multi-seed protocol across seeds [42, 101, 2024].
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf
import tf_slim as slim

tf.compat.v1.disable_eager_execution()

import config
import models
from irdt.dataset import MarvinAudioWrapper
from irdt.labels import MARVIN_LABEL_PERMUTATION, remap_labels
from tenet_fusion import fusion_tenet


CLASS_NAMES = ["Marvin", "Unknown Speech", "Background"]


def build_mtenet_args(cli_args, seed=42, kernel_list=[3, 9], mod="train"):
    return argparse.Namespace(
        sample_rate=16000,
        window_size_ms=30,
        window_stride_ms=10,
        background_frequency=0.8,
        background_max_volume=0.1,
        num_mel_bins=64,
        desired_samples=16000,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
        num_mfccs=40,
        timeshift_ms=100,
        wanted_words=["marvin"],
        dataset_path=cli_args.dataset_path,
        split_dir=cli_args.split_dir,
        steps=cli_args.steps,
        batch_size=cli_args.batch_size,
        buffer_size=1000,
        workers=4,
        step_save_checkpoint=cli_args.step_save,
        step_logging=cli_args.step_logging,
        max_to_keep=5,
        trainable_scopes="",
        optimizer="adam",
        lr_list=[0.002, 0.0005, 0.0001],
        boundaries=[1200, 2200],
        momentum=0.9,
        weight_decay=4e-5,
        input_file="",
        save_folder=cli_args.save_folder,
        checkpoint_path=None,
        arch="TENet6NarrowIRDTModel",
        dropout=0.5 if mod == "train" else 0.0,
        width_multiplier=1.0,
        num_classes=3,
        scope_name="",
        kernel_list=kernel_list,
        irdt_w=64,
        irdt_chan=[1, 2, 4, 8, 16, 32],
        irdt_win_per_segm=3,
        irdt_biti=24,
        irdt_log=False,
        mod=mod,
        dataset_name=mod,
        window_size_samples=480,
        window_stride_samples=160,
        spectrogram_length=98,
        seed=seed,
    )


def evaluate_split(ckpt_path, cli_args, kernel_list=[3, 9], split_name="valid"):
    eval_args = build_mtenet_args(cli_args, kernel_list=kernel_list, mod=split_name)
    eval_graph = tf.Graph()

    with eval_graph.as_default():
        session = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
        dataset = MarvinAudioWrapper(eval_args, split_name, False, session)
        wavs, labels = dataset.get_input_and_output_op()
        labels = remap_labels(labels, MARVIN_LABEL_PERMUTATION)

        model = models.__dict__[eval_args.arch](eval_args)
        model.build(wavs=wavs, labels=labels, is_training=True)

        session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        saver = tf.compat.v1.train.Saver(var_list=model.model_variables)
        saver.restore(session, ckpt_path)
        dataset.init_iterator()

        all_preds = []
        all_labels = []
        num_batches = (dataset.num_samples - 1) // eval_args.batch_size + 1

        for _ in range(num_batches):
            try:
                logits_val, labels_val = session.run([model.logits, labels])
                all_preds.extend(np.argmax(logits_val, axis=1))
                all_labels.extend(labels_val)
            except tf.errors.OutOfRangeError:
                break

        session.close()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = float(np.mean(all_preds == all_labels))
    return acc, all_preds, all_labels


def run_fusion(mtenet_ckpt, fused_save_dir, cli_args, kernel_list=[3, 9]):
    """
    Fuses multi-branch MTENet into single-branch TENet6-Narrow (15,120 params)
    with clean destination checkpoint scoping (excluding MTENet variables).
    """
    os.makedirs(fused_save_dir, exist_ok=True)
    fusion_graph = tf.Graph()

    with fusion_graph.as_default():
        session = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
        
        # Audio placeholder
        wavs = tf.compat.v1.placeholder(tf.float32, [1, 16000, 1], name="wavs")
        labels = tf.constant([0], dtype=tf.int64)

        # 1. Multi-branch model
        args_m = build_mtenet_args(cli_args, kernel_list=kernel_list, mod="eval")
        model_m = models.__dict__[args_m.arch](args_m)
        model_m.build(wavs=wavs, labels=labels, is_training=False)

        # 2. Target single-branch model (kernel_list=None -> scope='TENet6Narrow')
        args_t = build_mtenet_args(cli_args, kernel_list=None, mod="eval")
        model_t = models.__dict__[args_t.arch](args_t)
        model_t.build(wavs=wavs, labels=labels, is_training=False)

        session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        
        saver_m = tf.compat.v1.train.Saver(var_list=model_m.model_variables)
        saver_m.restore(session, mtenet_ckpt)

        # Scoped saver: ONLY save TENet6Narrow/ variables (15,120 parameters)
        t_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("TENet6Narrow/")]
        if not t_vars:
            t_vars = [v for v in tf.compat.v1.global_variables() if not v.name.startswith("MTENet")]
        saver_t = tf.compat.v1.train.Saver(var_list=t_vars)

        # Run linear fusion
        list_v = slim.get_variables_to_restore()
        fusion_tenet(list_v, saver_m, kernel_list, args_t, session)

        fused_prefix = os.path.join(fused_save_dir, "TENet6NarrowIRDTModel-fused")
        saver_t.save(session, fused_prefix)
        session.close()

    return fused_prefix


def train_mtenet_seed(seed, cli_args):
    print(f"\n==================================================================")
    print(f" TRAINING MTENet6-Narrow + iRDT (kernel_list=[3, 9]) | SEED {seed}")
    print(f"==================================================================")
    
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    train_args = build_mtenet_args(cli_args, seed=seed, kernel_list=[3, 9], mod="train")
    session = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
    dataset = MarvinAudioWrapper(train_args, "train", True, session)
    wavs, labels = dataset.get_input_and_output_op()
    labels = remap_labels(labels, MARVIN_LABEL_PERMUTATION)

    model = models.__dict__[train_args.arch](train_args)
    model.build(wavs=wavs, labels=labels, is_training=True)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    learning_rate = tf.compat.v1.train.piecewise_constant(
        global_step, train_args.boundaries, train_args.lr_list
    )
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = slim.learning.create_train_op(
        model.total_loss,
        optimizer,
        global_step=global_step,
        variables_to_train=tf.compat.v1.trainable_variables(),
    )

    seed_save_dir = os.path.join(cli_args.save_folder, f"seed_{seed}")
    os.makedirs(seed_save_dir, exist_ok=True)
    temp_dir = os.path.join(seed_save_dir, ".temp")
    os.makedirs(temp_dir, exist_ok=True)
    best_dir = os.path.join(seed_save_dir, "best_mtenet")
    os.makedirs(best_dir, exist_ok=True)

    temp_saver = tf.compat.v1.train.Saver(var_list=model.model_variables, max_to_keep=1)

    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
    dataset.init_iterator()

    best_val_acc = 0.0
    best_step = 0
    t0 = time.time()

    for step in range(1, cli_args.steps + 1):
        _, loss_val, acc_val, lr_val = session.run([train_op, model.total_loss, model.acc, learning_rate])

        if step % cli_args.step_logging == 0 or step == 1:
            print(f"[{time.strftime('%H:%M:%S')}] Seed {seed} | Step {step:5d}/{cli_args.steps} | Loss: {loss_val:.4f} | Train Acc: {acc_val*100:.2f}% | LR: {lr_val:.6f}")

        if step % cli_args.step_eval == 0 or step == cli_args.steps:
            temp_prefix = os.path.join(temp_dir, "eval_temp")
            temp_saver.save(session, temp_prefix)

            val_acc, _, _ = evaluate_split(temp_prefix, cli_args, kernel_list=[3, 9], split_name="valid")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_step = step
                best_prefix = os.path.join(best_dir, "MTENet6NarrowIRDTModel-best")
                # Copy checkpoint
                for fname in os.listdir(temp_dir):
                    if fname.startswith("eval_temp"):
                        shutil.copy2(os.path.join(temp_dir, fname), os.path.join(best_dir, fname.replace("eval_temp", "MTENet6NarrowIRDTModel-best")))
                with open(os.path.join(best_dir, "checkpoint"), "w") as f:
                    f.write('model_checkpoint_path: "MTENet6NarrowIRDTModel-best"\n')
                print(f"  --> [NEW BEST] Step {step} | Val Acc: {val_acc*100:.2f}%")

    session.close()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    best_mtenet_ckpt = os.path.join(best_dir, "MTENet6NarrowIRDTModel-best")
    print(f"\n[FUSION] Fusing MTENet weights for Seed {seed}...")
    fused_dir = os.path.join(seed_save_dir, "fused_tenet")
    fused_ckpt = run_fusion(best_mtenet_ckpt, fused_dir, cli_args, kernel_list=[3, 9])

    # Gap 4 Sanity Check: Evaluate real validation speech predictions before and after fusion
    print("[SANITY CHECK] Comparing pre-fusion vs post-fusion validation predictions...")
    mtenet_acc, mtenet_preds, val_labels = evaluate_split(best_mtenet_ckpt, cli_args, kernel_list=[3, 9], split_name="valid")
    fused_acc, fused_preds, _ = evaluate_split(fused_ckpt, cli_args, kernel_list=None, split_name="valid")

    diff_predictions = int(np.sum(mtenet_preds != fused_preds))
    print(f"  MTENet Val Accuracy: {mtenet_acc*100:.2f}%")
    print(f"  Fused  Val Accuracy: {fused_acc*100:.2f}%")
    print(f"  Flipped Predictions: {diff_predictions} / {len(val_labels)} (Target: 0)")

    return {
        "seed": seed,
        "best_step": best_step,
        "mtenet_val_acc": mtenet_acc,
        "fused_val_acc": fused_acc,
        "diff_predictions": diff_predictions,
        "mtenet_checkpoint": best_mtenet_ckpt,
        "fused_checkpoint": fused_ckpt,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MTENet with Fusion across seeds")
    parser.add_argument("--dataset_path", type=str, default=r"C:\XboxGames\Dataset\speech_commands_v0.02")
    parser.add_argument("--split_dir", type=str, default="configs/v1_irdt")
    parser.add_argument("--save_folder", type=str, default="save/MTENet_Fusion")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--step_logging", type=int, default=100)
    parser.add_argument("--step_eval", type=int, default=250)
    parser.add_argument("--step_save", type=int, default=1000)
    cli_args = parser.parse_args()

    seeds = [42, 101, 2024]
    all_results = {}
    for s in seeds:
        res = train_mtenet_seed(s, cli_args)
        all_results[f"seed_{s}"] = res

    out_file = os.path.join(cli_args.save_folder, "multi_seed_fusion_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    fused_accs = [r["fused_val_acc"] for r in all_results.values()]
    print("\n" + "=" * 65)
    print(" ALL SEEDS COMPLETED (MTENet + Fusion)")
    print(f" Fused Validation Accuracies: {[f'{a*100:.2f}%' for a in fused_accs]}")
    print(f" Mean Fused Val Accuracy:    {np.mean(fused_accs)*100:.2f}% +/- {np.std(fused_accs)*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
