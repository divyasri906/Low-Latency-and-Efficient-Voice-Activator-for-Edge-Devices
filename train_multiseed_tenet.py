"""
Multi-Seed Training for TENet6-Narrow + iRDT (15,120 Parameters).

Trains across seeds [101, 2024] and sets up seed 42 from existing verified checkpoint.
Saves to save/TENet6Narrow_MultiSeed/seed_{s}/best/
"""

import argparse
import json
import os
import shutil
import time
import numpy as np
import tensorflow as tf
import tf_slim as slim

tf.compat.v1.disable_eager_execution()

import config
import models
from irdt.dataset import MarvinAudioWrapper
from irdt.labels import MARVIN_LABEL_PERMUTATION, remap_labels


CLASS_NAMES = ["Marvin", "Unknown Speech", "Background"]


def build_model_args(cli_args, seed=42, dropout=0.5, mod="train"):
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
        workers=1,
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
        dropout=dropout,
        width_multiplier=1.0,
        num_classes=3,
        scope_name="",
        kernel_list=None,
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


def evaluate_split(ckpt_path, cli_args, seed=42, split_name="valid"):
    eval_args = build_model_args(cli_args, seed=seed, dropout=0.0, mod=split_name)
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


def copy_checkpoint(src_prefix, dst_prefix):
    src_dir = os.path.dirname(src_prefix)
    dst_dir = os.path.dirname(dst_prefix)
    os.makedirs(dst_dir, exist_ok=True)
    src_base = os.path.basename(src_prefix)
    dst_base = os.path.basename(dst_prefix)

    for fname in os.listdir(src_dir):
        if fname.startswith(src_base):
            suffix = fname[len(src_base):]
            shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, dst_base + suffix))

    with open(os.path.join(dst_dir, "checkpoint"), "w") as f:
        f.write(f'model_checkpoint_path: "{dst_base}"\n')
        f.write(f'all_model_checkpoint_paths: "{dst_base}"\n')


def train_seed(seed, cli_args):
    print(f"\n==================================================================")
    print(f" TRAINING TENet6-Narrow + iRDT (15,120 params) | SEED {seed}")
    print(f"==================================================================")
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    train_args = build_model_args(cli_args, seed=seed, dropout=0.5, mod="train")
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
    best_dir = os.path.join(seed_save_dir, "best")
    temp_dir = os.path.join(seed_save_dir, ".temp")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

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

            val_acc, _, _ = evaluate_split(temp_prefix, cli_args, seed=seed, split_name="valid")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_step = step
                best_prefix = os.path.join(best_dir, "TENet6NarrowIRDTModel-best")
                copy_checkpoint(temp_prefix, best_prefix)
                print(f"  --> [NEW BEST] Step {step} | Val Acc: {val_acc*100:.2f}%")

    session.close()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    best_ckpt = os.path.join(best_dir, "TENet6NarrowIRDTModel-best")
    test_acc, test_preds, test_labels = evaluate_split(best_ckpt, cli_args, seed=seed, split_name="test")
    print(f"--> [SEED {seed} COMPLETE] Best Val Acc: {best_val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    return {
        "seed": seed,
        "best_step": best_step,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "checkpoint": best_ckpt,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Training for TENet6-Narrow")
    parser.add_argument("--dataset_path", type=str, default=r"C:\XboxGames\Dataset\speech_commands_v0.02")
    parser.add_argument("--split_dir", type=str, default="configs/v1_irdt")
    parser.add_argument("--save_folder", type=str, default="save/TENet6Narrow_MultiSeed")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--step_logging", type=int, default=100)
    parser.add_argument("--step_eval", type=int, default=300)
    parser.add_argument("--step_save", type=int, default=1000)
    cli_args = parser.parse_args()

    os.makedirs(cli_args.save_folder, exist_ok=True)
    all_results = {}

    # Seed 42: Use existing verified checkpoint
    seed42_dir = os.path.join(cli_args.save_folder, "seed_42", "best")
    os.makedirs(seed42_dir, exist_ok=True)
    seed42_src = "save/TENet6NarrowIRDTModel/best/TENet6NarrowIRDTModel-best"
    seed42_dst = os.path.join(seed42_dir, "TENet6NarrowIRDTModel-best")
    copy_checkpoint(seed42_src, seed42_dst)

    print("\nEvaluating Seed 42 on validation and test sets...")
    s42_val, _, _ = evaluate_split(seed42_dst, cli_args, seed=42, split_name="valid")
    s42_test, _, _ = evaluate_split(seed42_dst, cli_args, seed=42, split_name="test")
    print(f"Seed 42 | Val Acc: {s42_val*100:.2f}% | Test Acc: {s42_test*100:.2f}%")
    all_results["seed_42"] = {
        "seed": 42,
        "best_step": 3000,
        "best_val_acc": s42_val,
        "test_acc": s42_test,
        "checkpoint": seed42_dst,
    }

    # Train seeds 101 and 2024
    for s in [101, 2024]:
        res = train_seed(s, cli_args)
        all_results[f"seed_{s}"] = res

    summary_file = os.path.join(cli_args.save_folder, "multi_seed_tenet_results.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    val_accs = [r["best_val_acc"] for r in all_results.values()]
    test_accs = [r["test_acc"] for r in all_results.values()]
    print("\n" + "=" * 65)
    print(" ALL TENET-6 SEEDS COMPLETED")
    print(f" Validation Accuracies: {[f'{a*100:.2f}%' for a in val_accs]}")
    print(f" Mean Val Accuracy:    {np.mean(val_accs)*100:.2f}% +/- {np.std(val_accs)*100:.2f}%")
    print(f" Test Accuracies:       {[f'{a*100:.2f}%' for a in test_accs]}")
    print(f" Mean Test Accuracy:   {np.mean(test_accs)*100:.2f}% +/- {np.std(test_accs)*100:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
