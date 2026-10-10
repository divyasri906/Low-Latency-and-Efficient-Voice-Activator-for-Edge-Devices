"""
Train TENet6NarrowIRDTModel on the 3-class Marvin KWS task:
  Class 0: Marvin (Keyword)
  Class 1: Unknown Speech
  Class 2: Background Noise

Features:
- Periodic validation tracking on configs/v1_irdt/valid.txt
- Automatically saves the BEST model checkpoint to save/TENet6NarrowIRDTModel/best/
- Final test set evaluation on configs/v1_irdt/test.txt with classification report and confusion matrix
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import config
import models
from irdt.dataset import MarvinAudioWrapper
from irdt.labels import MARVIN_LABEL_PERMUTATION, remap_labels
from sklearn.metrics import classification_report, confusion_matrix
import tf_slim as slim


CLASS_NAMES = ["Marvin", "Unknown Speech", "Background"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train TENet6-Narrow iRDT Marvin Model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"C:\XboxGames\Dataset\speech_commands_v0.02",
        help="Path to Speech Commands dataset root",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="configs/v1_irdt",
        help="Path to directory containing train.txt, valid.txt, test.txt",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="save/TENet6NarrowIRDTModel",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Total training steps (default: 3000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size (default: 100)",
    )
    parser.add_argument(
        "--step_logging",
        type=int,
        default=50,
        help="Logging interval in steps",
    )
    parser.add_argument(
        "--step_eval",
        type=int,
        default=150,
        help="Validation evaluation interval in steps",
    )
    parser.add_argument(
        "--step_save",
        type=int,
        default=500,
        help="Periodic checkpoint save interval in steps",
    )
    return parser.parse_args()


def build_model_args(cli_args, dropout=0.5, mod="train"):
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
    )


def evaluate_split(ckpt_path, cli_args, split_name="valid"):
    eval_args = build_model_args(cli_args, dropout=0.0, mod=split_name)
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
    acc = np.mean(all_preds == all_labels)
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

    # Write checkpoint text file
    with open(os.path.join(dst_dir, "checkpoint"), "w") as f:
        f.write(f'model_checkpoint_path: "{dst_base}"\n')
        f.write(f'all_model_checkpoint_paths: "{dst_base}"\n')


def main():
    cli_args = parse_args()
    os.makedirs(cli_args.save_folder, exist_ok=True)
    best_dir = os.path.join(cli_args.save_folder, "best")
    os.makedirs(best_dir, exist_ok=True)
    temp_dir = os.path.join(cli_args.save_folder, ".temp_ckpt")
    os.makedirs(temp_dir, exist_ok=True)

    print("=" * 75)
    print(" TENet6-Narrow + iRDT Marvin KWS Training & Best Weight Saver")
    print(f" Dataset Path:    {cli_args.dataset_path}")
    print(f" Split Dir:       {cli_args.split_dir}")
    print(f" Save Folder:     {cli_args.save_folder}")
    print(f" Best Weights:    {best_dir}")
    print(f" Total Steps:     {cli_args.steps} (batch size {cli_args.batch_size})")
    print(f" Classes (3):     {CLASS_NAMES}")
    print("=" * 75)

    # 1. Build Training Graph
    train_args = build_model_args(cli_args, dropout=0.5, mod="train")
    tf.compat.v1.reset_default_graph()

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

    saver = tf.compat.v1.train.Saver(var_list=model.model_variables, max_to_keep=5)
    temp_saver = tf.compat.v1.train.Saver(var_list=model.model_variables, max_to_keep=1)

    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
    dataset.init_iterator()

    print("\n[TRAINING START] Optimizer: Adam | LR schedule: [0.002 -> 0.0005 -> 0.0001]\n")

    best_val_acc = 0.0
    best_step = 0
    t_start = time.time()

    for step in range(1, cli_args.steps + 1):
        _, loss_val, acc_val, lr_val = session.run([train_op, model.total_loss, model.acc, learning_rate])

        # Logging
        if step % cli_args.step_logging == 0 or step == 1:
            elapsed = time.time() - t_start
            print(f"[{time.strftime('%H:%M:%S')}] Step {step:5d}/{cli_args.steps} | Loss: {loss_val:.4f} | Train Acc: {acc_val*100:5.2f}% | LR: {lr_val:.6f} | Elapsed: {elapsed:.1f}s")

        # Periodic checkpoint
        if step % cli_args.step_save == 0:
            ckpt_path = os.path.join(cli_args.save_folder, f"TENet6NarrowIRDTModel-{step}")
            saver.save(session, os.path.join(cli_args.save_folder, "TENet6NarrowIRDTModel"), global_step=step)

        # Validation evaluation
        if step % cli_args.step_eval == 0 or step == cli_args.steps:
            temp_prefix = os.path.join(temp_dir, "eval_temp")
            temp_saver.save(session, temp_prefix)

            val_acc, val_preds, val_labels = evaluate_split(temp_prefix, cli_args, split_name="valid")
            is_best = val_acc > best_val_acc

            marker = " *** NEW BEST ***" if is_best else ""
            print(f"  --> [VALIDATION] Step {step:5d} | Val Accuracy: {val_acc*100:5.2f}% (Best so far: {max(best_val_acc, val_acc)*100:5.2f}% at step {best_step if not is_best else step}){marker}")

            if is_best:
                best_val_acc = val_acc
                best_step = step
                best_prefix = os.path.join(best_dir, "TENet6NarrowIRDTModel-best")
                copy_checkpoint(temp_prefix, best_prefix)

    session.close()

    # Clean up temp
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 75)
    print(" TRAINING COMPLETE")
    print(f" Best Validation Accuracy: {best_val_acc*100:.2f}% at Step {best_step}")
    print(f" Best Weights Saved At:    {os.path.join(best_dir, 'TENet6NarrowIRDTModel-best')}")
    print("=" * 75)

    # 2. Final Evaluation on Held-out Test Set using the Best Checkpoint
    best_prefix = os.path.join(best_dir, "TENet6NarrowIRDTModel-best")
    print("\nRunning Final Test Set Evaluation using the BEST Model Weights...")
    test_acc, test_preds, test_labels = evaluate_split(best_prefix, cli_args, split_name="test")

    print("\n" + "=" * 75)
    print(f" FINAL TEST EVALUATION RESULTS (Accuracy: {test_acc*100:.2f}%)")
    print("=" * 75)
    print(f"Total Test Samples: {len(test_labels)} ({np.sum(test_preds == test_labels)} / {len(test_labels)} correct)\n")

    report = classification_report(test_labels, test_preds, target_names=CLASS_NAMES, digits=4)
    print("Classification Report:")
    print("-" * 75)
    print(report)

    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix:")
    print("-" * 75)
    header = f"{'True/Pred':<16} " + " ".join([f"{name[:8]:>8}" for name in CLASS_NAMES])
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:8d}" for val in row])
        print(f"{CLASS_NAMES[i]:<16} {row_str}")
    print("=" * 75)

    # Save metrics JSON
    metrics = {
        "best_step": best_step,
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "total_test_samples": int(len(test_labels)),
        "correct_test_samples": int(np.sum(test_preds == test_labels)),
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }
    with open(os.path.join(best_dir, "best_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics summary to: {os.path.join(best_dir, 'best_metrics.json')}")


if __name__ == "__main__":
    main()
