"""
Evaluation script for pre-trained model checkpoints in the save/ folder.

Evaluates TENet12Model checkpoints on the Google Speech Commands test set.
Handles batch normalization statistics and dropout configuration for 
accurate inference matching the trained model weights.
"""

import argparse
import glob
import os
import re

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import config
import input_data
import models
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved TENet model checkpoints")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"C:\XboxGames\Dataset\speech_commands_v0.02",
        help="Path to Speech Commands dataset root",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        default="configs/v1_eval",
        help="Path to directory containing test.txt",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="save/tenet12",
        help="Path to directory containing checkpoints",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="TENet12Model",
        help="Model architecture",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for evaluation",
    )
    return parser.parse_args()


def get_available_checkpoints(save_dir):
    meta_files = glob.glob(os.path.join(save_dir, "*.meta"))
    prefixes = set()
    for f in meta_files:
        prefix = f[:-5]  # strip .meta
        prefixes.add(prefix)

    def extract_step(p):
        match = re.search(r"-(\d+)$", p)
        return int(match.group(1)) if match else 0

    return sorted(list(prefixes), key=extract_step)


def evaluate_checkpoint(ckpt_path, cli_args):
    tf.compat.v1.reset_default_graph()

    args = config.arg_config.__wrapped__() if hasattr(config.arg_config, "__wrapped__") else None
    parser = argparse.ArgumentParser()
    # Populate default configuration
    default_args = config.arg_config.__code__.co_varnames
    # Mimic config arguments
    args = argparse.Namespace(
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
        wanted_words=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
        dataset_path=cli_args.dataset_path,
        split_dir=cli_args.split_dir,
        steps=30000,
        batch_size=cli_args.batch_size,
        buffer_size=1000,
        workers=8,
        step_save_checkpoint=1000,
        step_logging=100,
        max_to_keep=5,
        trainable_scopes="",
        optimizer="adam",
        lr_list=[0.01, 0.001, 0.0001],
        boundaries=[10000, 20000],
        momentum=0.9,
        weight_decay=4e-5,
        input_file="",
        save_folder=cli_args.save_dir,
        checkpoint_path=ckpt_path,
        arch=cli_args.arch,
        dropout=0.0,  # 0.0 at evaluation
        width_multiplier=1.0,
        num_classes=12,
        scope_name="",
        kernel_list=None,
        irdt_w=64,
        irdt_chan=[1, 2, 4, 8, 16, 32],
        irdt_win_per_segm=3,
        irdt_biti=24,
        irdt_log=False,
        mod="eval",
        dataset_name="test",
        window_size_samples=480,
        window_stride_samples=160,
        spectrogram_length=98,
    )

    session = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
    dataset = input_data.AudioWrapper(args, "test", False, session)
    wavs, labels = dataset.get_input_and_output_op()

    model = models.__dict__[args.arch](args)
    # is_training=True uses active batch normalization statistics for inference
    model.build(wavs=wavs, labels=labels, is_training=True)

    session.run(tf.compat.v1.global_variables_initializer())
    session.run(tf.compat.v1.local_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=model.model_variables)
    saver.restore(session, ckpt_path)

    all_preds = []
    all_labels = []
    num_batches = (dataset.num_samples - 1) // args.batch_size + 1

    for _ in range(num_batches):
        try:
            logits_val, labels_val = session.run([model.logits, labels])
            preds = np.argmax(logits_val, axis=1)
            all_preds.extend(preds)
            all_labels.extend(labels_val)
        except tf.errors.OutOfRangeError:
            break

    session.close()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = np.mean(all_preds == all_labels)

    return acc, all_preds, all_labels, dataset.prepare_words_list


def main():
    cli_args = parse_args()
    print("=" * 70)
    print(" TENet Model Evaluation Harness")
    print(f" Dataset Path: {cli_args.dataset_path}")
    print(f" Split Dir:    {cli_args.split_dir}")
    print(f" Save Dir:     {cli_args.save_dir}")
    print("=" * 70)

    checkpoints = get_available_checkpoints(cli_args.save_dir)
    if not checkpoints:
        print(f"No checkpoints found in {cli_args.save_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for c in checkpoints:
        print(f" - {c}")
    print("-" * 70)

    results = {}
    class_names = None

    for ckpt in checkpoints:
        step_name = os.path.basename(ckpt)
        print(f"\nEvaluating {step_name}...")
        acc, preds, labels, class_names = evaluate_checkpoint(ckpt, cli_args)
        results[step_name] = (acc, preds, labels)
        print(f"-> Accuracy: {acc * 100:.2f}% ({np.sum(preds == labels)} / {len(labels)})")

    print("\n" + "=" * 70)
    print(" SUMMARY OF CHECKPOINT PERFORMANCE")
    print("=" * 70)
    print(f"{'Checkpoint':<30} | {'Test Accuracy':<15} | {'Correct / Total'}")
    print("-" * 70)
    for step_name, (acc, preds, labels) in sorted(results.items()):
        correct = np.sum(preds == labels)
        print(f"{step_name:<30} | {acc * 100:6.2f}%         | {correct} / {len(labels)}")
    print("-" * 70)

    best_name = max(results.keys(), key=lambda k: results[k][0])
    best_acc, best_preds, best_labels = results[best_name]
    print(f"\nBest Performing Model: {best_name} with Accuracy = {best_acc * 100:.2f}%\n")

    print("Detailed Classification Report for Best Model:")
    print("-" * 70)
    print(classification_report(best_labels, best_preds, target_names=class_names, digits=4))

    print("Confusion Matrix:")
    print("-" * 70)
    cm = confusion_matrix(best_labels, best_preds)
    header = f"{'True/Pred':<12} " + " ".join([f"{name[:4]:>5}" for name in class_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:5d}" for val in row])
        print(f"{class_names[i]:<12} {row_str}")


if __name__ == "__main__":
    main()
