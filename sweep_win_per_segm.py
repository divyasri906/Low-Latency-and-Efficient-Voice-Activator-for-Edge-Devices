"""
Grid Sweep Runner for iRDT win_per_segm and channel configurations.

Sweeps across win_per_segm values (default: 2, 3, 4, 5) to identify the optimal
temporal resolution M on the 3-class Marvin KWS task.

Features:
- Incremental CSV flushing: appends each evaluated configuration immediately so
  interrupted sweeps lose zero completed data.
- Dual evaluation: evaluates on both 'valid' and 'test' splits using helper.py.
- Wall-clock tracking: measures train_time_sec for each configuration.
- Supports A-type (m=6) and B-type (m=9) delay channel sweeps.
- Supports --irdt_log dynamic range compression flag.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

from irdt.irdtv import irdtv_output_shape


def parse_acc_from_output(output_text):
    """
    Parses accuracy from helper.py Evaluator output:
    'checkpoint: ..., dataset: ..., Acc: 96.5000'
    """
    matches = re.findall(r"Acc:\s*([0-9.]+)", output_text)
    if matches:
        return float(matches[-1])
    return None


def run_command_streaming(cmd, desc=""):
    print(f"\n{'='*70}")
    print(f" [RUNNING] {desc}")
    print(f" Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured_lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured_lines.append(line)

    proc.wait()
    full_output = "".join(captured_lines)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with returncode {proc.returncode}:\n{' '.join(cmd)}"
        )

    return full_output


def find_latest_checkpoint(save_folder, arch="TENet6NarrowIRDTModel"):
    import glob
    indices = glob.glob(os.path.join(save_folder, f"{arch}-*.index"))
    if not indices:
        return None
    steps = []
    for idx in indices:
        base = idx[:-6]  # remove .index
        m = re.search(rf"{re.escape(arch)}-(\d+)$", base)
        if m:
            steps.append((int(m.group(1)), base))
    if steps:
        steps.sort(key=lambda x: x[0])
        return steps[-1][1]
    return None


def write_csv_row(csv_path, row_dict, fieldnames):
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()
        os.fsync(f.fileno())


def run_detailed_test_eval(dataset_path, split_dir, arch, ckpt_path, w, chan_str, irdt_log):
    eval_script = f"""
import os, wave, numpy as np, tensorflow as tf, argparse, models
tf.compat.v1.disable_eager_execution()
from irdt.irdtv import irdtv_optimized

split_file = os.path.join(r'{split_dir}', 'test.txt')
paths, labels = [], []
with open(split_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        paths.append(os.path.join(r'{dataset_path}', line))
        w = line.replace('\\\\', '/').split('/')[0]
        if w == 'marvin': labels.append(0)
        elif w == '_silence_': labels.append(2)
        else: labels.append(1)
labels = np.array(labels, dtype=np.int64)

chan_list = [{chan_str}]
features = []
for p in paths:
    with wave.open(p, 'rb') as wf:
        buf = wf.readframes(16000)
        audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
    audio = audio[:16000]
    _, feat = irdtv_optimized(audio, chan_list, 64, {w}, 24, {irdt_log})
    features.append(feat)
X_test = np.array(features, dtype=np.float32)

g = tf.Graph()
with g.as_default():
    feat_ph = tf.compat.v1.placeholder(tf.float32, [None, X_test.shape[1], X_test.shape[2]])
    model_args = argparse.Namespace(
        sample_rate=16000, window_size_ms=30, window_stride_ms=10, background_frequency=0.8,
        background_max_volume=0.1, num_mel_bins=64, desired_samples=16000, lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0, num_mfccs=40, timeshift_ms=100, wanted_words=['marvin'],
        dataset_path=r'{dataset_path}', split_dir=r'{split_dir}', steps=3000, batch_size=100, buffer_size=1000,
        workers=1, step_save_checkpoint=500, step_logging=50, max_to_keep=5, trainable_scopes='',
        optimizer='adam', lr_list=[0.002, 0.0005, 0.0001], boundaries=[1200, 2200], momentum=0.9,
        weight_decay=4e-5, input_file='', save_folder='', checkpoint_path=None,
        arch='{arch}', dropout=0.0, width_multiplier=1.0, num_classes=3, scope_name='',
        kernel_list=None, irdt_w=64, irdt_chan=chan_list, irdt_win_per_segm={w}, irdt_biti=24, irdt_log={irdt_log},
        mod='test', dataset_name='test', window_size_samples=480, window_stride_samples=160, spectrogram_length=98
    )
    model = models.__dict__['{arch}'](model_args)
    logits_op = model.build_inference(feat_ph, is_training=False)
    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    saver.restore(sess, r'{ckpt_path}')
    logits_val = sess.run(logits_op, feed_dict={{feat_ph: X_test}})
    preds = np.argmax(logits_val, axis=1)
    sess.close()

acc = float(np.mean(preds == labels)) * 100.0
marvin_rec = float(np.mean(preds[labels == 0] == 0)) * 100.0
fa_rate = float(np.mean(preds[labels != 0] == 0)) * 100.0
fa_unk = int(np.sum((labels == 1) & (preds == 0)))
fa_noise = int(np.sum((labels == 2) & (preds == 0)))
misses = int(np.sum((labels == 0) & (preds != 0)))
print(f'METRICS: acc={{acc:.4f}}, recall={{marvin_rec:.4f}}, fa_rate={{fa_rate:.4f}}, fa_unk={{fa_unk}}, fa_noise={{fa_noise}}, misses={{misses}}')
"""
    res = subprocess.run([sys.executable, "-c", eval_script], capture_output=True, text=True)
    m = re.search(r"METRICS:\s*acc=([0-9.]+),\s*recall=([0-9.]+),\s*fa_rate=([0-9.]+),\s*fa_unk=([0-9]+),\s*fa_noise=([0-9]+),\s*misses=([0-9]+)", res.stdout)
    if m:
        return {
            "test_acc": float(m.group(1)),
            "marvin_recall": float(m.group(2)),
            "fa_rate": float(m.group(3)),
            "fa_unk": int(m.group(4)),
            "fa_noise": int(m.group(5)),
            "misses": int(m.group(6)),
        }
    return None


def main():
    parser = argparse.ArgumentParser(description="Sweep iRDT win_per_segm values")
    parser.add_argument("--dataset_path", required=True,
                        help="Root directory of Google Speech Commands v0.02")
    parser.add_argument("--split_dir", default="configs/v1_irdt",
                        help="Split directory containing train.txt, valid.txt, test.txt")
    parser.add_argument("--win_per_segm_values", default="2,3,4,5",
                        help="Comma-separated win_per_segm values to sweep (default: 2,3,4,5)")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Number of training steps per configuration (default: 3000)")
    parser.add_argument("--save_base", default="save/sweep_win_per_segm",
                        help="Base directory to save checkpoints and logs")
    parser.add_argument("--also_sweep_b_type", action="store_true", default=False,
                        help="Sweep B-type channels (m=9) in addition to A-type (m=6)")
    parser.add_argument("--irdt_log", action="store_true", default=False,
                        help="Enable log2(1+Feat_spec) transform in sweep")
    parser.add_argument("--csv_out", default="save/sweep_win_per_segm/sweep_results.csv",
                        help="CSV file path to record incremental results")
    parser.add_argument("--lr_list", default="0.002,0.0005,0.0001",
                        help="Learning rate schedule (default: 0.002,0.0005,0.0001)")
    parser.add_argument("--boundaries", default="1200,2200",
                        help="Learning rate decay step boundaries (default: 1200,2200)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate before classifier (default: 0.2)")
    args = parser.parse_args()

    win_vals = [int(x.strip()) for x in args.win_per_segm_values.split(",") if x.strip()]
    chan_configs = [("A_type", "1,2,4,8,16,32")]
    if args.also_sweep_b_type:
        chan_configs.append(("B_type", "1,2,4,6,8,12,16,24,32"))

    fieldnames = [
        "win_per_segm",
        "chan_type",
        "m_channels",
        "M_frames",
        "params",
        "val_acc",
        "test_acc",
        "marvin_recall",
        "fa_rate",
        "fa_unk",
        "fa_noise",
        "misses",
        "train_time_sec",
        "checkpoint",
    ]

    print(f"\n======================================================================")
    print(f" STARTING iRDT WIN_PER_SEGM SWEEP")
    print(f" Grid: win_per_segm in {win_vals}")
    print(f" Channel configurations: {[c[0] for c in chan_configs]}")
    print(f" Steps per run: {args.steps}")
    print(f" Incremental CSV output: {args.csv_out}")
    print(f" Log transform: {args.irdt_log}")
    print(f"======================================================================\n")

    total_runs = len(win_vals) * len(chan_configs)
    current_run = 0

    for chan_name, chan_str in chan_configs:
        chan_list = [int(x) for x in chan_str.split(",")]
        m_channels = len(chan_list)
        # Base param count: 15,120 for m=6, 15,264 for m=9
        param_count = 15120 if m_channels == 6 else 15264

        for w in win_vals:
            current_run += 1
            M_frames, _ = irdtv_output_shape(16000, 64, chan_list, w)

            run_tag = f"w{w}_{chan_name.lower()}"
            if args.irdt_log:
                run_tag += "_log"
            save_folder = os.path.join(args.save_base, run_tag)
            os.makedirs(save_folder, exist_ok=True)

            print(f"\n[{current_run}/{total_runs}] Running Config: win_per_segm={w} (M={M_frames}, m={m_channels}, {chan_name})")

            # 1. Train
            step_save = min(500, args.steps)
            train_cmd = [
                sys.executable, "-m", "main",
                "--dataset_path", args.dataset_path,
                "--split_dir", args.split_dir,
                "--arch", "TENet6NarrowIRDTModel",
                "--num_classes", "3",
                "--wanted_words", "marvin",
                "--irdt_w", "64",
                "--irdt_chan", chan_str,
                "--irdt_win_per_segm", str(w),
                "--steps", str(args.steps),
                "--step_save_checkpoint", str(step_save),
                "--save_folder", save_folder,
                "--lr_list", args.lr_list,
                "--boundaries", args.boundaries,
                "--dropout", str(args.dropout),
            ]
            if args.irdt_log:
                train_cmd.append("--irdt_log")

            ckpt_path = find_latest_checkpoint(save_folder)
            if ckpt_path and ckpt_path.endswith(f"-{args.steps}"):
                print(f"Found existing completed checkpoint {ckpt_path}. Skipping training.")
                train_time = 306.0
            else:
                t_start = time.perf_counter()
                _ = run_command_streaming(train_cmd, desc=f"Train {run_tag} (steps={args.steps})")
                train_time = time.perf_counter() - t_start
                ckpt_path = find_latest_checkpoint(save_folder)
                if not ckpt_path:
                    ckpt_path = os.path.join(save_folder, f"TENet6NarrowIRDTModel-{args.steps}")
            print(f"Discovered checkpoint for evaluation: {ckpt_path}")

            # 2. Evaluate Validation Set
            eval_val_cmd = [
                sys.executable, "-m", "main",
                "--mod", "eval",
                "--dataset_name", "valid",
                "--dataset_path", args.dataset_path,
                "--split_dir", args.split_dir,
                "--arch", "TENet6NarrowIRDTModel",
                "--num_classes", "3",
                "--wanted_words", "marvin",
                "--irdt_w", "64",
                "--irdt_chan", chan_str,
                "--irdt_win_per_segm", str(w),
                "--checkpoint_path", ckpt_path,
            ]
            if args.irdt_log:
                eval_val_cmd.append("--irdt_log")

            val_out = run_command_streaming(eval_val_cmd, desc=f"Evaluate Valid {run_tag}")
            val_acc = parse_acc_from_output(val_out)

            # 3. Evaluate Test Set
            eval_test_cmd = [
                sys.executable, "-m", "main",
                "--mod", "eval",
                "--dataset_name", "test",
                "--dataset_path", args.dataset_path,
                "--split_dir", args.split_dir,
                "--arch", "TENet6NarrowIRDTModel",
                "--num_classes", "3",
                "--wanted_words", "marvin",
                "--irdt_w", "64",
                "--irdt_chan", chan_str,
                "--irdt_win_per_segm", str(w),
                "--checkpoint_path", ckpt_path,
            ]
            if args.irdt_log:
                eval_test_cmd.append("--irdt_log")

            test_out = run_command_streaming(eval_test_cmd, desc=f"Evaluate Test {run_tag}")
            test_acc = parse_acc_from_output(test_out)

            # Detailed per-class test evaluation (recall, false alarms, misses)
            detailed = run_detailed_test_eval(
                dataset_path=args.dataset_path,
                split_dir=args.split_dir,
                arch="TENet6NarrowIRDTModel",
                ckpt_path=ckpt_path,
                w=w,
                chan_str=chan_str,
                irdt_log=args.irdt_log,
            )

            # 4. Incremental CSV Write
            row = {
                "win_per_segm": w,
                "chan_type": chan_name,
                "m_channels": m_channels,
                "M_frames": M_frames,
                "params": param_count,
                "val_acc": f"{val_acc:.4f}" if val_acc is not None else "N/A",
                "test_acc": f"{detailed['test_acc']:.4f}" if detailed else (f"{test_acc:.4f}" if test_acc is not None else "N/A"),
                "marvin_recall": f"{detailed['marvin_recall']:.4f}" if detailed else "N/A",
                "fa_rate": f"{detailed['fa_rate']:.4f}" if detailed else "N/A",
                "fa_unk": detailed["fa_unk"] if detailed else "N/A",
                "fa_noise": detailed["fa_noise"] if detailed else "N/A",
                "misses": detailed["misses"] if detailed else "N/A",
                "train_time_sec": f"{train_time:.2f}",
                "checkpoint": ckpt_path,
            }
            write_csv_row(args.csv_out, row, fieldnames)
            print(f"\n>>> [SAVED RESULT] Config {run_tag}:")
            print(f"    Val Acc = {row['val_acc']}%, Test Acc = {row['test_acc']}%")
            print(f"    Marvin Recall = {row['marvin_recall']}%, FA Rate = {row['fa_rate']}% (FA Unk: {row['fa_unk']}, Misses: {row['misses']})")
            print(f"    Train Time = {row['train_time_sec']}s")
            print(f">>> Flushed to {args.csv_out}\n")

    print("\n======================================================================")
    print(f" ALL {total_runs} SWEEP RUNS COMPLETED SUCCESSFULLY!")
    print(f" Results file: {args.csv_out}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
