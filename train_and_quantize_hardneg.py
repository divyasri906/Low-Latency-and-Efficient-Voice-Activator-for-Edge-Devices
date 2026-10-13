"""
Retrain TENet6-Narrow with Logarithmic iRDT including Hard Negatives (bargain, martin, margin, etc.)
and Quantize to Full INT8 TFLite for ESP32-S3 deployment.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import glob
import wave
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

DATASET_PATH = r"C:\XboxGames\Dataset\speech_commands_v0.02"
SPLIT_DIR = "configs/v1_irdt_hardneg"
SAVE_DIR = "save/tenet_log_hardneg"
STEPS = 3000
SEED = 101

def train_model():
    ckpt_target = os.path.join(SAVE_DIR, f"TENet6NarrowIRDTModel-{STEPS}.index")
    if os.path.exists(ckpt_target):
        print(f"--> [EXISTS] Checkpoint already exists at {ckpt_target}. Skipping training.")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    print("=" * 80)
    print("STEP 1: TRAINING TENet6-Narrow + Log-iRDT with Hard Negatives")
    print(f"Dataset : {DATASET_PATH}")
    print(f"Splits  : {SPLIT_DIR} (3,562 train clips, incl. 362 hard negatives)")
    print(f"Target  : {SAVE_DIR} | Steps: {STEPS} | Seed: {SEED}")
    print("=" * 80)

    cmd = [
        sys.executable, "-u", "-m", "main",
        "--dataset_path", DATASET_PATH,
        "--split_dir", SPLIT_DIR,
        "--arch", "TENet6NarrowIRDTModel",
        "--num_classes", "3",
        "--wanted_words", "marvin",
        "--irdt_w", "64",
        "--irdt_chan", "1,2,4,8,16,32",
        "--irdt_win_per_segm", "5",
        "--irdt_log",
        "--steps", str(STEPS),
        "--step_save_checkpoint", "500",
        "--step_logging", "100",
        "--save_folder", SAVE_DIR,
        "--lr_list", "0.002,0.0005,0.0001",
        "--boundaries", "1200,2200",
        "--dropout", "0.2",
        "--seed", str(SEED),
        "--width_multiplier", "1.0",
    ]

    t0 = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Training failed with code {proc.returncode}")
    print(f"\n--> [DONE] Training completed in {elapsed:.1f}s ({elapsed/60:.2f} mins).\n")


def freeze_and_quantize():
    print("=" * 80)
    print("STEP 2: FREEZING GRAPH & INT8 QUANTIZATION")
    print("=" * 80)

    ckpt_path = os.path.join(SAVE_DIR, f"TENet6NarrowIRDTModel-{STEPS}")
    pb_path = os.path.join(SAVE_DIR, "frozen_log_model.pb")

    g = tf.Graph()
    with g.as_default():
        feat_ph = tf.compat.v1.placeholder(tf.float32, [None, 50, 6], name='input_features')
        model_args = argparse.Namespace(
            sample_rate=16000, window_size_ms=30, window_stride_ms=10,
            background_frequency=0.8, background_max_volume=0.1, num_mel_bins=64,
            desired_samples=16000, lower_edge_hertz=80.0, upper_edge_hertz=7600.0,
            num_mfccs=40, timeshift_ms=100, wanted_words=['marvin'],
            dataset_path='', split_dir='', steps=STEPS,
            batch_size=1, buffer_size=1000, workers=1,
            step_save_checkpoint=500, step_logging=50, max_to_keep=5,
            trainable_scopes='', optimizer='adam', lr_list=[0.002, 0.0005, 0.0001],
            boundaries=[1200, 2200], momentum=0.9, weight_decay=4e-5,
            input_file='', save_folder='', checkpoint_path=None,
            arch='TENet6NarrowIRDTModel', dropout=0.0, width_multiplier=1.0,
            num_classes=3, scope_name='', kernel_list=None, irdt_w=64,
            irdt_chan=[1, 2, 4, 8, 16, 32], irdt_win_per_segm=5, irdt_biti=24,
            irdt_log=True, mod='eval', dataset_name='eval',
            window_size_samples=480, window_stride_samples=160, spectrogram_length=98,
        )
        model = models.TENet6NarrowIRDTModel(model_args)
        logits_op = model.build_inference(feat_ph, is_training=False)
        probs_op = tf.nn.softmax(logits_op, name='output_probs')

        sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, ckpt_path)

        frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['output_probs']
        )
        with open(pb_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
        print(f"--> Frozen protobuf graph saved to {pb_path} ({os.path.getsize(pb_path)} bytes)")
        sess.close()

    # Load representative calibration dataset from test split
    print("Loading representative calibration samples...")
    X_calib, _, _, _ = load_split_data(DATASET_PATH, os.path.join(SPLIT_DIR, "test.txt"), w=5, irdt_log=True)

    def representative_dataset():
        for i in range(min(100, len(X_calib))):
            yield [X_calib[i:i+1].astype(np.float32)]

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=pb_path,
        input_arrays=['input_features'],
        output_arrays=['output_probs'],
        input_shapes={'input_features': [1, 50, 6]}
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    tflite_quant = converter.convert()

    # Save to local save dir and primary deployment path
    tflite_save_path = os.path.join(SAVE_DIR, "best_marvin_model_log_int8.tflite")
    tflite_root_path = "best_marvin_model_log_int8.tflite"

    with open(tflite_save_path, "wb") as f:
        f.write(tflite_quant)
    with open(tflite_root_path, "wb") as f:
        f.write(tflite_quant)

    print(f"--> [SUCCESS] INT8 TFLite saved: {tflite_root_path} ({len(tflite_quant)/1024:.2f} KB)")


def evaluate_test_split():
    print("\n" + "=" * 80)
    print("STEP 3: EVALUATING TEST SPLIT ON INT8 QUANTIZED MODEL")
    print("=" * 80)

    tflite_path = "best_marvin_model_log_int8.tflite"
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    test_file = os.path.join(SPLIT_DIR, "test.txt")
    X_test, y_test, _, _ = load_split_data(DATASET_PATH, test_file, w=5, irdt_log=True)

    preds = []
    probs_list = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1].astype(np.float32)
        interp.set_tensor(inp_idx, sample)
        interp.invoke()
        p = interp.get_tensor(out_idx)[0]
        preds.append(np.argmax(p))
        probs_list.append(p)

    preds = np.array(preds)
    acc = float(np.mean(preds == y_test) * 100.0)

    # Class 0: Marvin, Class 1: Unknown, Class 2: Background
    y_m = (y_test == 0)
    p_m = (preds == 0)
    tp = int(np.sum(y_m & p_m))
    fn = int(np.sum(y_m & ~p_m))
    fp = int(np.sum(~y_m & p_m))
    tn = int(np.sum(~y_m & ~p_m))
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0

    print(f"Test Samples : {len(y_test)}")
    print(f"Overall Acc  : {acc:.2f}%")
    print(f"Marvin Recall: {recall:.2f}% ({tp}/{tp+fn})")
    print(f"Marvin Prec  : {precision:.2f}% ({tp}/{tp+fp})")
    print(f"Confusion    : TP={tp}, FN={fn}, FP={fp}, TN={tn}")

    return {"acc": acc, "recall": recall, "precision": precision}


def evaluate_all_foils():
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATING RE-TRAINED INT8 MODEL ON 50 HELD-OUT TEST FOILS")
    print("=" * 80)

    tflite_path = "best_marvin_model_log_int8.tflite"
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    test_split_file = os.path.join(SPLIT_DIR, "test.txt")
    with open(test_split_file, "r") as f:
        test_lines = [l.strip() for l in f if l.strip()]

    test_foils = [l for l in test_lines if "hard_negatives" in l]
    print(f"Evaluating {len(test_foils)} held-out test hard-negative clips from {test_split_file}...")

    results_by_word = {}
    false_alarms = 0
    results_list = []

    for rel_path in test_foils:
        fpath = os.path.join(DATASET_PATH, rel_path)
        fname = os.path.basename(fpath)
        word = fname.split('_')[0]
        if word not in results_by_word:
            results_by_word[word] = {'total': 0, 'pred_marvin': 0, 'pred_unknown': 0, 'probs_m': []}

        with wave.open(fpath, 'rb') as wf:
            buf = wf.readframes(wf.getnframes())
        raw = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        _, feat = irdtv_optimized(raw, [1, 2, 4, 8, 16, 32], 64, 5, 24, log=True)
        interp.set_tensor(inp_idx, np.expand_dims(feat.astype(np.float32), axis=0))
        interp.invoke()
        probs = interp.get_tensor(out_idx)[0]

        p_m = float(probs[0])
        pred = int(np.argmax(probs))

        results_by_word[word]['total'] += 1
        results_by_word[word]['probs_m'].append(p_m)
        is_trigger = (pred == 0 or p_m >= 0.70)
        if is_trigger:
            results_by_word[word]['pred_marvin'] += 1
            false_alarms += 1
        else:
            results_by_word[word]['pred_unknown'] += 1

        results_list.append({
            'file': rel_path,
            'word': word,
            'p_marvin': round(p_m, 4),
            'pred': pred,
            'triggered': is_trigger
        })

    fa_pct = (false_alarms / len(test_foils)) * 100.0 if len(test_foils) > 0 else 0.0

    print("=" * 80)
    print(f"HELD-OUT TEST FOILS RESULTS ({len(test_foils)} CLIPS):")
    print(f"Overall False Trigger Rate (tau=0.70): {false_alarms}/{len(test_foils)} ({fa_pct:.2f}%)")
    print("-" * 80)
    print(f"{'Word':<12} | {'Total':<6} | {'False Trigger':<16} | {'Correct (Unknown)':<18} | {'Avg P(Marvin)':<14}")
    print("-" * 80)
    for word, data in sorted(results_by_word.items()):
        w_fa = data['pred_marvin'] / data['total'] * 100.0
        w_unk = data['pred_unknown'] / data['total'] * 100.0
        avg_pm = np.mean(data['probs_m']) * 100.0
        print(f"{word:<12} | {data['total']:6d} | {data['pred_marvin']:3d} ({w_fa:5.1f}%)        | {data['pred_unknown']:3d} ({w_unk:5.1f}%)           | {avg_pm:5.2f}%")
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    summary = {
        "total_test_foils": len(test_foils),
        "false_alarms": false_alarms,
        "false_alarm_rate_pct": round(fa_pct, 2),
        "threshold": 0.70,
        "results": results_list
    }
    with open("results/test_foils_evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "false_alarms": false_alarms,
        "fa_pct": fa_pct,
        "total_test_foils": len(test_foils),
        "results_by_word": results_by_word
    }


def update_c_headers():
    print("\n" + "=" * 80)
    print("STEP 5: UPDATING C HEADERS FOR ESP32-S3 FIRMWARE")
    print("=" * 80)

    tflite_path = "best_marvin_model_log_int8.tflite"
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()

    model_h_content = f"""// Auto-generated from {tflite_path} (Retrained with Hard Negatives)
// Model size: {len(model_bytes)} bytes
#ifndef TENET6_NARROW_MODEL_DATA_H_
#define TENET6_NARROW_MODEL_DATA_H_

#include <stdint.h>

const unsigned int g_tenet6_narrow_model_data_len = {len(model_bytes)};
alignas(16) const unsigned char g_tenet6_narrow_model_data[] = {{
"""
    hex_lines = []
    for i in range(0, len(model_bytes), 12):
        chunk = model_bytes[i:i+12]
        hex_lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    model_h_content += "\n".join(hex_lines) + "\n};\n\n#endif  // TENET6_NARROW_MODEL_DATA_H_\n"

    target_paths = [
        "save/tenet_narrow/tenet6_narrow_model_data.h",
        "save/tenet_narrow/ESP32S3_TENet_Benchmark/tenet6_narrow_model_data.h",
        "save/tenet_narrow/ESP32S3_TENet_RealTime_Pipeline/tenet6_narrow_model_data.h",
        r"C:\Users\sinth\Downloads\ESP32S3_TENet_Arduino_Package\ESP32S3_TENet_Benchmark\tenet6_narrow_model_data.h"
    ]

    for p in target_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(model_h_content)
        print(f"--> Updated: {p}")

    print("[SUCCESS] All C headers updated successfully!")


def main():
    train_model()
    freeze_and_quantize()
    test_metrics = evaluate_test_split()
    foil_metrics = evaluate_all_foils()
    update_c_headers()

    summary = {
        "test_metrics": test_metrics,
        "foil_metrics": {
            "false_alarms_50": foil_metrics["false_alarms_50"],
            "false_alarms_80": foil_metrics["false_alarms_80"],
            "fa_pct_50": foil_metrics["fa_pct_50"],
            "fa_pct_80": foil_metrics["fa_pct_80"],
        }
    }
    with open("retrain_hardneg_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[ALL COMPLETE] Full pipeline finished and logged to retrain_hardneg_results.json!")


if __name__ == "__main__":
    main()
