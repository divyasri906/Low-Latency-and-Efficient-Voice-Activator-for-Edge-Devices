import os
import sys
import argparse
import glob
import wave
import json
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import models
from irdt.irdtv import irdtv_optimized
from evaluate_optimized_w5 import load_split_data

DATASET_PATH = r"C:\XboxGames\Dataset\speech_commands_v0.02"
SPLIT_DIR = "configs/v1_irdt_hardneg"

def freeze_and_quantize(ckpt_path, out_tflite_path="best_marvin_model_log_int8.tflite"):
    print("=" * 80)
    print(f"FREEZING CHECKPOINT: {ckpt_path}")
    print("=" * 80)

    save_dir = os.path.dirname(ckpt_path)
    pb_path = os.path.join(save_dir, "frozen_checkpoint.pb")

    g = tf.Graph()
    with g.as_default():
        feat_ph = tf.compat.v1.placeholder(tf.float32, [None, 50, 6], name='input_features')
        model_args = argparse.Namespace(
            sample_rate=16000, window_size_ms=30, window_stride_ms=10,
            background_frequency=0.8, background_max_volume=0.1, num_mel_bins=64,
            desired_samples=16000, lower_edge_hertz=80.0, upper_edge_hertz=7600.0,
            num_mfccs=40, timeshift_ms=100, wanted_words=['marvin'],
            dataset_path='', split_dir='', steps=3000,
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
        print(f"--> Frozen protobuf graph saved ({os.path.getsize(pb_path)} bytes)")
        sess.close()

    print("Loading calibration data for INT8 quantization...")
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

    with open(out_tflite_path, "wb") as f:
        f.write(tflite_quant)

    print(f"--> [SUCCESS] Full INT8 TFLite model generated: {out_tflite_path} ({len(tflite_quant)/1024:.2f} KB)")
    return out_tflite_path


def evaluate_tflite_on_splits(tflite_path):
    print("\n" + "=" * 80)
    print("EVALUATING QUANTIZED INT8 TFLITE MODEL")
    print("=" * 80)

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    # 1. Test Split
    test_file = os.path.join(SPLIT_DIR, "test.txt")
    X_test, y_test, _, _ = load_split_data(DATASET_PATH, test_file, w=5, irdt_log=True)

    preds = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1].astype(np.float32)
        interp.set_tensor(inp_idx, sample)
        interp.invoke()
        p = interp.get_tensor(out_idx)[0]
        preds.append(np.argmax(p))

    preds = np.array(preds)
    acc = float(np.mean(preds == y_test) * 100.0)
    y_m = (y_test == 0)
    p_m = (preds == 0)
    tp = int(np.sum(y_m & p_m))
    fn = int(np.sum(y_m & ~p_m))
    fp = int(np.sum(~y_m & p_m))
    tn = int(np.sum(~y_m & ~p_m))
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0

    print(f"--- TEST SPLIT (N = {len(y_test)}) ---")
    print(f"Overall Accuracy   : {acc:.2f}%")
    print(f"Marvin Recall      : {recall:.2f}% ({tp}/{tp+fn})")
    print(f"Marvin Precision   : {precision:.2f}% ({tp}/{tp+fp})")
    print(f"Confusion Matrix   : TP={tp}, FN={fn}, FP={fp}, TN={tn}")

    # 2. Hard Negative Foils
    files = sorted(glob.glob(os.path.join(DATASET_PATH, "hard_negatives", "*.wav")))
    results_by_word = {}
    false_alarms_50 = 0
    false_alarms_80 = 0

    for fpath in files:
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

        p_m, p_u, p_s = float(probs[0]), float(probs[1]), float(probs[2])
        pred = np.argmax(probs)

        results_by_word[word]['total'] += 1
        results_by_word[word]['probs_m'].append(p_m)
        if pred == 0:
            results_by_word[word]['pred_marvin'] += 1
            false_alarms_50 += 1
        elif pred == 1:
            results_by_word[word]['pred_unknown'] += 1

        if p_m >= 0.80:
            false_alarms_80 += 1

    fa_pct_50 = false_alarms_50 / len(files) * 100.0
    print("\n--- HARD-NEGATIVE FOILS (N = 462) ---")
    print(f"Overall False Trigger Rate (tau=0.50): {false_alarms_50}/{len(files)} ({fa_pct_50:.2f}%)")
    print(f"Overall False Trigger Rate (tau=0.80): {false_alarms_80}/{len(files)} ({false_alarms_80/len(files)*100:.2f}%)")
    print("-" * 80)
    print(f"{'Word':<12} | {'Total':<6} | {'False Trigger (Marvin)':<24} | {'Correct (Unknown)':<18} | {'Avg P(Marvin)':<14}")
    print("-" * 80)
    for word, data in sorted(results_by_word.items()):
        w_fa = data['pred_marvin'] / data['total'] * 100.0
        w_unk = data['pred_unknown'] / data['total'] * 100.0
        avg_pm = np.mean(data['probs_m']) * 100.0
        print(f"{word:<12} | {data['total']:6d} | {data['pred_marvin']:3d} ({w_fa:5.1f}%)               | {data['pred_unknown']:3d} ({w_unk:5.1f}%)           | {avg_pm:5.2f}%")
    print("=" * 80)


def update_headers(tflite_path):
    print("\nUpdating C headers...")
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
        print(f"--> Exported header: {p}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "save/tenet_log_hardneg/TENet6NarrowIRDTModel-1000"
    tflite = freeze_and_quantize(ckpt)
    evaluate_tflite_on_splits(tflite)
    update_headers(tflite)
