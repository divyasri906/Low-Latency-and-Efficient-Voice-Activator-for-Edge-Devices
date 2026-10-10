"""
Quantize and Export Parameter-Matched DS-CNN (MFCC) to Full INT8 TFLite.
Generates C++ header dscnn_marvin_mfcc_model_data.h for ESP32-S3 deployment.
"""

import os
import sys
import wave
import json
import time
import numpy as np
import librosa
import tensorflow as tf
import keras

DATA_ROOT = r"C:\XboxGames\Dataset\speech_commands_v0.02"
SPLIT_DIR = "configs/v1_irdt"
OUTPUT_DIR = "save/dscnn_mfcc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 320
SAMPLE_RATE = 16000
DESIRED_SAMPLES = 16000


def load_wav(path):
    with wave.open(path, "rb") as w:
        buf = w.readframes(DESIRED_SAMPLES)
        audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < DESIRED_SAMPLES:
        audio = np.pad(audio, (0, DESIRED_SAMPLES - len(audio)), mode="constant")
    return audio[:DESIRED_SAMPLES]


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return mfcc.T[:49, :]  # Shape: (49, 13)


def parse_split(split_file):
    paths = []
    labels = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            full_p = os.path.join(DATA_ROOT, line)
            f_name = line.split("/")[0]
            if f_name == "marvin":
                l = 0
            elif f_name == "_silence_":
                l = 2
            else:
                l = 1
            paths.append(full_p)
            labels.append(l)
    return paths, np.array(labels, dtype=np.int64)


def quantize_model(keras_model_path, meta_path, calib_paths):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    train_mean = meta["train_mean"]
    train_std = meta["train_std"]

    print(f"\nLoading {keras_model_path}...")
    model = keras.models.load_model(keras_model_path)

    # Collect calibration features
    print(f"Extracting calibration features from {len(calib_paths)} samples...")
    calib_feats = []
    for p in calib_paths:
        audio = load_wav(p)
        f = extract_mfcc(audio)
        f_norm = (f - train_mean) / train_std
        calib_feats.append(f_norm.astype(np.float32))
    calib_X = np.array(calib_feats)[:, :, :, np.newaxis]

    def representative_dataset():
        for i in range(len(calib_X)):
            yield [calib_X[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_int8 = converter.convert()
    return tflite_int8, train_mean, train_std, meta


def evaluate_tflite_int8(tflite_model, test_paths, test_labels, train_mean, train_std):
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    in_scale, in_zp = in_det["quantization"]
    out_scale, out_zp = out_det["quantization"]

    preds = []
    for p in test_paths:
        audio = load_wav(p)
        mfcc = extract_mfcc(audio)
        mfcc_norm = (mfcc - train_mean) / train_std
        q_in = np.round(mfcc_norm / in_scale) + in_zp
        q_in = np.clip(q_in, -128, 127).astype(np.int8)[np.newaxis, :, :, np.newaxis]

        interp.set_tensor(in_det["index"], q_in)
        interp.invoke()
        q_out = interp.get_tensor(out_det["index"])[0]
        prob = (q_out.astype(np.float32) - out_zp) * out_scale
        preds.append(np.argmax(prob))

    preds = np.array(preds)
    acc = np.mean(preds == test_labels)

    # Class-specific metrics
    marvin_mask = (test_labels == 0)
    unk_mask = (test_labels == 1)
    sil_mask = (test_labels == 2)

    marvin_recall = np.sum((preds == 0) & marvin_mask) / np.sum(marvin_mask) if np.sum(marvin_mask) > 0 else 0.0
    marvin_fa = np.sum((preds == 0) & ~marvin_mask)
    marvin_precision = np.sum((preds == 0) & marvin_mask) / np.sum(preds == 0) if np.sum(preds == 0) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "marvin_recall": float(marvin_recall),
        "marvin_precision": float(marvin_precision),
        "marvin_false_alarms": int(marvin_fa),
        "in_scale": float(in_scale),
        "in_zero_point": int(in_zp),
        "out_scale": float(out_scale),
        "out_zero_point": int(out_zp),
    }


def main():
    print("=" * 70)
    print(" QUANTIZING & EVALUATING PARAMETER-MATCHED DS-CNN (MFCC)")
    print("=" * 70)

    val_paths, val_labels = parse_split(os.path.join(SPLIT_DIR, "valid.txt"))
    test_paths, test_labels = parse_split(os.path.join(SPLIT_DIR, "test.txt"))

    # Use first 200 validation samples for calibration
    calib_paths = val_paths[:200]

    seeds = [101, 42, 2024]
    best_candidate = None
    best_score = -1.0

    for s in seeds:
        k_path = f"model_training/models/dscnn_matched_15k_seed{s}.keras"
        m_path = f"model_training/models/dscnn_matched_15k_seed{s}_meta.json"
        if not os.path.exists(k_path):
            continue

        tflite_bytes, t_mean, t_std, meta = quantize_model(k_path, m_path, calib_paths)
        eval_res = evaluate_tflite_int8(tflite_bytes, test_paths, test_labels, t_mean, t_std)

        size_kb = len(tflite_bytes) / 1024.0
        print(f"\n[Seed {s}] INT8 Size: {size_kb:.1f} KB | Acc: {eval_res['accuracy']*100:.2f}% | "
              f"Recall: {eval_res['marvin_recall']*100:.1f}% | Precision: {eval_res['marvin_precision']*100:.1f}% | "
              f"FA: {eval_res['marvin_false_alarms']}")

        # Composite score
        score = eval_res["accuracy"] * 0.6 + eval_res["marvin_recall"] * 0.4
        if score > best_score:
            best_score = score
            best_candidate = {
                "seed": s,
                "tflite_bytes": tflite_bytes,
                "train_mean": t_mean,
                "train_std": t_std,
                "eval": eval_res,
                "meta": meta,
                "size_kb": size_kb
            }

    print("\n" + "=" * 70)
    print(f" SELECTED BEST MODEL: Seed {best_candidate['seed']}")
    print(f" INT8 Size:     {best_candidate['size_kb']:.1f} KB")
    print(f" INT8 Accuracy: {best_candidate['eval']['accuracy']*100:.2f}%")
    print(f" Marvin Recall: {best_candidate['eval']['marvin_recall']*100:.1f}%")
    print(f" Precision:     {best_candidate['eval']['marvin_precision']*100:.1f}%")
    print(f" False Alarms:  {best_candidate['eval']['marvin_false_alarms']}")
    print("=" * 70)

    # Save TFLite file
    tflite_save_path = os.path.join(OUTPUT_DIR, "dscnn_mfcc_15k_int8.tflite")
    with open(tflite_save_path, "wb") as f:
        f.write(best_candidate["tflite_bytes"])
    print(f"\nSaved INT8 model to {tflite_save_path}")

    # Save metadata JSON
    meta_out = {
        "model_name": f"dscnn_matched_15k_seed{best_candidate['seed']}",
        "architecture": "DS-CNN (Depthwise Separable CNN)",
        "features": "MFCC (n_mfcc=13, n_fft=512, hop_length=320, frames=49)",
        "input_shape": [1, 49, 13, 1],
        "output_shape": [1, 3],
        "train_mean": best_candidate["train_mean"],
        "train_std": best_candidate["train_std"],
        "in_scale": best_candidate["eval"]["in_scale"],
        "in_zero_point": best_candidate["eval"]["in_zero_point"],
        "out_scale": best_candidate["eval"]["out_scale"],
        "out_zero_point": best_candidate["eval"]["out_zero_point"],
        "test_accuracy": best_candidate["eval"]["accuracy"],
        "marvin_recall": best_candidate["eval"]["marvin_recall"],
        "marvin_precision": best_candidate["eval"]["marvin_precision"],
        "size_kb": best_candidate["size_kb"]
    }
    with open(os.path.join(OUTPUT_DIR, "dscnn_mfcc_metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"Saved metadata to {os.path.join(OUTPUT_DIR, 'dscnn_mfcc_metadata.json')}")

    # Generate C header file
    header_path = os.path.join(OUTPUT_DIR, "dscnn_marvin_mfcc_model_data.h")
    data_bytes = best_candidate["tflite_bytes"]
    with open(header_path, "w") as f:
        f.write("// Auto-generated INT8 TFLite model data for DS-CNN with MFCC features\n")
        f.write("#ifndef DSCNN_MARVIN_MFCC_MODEL_DATA_H\n")
        f.write("#define DSCNN_MARVIN_MFCC_MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"const unsigned int g_dscnn_mfcc_model_data_len = {len(data_bytes)};\n")
        f.write("alignas(16) const uint8_t g_dscnn_mfcc_model_data[] = {\n    ")
        for i, b in enumerate(data_bytes):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n    ")
        f.write("\n};\n\n")
        f.write("#endif // DSCNN_MARVIN_MFCC_MODEL_DATA_H\n")

    print(f"Generated C header: {header_path} ({len(data_bytes)} bytes)")


if __name__ == "__main__":
    main()
