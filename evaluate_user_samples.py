"""
Inference script to evaluate user-provided audio samples in `clipped_1sec` folder
using:
1. Best Single Model: Seed 101 (W=5 TENet6-Narrow + iRDT)
2. Best Ensemble: 3-Seed Soft Average Ensemble (Seed 42, 101, 2024)
3. Individual Seed 42 & Seed 2024 models for complete audit trail.
"""

import os
import glob
import wave
import json
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import models
from irdt.irdtv import irdtv_optimized


def load_audio_and_extract_features(folder_path, w=5, chan_list=[1, 2, 4, 8, 16, 32], irdt_log=False):
    wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {folder_path}")

    features = []
    file_info = []

    print(f"Found {len(wav_files)} audio samples in {folder_path}.\nExtracting iRDT features...")
    for fpath in wav_files:
        fname = os.path.basename(fpath)
        with wave.open(fpath, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            buf = wf.readframes(n_frames)

        audio_int16 = np.frombuffer(buf, dtype=np.int16)
        if n_channels > 1:
            audio_int16 = audio_int16[::n_channels]  # take first channel

        audio = audio_int16.astype(np.float32) / 32768.0

        # Resample or pad/crop to 16,000 samples
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)), mode="constant")
        elif len(audio) > 16000:
            audio = audio[:16000]

        # Calculate signal RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2)))

        # Extract iRDT feature: shape (50, 6)
        _, feat = irdtv_optimized(audio, chan_list, 64, w, 24, irdt_log)
        features.append(feat)

        file_info.append({
            "filename": fname,
            "path": fpath,
            "samplerate": framerate,
            "orig_frames": n_frames,
            "rms_energy": rms
        })

    X = np.array(features, dtype=np.float32)
    return X, file_info


def get_model_probs(X, ckpt_path, batch_size=100):
    g = tf.Graph()
    with g.as_default():
        feat_ph = tf.compat.v1.placeholder(tf.float32, [None, X.shape[1], X.shape[2]])
        model_args = argparse.Namespace(
            sample_rate=16000, window_size_ms=30, window_stride_ms=10,
            background_frequency=0.8, background_max_volume=0.1, num_mel_bins=64,
            desired_samples=16000, lower_edge_hertz=80.0, upper_edge_hertz=7600.0,
            num_mfccs=40, timeshift_ms=100, wanted_words=["marvin"],
            dataset_path="", split_dir="", steps=3000,
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
            p = sess.run(probs_tensor, feed_dict={feat_ph: batch_X})
            all_probs.append(p)

        sess.close()
    return np.vstack(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=r"C:\TENet-kws-irdt\clipped_1sec")
    parser.add_argument("--base_save_dir", default="save/ensemble_w5")
    parser.add_argument("--out_json", default="user_samples_inference_results.json")
    args = parser.parse_args()

    X, file_info = load_audio_and_extract_features(args.folder, w=5)

    ckpt_42 = os.path.join(args.base_save_dir, "seed_42", "TENet6NarrowIRDTModel-3000")
    ckpt_101 = os.path.join(args.base_save_dir, "seed_101", "TENet6NarrowIRDTModel-3000")
    ckpt_2024 = os.path.join(args.base_save_dir, "seed_2024", "TENet6NarrowIRDTModel-3000")

    print("\nRunning inference for Seed 101 (Best Single Model)...")
    p_101 = get_model_probs(X, ckpt_101)

    print("Running inference for Seed 42...")
    p_42 = get_model_probs(X, ckpt_42)

    print("Running inference for Seed 2024...")
    p_2024 = get_model_probs(X, ckpt_2024)

    # 3-Seed Ensemble soft average
    p_ens = (p_42 + p_101 + p_2024) / 3.0

    classes = ["marvin", "_unknown_", "_silence_"]

    results = []
    print("\n" + "=" * 115)
    print(f"{'#':<3} | {'Filename':<32} | {'Ensemble P(marvin)':<18} | {'Seed101 P(marvin)':<17} | {'Ens Decision':<13} | {'RMS':<7}")
    print("=" * 115)

    ens_detected_50 = 0
    ens_detected_80 = 0
    ens_detected_95 = 0

    s101_detected_50 = 0
    s101_detected_80 = 0
    s101_detected_95 = 0

    for i in range(len(file_info)):
        fname = file_info[i]["filename"]
        rms = file_info[i]["rms_energy"]

        # Ensemble
        ens_marvin_prob = float(p_ens[i, 0])
        ens_unk_prob = float(p_ens[i, 1])
        ens_noise_prob = float(p_ens[i, 2])
        ens_pred_class = classes[int(np.argmax(p_ens[i]))]

        # Seed 101
        s101_marvin_prob = float(p_101[i, 0])
        s101_unk_prob = float(p_101[i, 1])
        s101_noise_prob = float(p_101[i, 2])
        s101_pred_class = classes[int(np.argmax(p_101[i]))]

        # Counts
        if ens_marvin_prob >= 0.50: ens_detected_50 += 1
        if ens_marvin_prob >= 0.80: ens_detected_80 += 1
        if ens_marvin_prob >= 0.95: ens_detected_95 += 1

        if s101_marvin_prob >= 0.50: s101_detected_50 += 1
        if s101_marvin_prob >= 0.80: s101_detected_80 += 1
        if s101_marvin_prob >= 0.95: s101_detected_95 += 1

        status_str = "DETECTION" if ens_pred_class == "marvin" else f"MISSED ({ens_pred_class})"

        print(f"{i+1:2d}  | {fname:<32} | {ens_marvin_prob*100:6.2f}%             | {s101_marvin_prob*100:6.2f}%            | {status_str:<13} | {rms:.4f}")

        sample_res = {
            "index": i + 1,
            "filename": fname,
            "rms_energy": rms,
            "ensemble": {
                "p_marvin": ens_marvin_prob,
                "p_unknown": ens_unk_prob,
                "p_silence": ens_noise_prob,
                "predicted_label": ens_pred_class,
                "detected_tau_0_50": bool(ens_marvin_prob >= 0.50),
                "detected_tau_0_80": bool(ens_marvin_prob >= 0.80),
                "detected_tau_0_95": bool(ens_marvin_prob >= 0.95)
            },
            "seed_101": {
                "p_marvin": s101_marvin_prob,
                "p_unknown": s101_unk_prob,
                "p_silence": s101_noise_prob,
                "predicted_label": s101_pred_class,
                "detected_tau_0_50": bool(s101_marvin_prob >= 0.50),
                "detected_tau_0_80": bool(s101_marvin_prob >= 0.80),
                "detected_tau_0_95": bool(s101_marvin_prob >= 0.95)
            },
            "seed_42": {
                "p_marvin": float(p_42[i, 0]),
                "p_unknown": float(p_42[i, 1]),
                "p_silence": float(p_42[i, 2])
            },
            "seed_2024": {
                "p_marvin": float(p_2024[i, 0]),
                "p_unknown": float(p_2024[i, 1]),
                "p_silence": float(p_2024[i, 2])
            }
        }
        results.append(sample_res)

    print("=" * 115)
    total = len(file_info)
    print(f"\nSUMMARY STATISTICS (Total Samples = {total}):")
    print("-" * 65)
    print(f"3-Seed Ensemble Performance:")
    print(f"  - Detected at tau=0.50 (argmax):  {ens_detected_50}/{total} ({ens_detected_50/total*100:.2f}%)")
    print(f"  - Detected at tau=0.80 (balanced): {ens_detected_80}/{total} ({ens_detected_80/total*100:.2f}%)")
    print(f"  - Detected at tau=0.95 (ultra-strict): {ens_detected_95}/{total} ({ens_detected_95/total*100:.2f}%)")
    print("-" * 65)
    print(f"Best Single Model (Seed 101) Performance:")
    print(f"  - Detected at tau=0.50 (argmax):  {s101_detected_50}/{total} ({s101_detected_50/total*100:.2f}%)")
    print(f"  - Detected at tau=0.80 (balanced): {s101_detected_80}/{total} ({s101_detected_80/total*100:.2f}%)")
    print(f"  - Detected at tau=0.95 (ultra-strict): {s101_detected_95}/{total} ({s101_detected_95/total*100:.2f}%)")
    print("=" * 65)

    summary = {
        "total_samples": total,
        "ensemble_detected_tau_0_50": ens_detected_50,
        "ensemble_detected_tau_0_80": ens_detected_80,
        "ensemble_detected_tau_0_95": ens_detected_95,
        "seed101_detected_tau_0_50": s101_detected_50,
        "seed101_detected_tau_0_80": s101_detected_80,
        "seed101_detected_tau_0_95": s101_detected_95,
        "samples": results
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to {args.out_json}")


if __name__ == "__main__":
    main()
