"""
Train and Benchmark DS-CNN (Depthwise Separable CNN) with iRDT Features.

Features:
- iRDT Feature Front-End (W=5, M=50, channels=[1,2,4,8,16,32], input shape: (50, 6, 1))
- Logarithmic transform log2(1 + Feat_spec) for robust INT8 quantization (<0.1% drop)
- Rigorous data augmentation: random time-shift (+/-100ms) + background noise mixing (80%, vol <= 0.10)
- DS-CNN Architecture with Depthwise Separable convolutions and residual connections
- Models trained:
    1. DS-CNN-15k (filters=52, 4 blocks, 15,187 params) -- matched to TENet6-Narrow
    2. DS-CNN-21k (filters=64, 4 blocks, 21,763 params) -- 20k parameter footprint
    3. DS-CNN-32k (filters=80, 4 blocks, 32,323 params) -- 30k parameter footprint
- Automatic INT8 TFLite export and evaluation
- Full TinyML benchmark: Accuracy, Recall, Precision, Latency, CPU Utilization, and Model Size
"""

import os
import sys
import time
import json
import wave
import psutil
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from irdt.irdtv import irdtv_optimized


SAMPLE_RATE = 16000
DESIRED_SAMPLES = 16000
CLASS_NAMES = ["marvin", "unknown", "background"]


def parse_split_file(split_path, data_root):
    paths = []
    labels = []
    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            full_path = os.path.join(data_root, line)
            w = line.replace("\\", "/").split("/")[0]
            if w == "marvin":
                label = 0
            elif w == "_silence_":
                label = 2
            else:
                label = 1
            paths.append(full_path)
            labels.append(label)
    return paths, np.array(labels, dtype=np.int64)


def load_raw_audio(path):
    with wave.open(path, "rb") as wf:
        buf = wf.readframes(DESIRED_SAMPLES)
        audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < DESIRED_SAMPLES:
        audio = np.pad(audio, (0, DESIRED_SAMPLES - len(audio)), mode="constant")
    return audio[:DESIRED_SAMPLES]


def load_bg_clips(data_root):
    bg_dir = os.path.join(data_root, "_background_noise_")
    clips = []
    if os.path.exists(bg_dir):
        for f in os.listdir(bg_dir):
            if f.endswith(".wav"):
                p = os.path.join(bg_dir, f)
                with wave.open(p, "rb") as wf:
                    buf = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
                    if len(audio) >= DESIRED_SAMPLES:
                        clips.append(audio)
    return clips


def augment_audio(audio, bg_clips, rng, shift_samples=1600, bg_freq=0.80, bg_max_vol=0.10):
    # 1. Random Time-Shift
    shift = rng.integers(-shift_samples, shift_samples + 1)
    if shift > 0:
        audio = np.pad(audio[:-shift], (shift, 0), mode="constant")
    elif shift < 0:
        audio = np.pad(audio[-shift:], (0, -shift), mode="constant")

    # 2. Background Noise Mixing
    if bg_clips and rng.random() < bg_freq:
        bg = bg_clips[rng.integers(0, len(bg_clips))]
        max_start = len(bg) - DESIRED_SAMPLES
        start = rng.integers(0, max_start + 1) if max_start > 0 else 0
        bg_slice = bg[start:start + DESIRED_SAMPLES]
        vol = rng.uniform(0.0, bg_max_vol)
        audio = np.clip(audio + vol * bg_slice, -1.0, 1.0)

    return audio


def extract_irdt_features(audio_list, chan_list=[1, 2, 4, 8, 16, 32], w=5, log=True):
    feats = []
    for audio in audio_list:
        _, f = irdtv_optimized(audio, chan_list, 64, w, 24, log)
        feats.append(f)
    return np.array(feats, dtype=np.float32)[:, :, :, np.newaxis]


def build_dscnn_model(filters=52, num_blocks=4, residual=True, num_classes=3, name=None):
    """
    Builds DS-CNN model optimized for iRDT feature input (50, 6, 1).
    """
    if name is None:
        name = f"DSCNN_{filters}f_{num_blocks}b"
    inp = layers.Input(shape=(50, 6, 1), name="irdt_input")
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False, name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    for i in range(num_blocks):
        res = x
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False, name=f"b{i+1}_dw")(x)
        x = layers.BatchNormalization(name=f"b{i+1}_dw_bn")(x)
        x = layers.ReLU(name=f"b{i+1}_dw_relu")(x)

        x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False, name=f"b{i+1}_pw")(x)
        x = layers.BatchNormalization(name=f"b{i+1}_pw_bn")(x)
        if residual:
            x = layers.add([res, x], name=f"b{i+1}_add")
        x = layers.ReLU(name=f"b{i+1}_pw_relu")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inputs=inp, outputs=out, name=name)


def convert_to_int8_tflite(model, rep_data, save_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_gen():
        for i in range(min(100, len(rep_data))):
            yield [rep_data[i:i+1].astype(np.float32)]

    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_bytes)
    return len(tflite_bytes)


def evaluate_tflite_model(tflite_path, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]

    in_scale, in_zero = in_details["quantization"]
    out_scale, out_zero = out_details["quantization"]

    preds = []
    for i in range(len(X_test)):
        sample = X_test[i:i+1]
        if in_details["dtype"] == np.int8:
            sample_q = np.clip(np.round(sample / in_scale) + in_zero, -128, 127).astype(np.int8)
        else:
            sample_q = sample.astype(np.float32)

        interpreter.set_tensor(in_details["index"], sample_q)
        interpreter.invoke()
        out = interpreter.get_tensor(out_details["index"])

        if out_details["dtype"] == np.int8:
            out_deq = (out.astype(np.float32) - out_zero) * out_scale
        else:
            out_deq = out

        preds.append(np.argmax(out_deq[0]))

    preds = np.array(preds)
    acc = float(np.mean(preds == y_test)) * 100.0

    # Binary KWS metrics for Marvin (class 0)
    tp = int(np.sum((preds == 0) & (y_test == 0)))
    fn = int(np.sum((preds != 0) & (y_test == 0)))
    fp = int(np.sum((preds == 0) & (y_test != 0)))
    tn = int(np.sum((preds != 0) & (y_test != 0)))

    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fa_rate = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0

    return {
        "acc": round(acc, 2),
        "recall": round(recall, 2),
        "precision": round(precision, 2),
        "fa_rate": round(fa_rate, 2),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn
    }


def benchmark_latency_and_cpu(model, sample_input, sample_audio, num_runs=200):
    # 1. Feature extraction latency
    feat_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = irdtv_optimized(sample_audio, [1, 2, 4, 8, 16, 32], 64, 5, 24, True)
        feat_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. Neural network inference latency
    # Warmup
    for _ in range(20):
        _ = model.predict(sample_input, verbose=0)

    nn_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model.predict(sample_input, verbose=0)
        nn_times.append((time.perf_counter() - t0) * 1000.0)

    # 3. CPU Utilization during active inference loop (sustained 1.0 second)
    proc = psutil.Process()
    proc.cpu_percent(interval=None)
    start = time.time()
    iters = 0
    while time.time() - start < 1.0:
        _ = model.predict(sample_input, verbose=0)
        iters += 1
    raw_cpu = proc.cpu_percent(interval=None)
    cpu_per_core = raw_cpu / max(1, psutil.cpu_count())

    return {
        "feat_mean_ms": round(float(np.mean(feat_times)), 3),
        "nn_mean_ms": round(float(np.mean(nn_times)), 3),
        "nn_median_ms": round(float(np.median(nn_times)), 3),
        "nn_p95_ms": round(float(np.percentile(nn_times, 95)), 3),
        "e2e_mean_ms": round(float(np.mean(feat_times)) + float(np.mean(nn_times)), 3),
        "cpu_util_normalized_pct": round(cpu_per_core, 1),
        "cpu_util_raw_pct": round(raw_cpu, 1)
    }


def train_single_dscnn(config, data, save_dir, seed=42):
    keras.utils.set_random_seed(seed)
    name = config["name"]
    filters = config["filters"]
    blocks = config["blocks"]
    epochs = config.get("epochs", 35)

    print(f"\n{'='*75}")
    print(f" TRAINING: {name} (Filters={filters}, Blocks={blocks})")
    print(f"{'='*75}")

    model = build_dscnn_model(filters=filters, num_blocks=blocks, residual=True, num_classes=3, name=name)
    total_params = model.count_params()
    print(f"--> Architecture: {name} | Total Parameters: {total_params:,}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    keras_path = os.path.join(save_dir, f"{name}.keras")
    tflite_fp32_path = os.path.join(save_dir, f"{name}_fp32.tflite")
    tflite_int8_path = os.path.join(save_dir, f"{name}_int8.tflite")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=keras_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc = float(np.max(history.history["val_accuracy"])) * 100.0
    print(f"--> [BEST VAL ACCURACY] {best_val_acc:.2f}%")

    # Evaluate FP32 on Test Set
    test_loss, test_acc_fp32 = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    test_acc_fp32 *= 100.0

    raw_preds = model.predict(data["X_test"], verbose=0)
    preds = np.argmax(raw_preds, axis=-1)

    tp = int(np.sum((preds == 0) & (data["y_test"] == 0)))
    fn = int(np.sum((preds != 0) & (data["y_test"] == 0)))
    fp = int(np.sum((preds == 0) & (data["y_test"] != 0)))
    tn = int(np.sum((preds != 0) & (data["y_test"] != 0)))

    recall_fp32 = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    precision_fp32 = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fa_rate_fp32 = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0

    # Save FP32 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32_tflite_bytes = converter.convert()
    with open(tflite_fp32_path, "wb") as f:
        f.write(fp32_tflite_bytes)
    fp32_size_kb = len(fp32_tflite_bytes) / 1024.0

    # Export & Evaluate INT8 Quantized TFLite
    print(f"--> Exporting INT8 quantized TFLite to {tflite_int8_path}...")
    int8_size_bytes = convert_to_int8_tflite(model, data["X_train"][:300], tflite_int8_path)
    int8_size_kb = int8_size_bytes / 1024.0

    int8_metrics = evaluate_tflite_model(tflite_int8_path, data["X_test"], data["y_test"])
    print(f"--> [INT8 EVALUATION] Accuracy: {int8_metrics['acc']:.2f}% | Recall: {int8_metrics['recall']:.2f}% | Precision: {int8_metrics['precision']:.2f}%")

    # Latency & CPU Benchmark
    print("--> Benchmarking Latency and CPU Utilization...")
    sample_inp = data["X_test"][:1]
    sample_aud = data["raw_test_audio"][0]
    perf = benchmark_latency_and_cpu(model, sample_inp, sample_aud)

    return {
        "name": name,
        "filters": filters,
        "blocks": blocks,
        "parameters": total_params,
        "keras_path": keras_path,
        "tflite_fp32_path": tflite_fp32_path,
        "tflite_int8_path": tflite_int8_path,
        "fp32_size_kb": round(fp32_size_kb, 2),
        "int8_size_kb": round(int8_size_kb, 2),
        "best_val_acc": round(best_val_acc, 2),
        "fp32_metrics": {
            "test_acc": round(test_acc_fp32, 2),
            "recall": round(recall_fp32, 2),
            "precision": round(precision_fp32, 2),
            "fa_rate": round(fa_rate_fp32, 2),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn
        },
        "int8_metrics": int8_metrics,
        "performance": perf
    }


def main():
    data_root = r"C:\XboxGames\Dataset\speech_commands_v0.02"
    split_dir = "configs/v1_irdt"
    save_dir = "save/dscnn_irdt"
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print(" DS-CNN WITH iRDT TRAINING & BENCHMARK PIPELINE")
    print(f" Data Root:  {data_root}")
    print(f" Save Dir:   {save_dir}")
    print("=" * 75)

    # 1. Load Split Lists
    train_paths, train_y = parse_split_file(os.path.join(split_dir, "train.txt"), data_root)
    val_paths, val_y = parse_split_file(os.path.join(split_dir, "valid.txt"), data_root)
    test_paths, test_y = parse_split_file(os.path.join(split_dir, "test.txt"), data_root)
    print(f"Dataset Loaded: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")

    # 2. Load Background Noise Clips for Augmentation
    bg_clips = load_bg_clips(data_root)
    print(f"Loaded {len(bg_clips)} background noise files.")

    # 3. Load Raw Audio into Memory
    print("Preloading raw audio waveforms into RAM...")
    raw_train = [load_raw_audio(p) for p in train_paths]
    raw_val = [load_raw_audio(p) for p in val_paths]
    raw_test = [load_raw_audio(p) for p in test_paths]

    # 4. Generate Multi-Pass Augmented Training Data (4 passes = 12,800 samples)
    print("Synthesizing multi-pass augmented training set (time shift +/-100ms, noise mix 80%)...")
    rng = np.random.default_rng(42)
    aug_train_audio = []
    aug_train_y = []

    # Pass 0: Clean unaugmented
    aug_train_audio.extend(raw_train)
    aug_train_y.extend(train_y)

    # Passes 1..3: Augmented
    for _ in range(3):
        for audio, lbl in zip(raw_train, train_y):
            aug_audio = augment_audio(audio.copy(), bg_clips, rng)
            aug_train_audio.append(aug_audio)
            aug_train_y.append(lbl)

    aug_train_y = np.array(aug_train_y, dtype=np.int64)
    print(f"Total Augmented Training Samples: {len(aug_train_audio)}")

    # 5. Extract iRDT Features (Log Transform enabled)
    print("Extracting iRDT features (W=5, M=50, log2(1+x))...")
    t0 = time.time()
    X_train = extract_irdt_features(aug_train_audio, log=True)
    X_val = extract_irdt_features(raw_val, log=True)
    X_test = extract_irdt_features(raw_test, log=True)
    print(f"Features extracted in {time.time()-t0:.2f}s! Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Standardize based on training set statistics
    train_mean = float(np.mean(X_train))
    train_std = float(np.std(X_train)) + 1e-7
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    data_bundle = {
        "X_train": X_train, "y_train": aug_train_y,
        "X_val": X_val, "y_val": val_y,
        "X_test": X_test, "y_test": test_y,
        "raw_test_audio": raw_test,
        "norm_stats": {"mean": train_mean, "std": train_std}
    }

    # Save normalization stats
    with open(os.path.join(save_dir, "normalization_stats.json"), "w") as f:
        json.dump(data_bundle["norm_stats"], f, indent=2)

    # 6. Model Configurations to Train
    configs = [
        {"name": "dscnn_irdt_15k", "filters": 52, "blocks": 4, "epochs": 35},
        {"name": "dscnn_irdt_21k", "filters": 64, "blocks": 4, "epochs": 35},
        {"name": "dscnn_irdt_32k", "filters": 80, "blocks": 4, "epochs": 35},
    ]

    all_results = {}
    for cfg in configs:
        res = train_single_dscnn(cfg, data_bundle, save_dir, seed=42)
        all_results[cfg["name"]] = res

    # 7. Save Benchmark Results
    output_json = os.path.join(save_dir, "dscnn_irdt_benchmark_results.json")
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n--> Benchmark results saved to {output_json}")

    # 8. Display Formatted Benchmark Comparison Table
    print("\n" + "=" * 115)
    print(f"{'Model Architecture':<22} | {'Params':<8} | {'FP32 Size':<10} | {'INT8 Size':<10} | {'FP32 Acc':<9} | {'INT8 Acc':<9} | {'Recall':<8} | {'Precision':<10} | {'Latency':<9} | {'CPU Util'}")
    print("-" * 115)
    for k, v in all_results.items():
        fp32_m = v["fp32_metrics"]
        int8_m = v["int8_metrics"]
        perf = v["performance"]
        print(f"{v['name']:<22} | {v['parameters']:<8,d} | {v['fp32_size_kb']:<7.1f} KB | {v['int8_size_kb']:<7.1f} KB | {fp32_m['test_acc']:<7.2f}% | {int8_m['acc']:<7.2f}% | {int8_m['recall']:<6.1f}% | {int8_m['precision']:<8.1f}% | {perf['e2e_mean_ms']:<6.2f} ms | {perf['cpu_util_normalized_pct']:.1f}%")
    print("=" * 115)


if __name__ == "__main__":
    main()
