# TENet6-Narrow + iRDT Keyword Spotting Model ("Marvin")

Ultra-low footprint, real-time Keyword Spotting (KWS) model combining **Integrated Rate-Distortion Theory (iRDT)** temporal delay features with a **Narrow Temporal Edge Network (TENet6-Narrow)**.

---

## 1. Specifications & Footprint

| Specification | Value | Notes |
| :--- | :--- | :--- |
| **Target Keyword** | `"marvin"` | 3-class classification (`marvin`, `_unknown_`, `_silence_`) |
| **Model Architecture** | `TENet6NarrowIRDTModel` | 6 Temporal Conv Blocks + Depthwise Separable Conv |
| **Parameter Count** | **15,120** | Only **14.77 KB** (INT8) / **59.06 KB** (FP32) |
| **Computational Cost** | **236,112 MACs** | 0.24 MMACs per 1.0s inference |
| **End-to-End Latency** | **1.84 ms** | Real-time factor: < 0.002× (ARM Cortex-M / Edge CPU) |
| **Peak SRAM Usage** | **3.20 KB** | Runs comfortably within limited MCU memory (e.g. STM32, ESP32) |
| **Input Audio** | 16 kHz, 16-bit Mono PCM | 1.0 second window = 16,000 samples |
| **Front-End Feature** | iRDT ($W=5$, $M=50$, $C=6$) | $A$-type delays: `[1, 2, 4, 8, 16, 32]`, 64 filterbanks |

---

## 2. Benchmark Performance

Evaluated on the Google Speech Commands v2 test split (400 samples: 100 Marvins, 200 non-target speech words, 100 ambient noise segments) with true moving BatchNorm statistics (`is_training=False`):

### Single Model (Seed 101 — Best Standalone Model)

| Operating Threshold ($\tau$) | Overall Accuracy | Marvin Recall | Speech False Alarm Rate | Background Noise FA Rate |
| :--- | :---: | :---: | :---: | :---: |
| **$\tau = 0.50$ (Argmax)** | **97.25%** | **99.00%** (99/100) | 3.33% (10/200) | **0.00%** (0/100) |
| **$\tau = 0.80$ (Balanced)** | **97.50%** | **95.00%** (95/100) | 1.67% (5/200) | **0.00%** (0/100) |
| **$\tau = 0.95$ (High Precision)**| **97.00%** | **88.00%** (88/100) | **0.33%** (1/200) | **0.00%** (0/100) |

### 3-Seed Soft Average Ensemble ($P_{\text{ens}} = \frac{P_{42} + P_{101} + P_{2024}}{3}$)

| Operating Threshold ($\tau$) | Overall Accuracy | Marvin Recall | Speech False Alarm Rate | Background Noise FA Rate |
| :--- | :---: | :---: | :---: | :---: |
| **$\tau = 0.50$ (Argmax)** | **97.25%** | **99.00%** (99/100) | 3.33% (10/200) | **0.00%** (0/100) |
| **$\tau = 0.95$ (Zero-FA Mode)** | **97.75%** | **91.00%** (91/100) | **0.00%** (0/300) | **0.00%** (0/100) |

---

## 3. Actual Model Files on Disk

The model is saved natively using standard TensorFlow checkpoint format. All files are located in:

### Best Single Model Folder: `save/ensemble_w5/seed_101/`

| Exact File Name | File Size | Description |
| :--- | :---: | :--- |
| **`TENet6NarrowIRDTModel-3000.data-00000-of-00001`** | **63,424 bytes** | Model weights (variables) at final step 3000 |
| **`TENet6NarrowIRDTModel-3000.index`** | **4,195 bytes** | TensorFlow variable key-to-offset index |
| **`TENet6NarrowIRDTModel-3000.meta`** | **1,566,541 bytes** | Complete TensorFlow graph definition & metadata |
| **`checkpoint`** | **337 bytes** | Plaintext manifest specifying `TENet6NarrowIRDTModel-3000` |

### Multi-Seed Ensemble Checkpoint Folders

- Seed 42: `save/ensemble_w5/seed_42/`
  - `TENet6NarrowIRDTModel-3000.data-00000-of-00001`
  - `TENet6NarrowIRDTModel-3000.index`
  - `TENet6NarrowIRDTModel-3000.meta`
  - `checkpoint`
- Seed 2024: `save/ensemble_w5/seed_2024/`
  - `TENet6NarrowIRDTModel-3000.data-00000-of-00001`
  - `TENet6NarrowIRDTModel-3000.index`
  - `TENet6NarrowIRDTModel-3000.meta`
  - `checkpoint`

---

## 4. Class Indices & Output Probability Format

The model outputs a 3-element probability vector $\mathbf{P} = [p_0, p_1, p_2]$ normalized via Softmax:

- **Index 0 (`marvin`)**: Probability that the utterance is the target keyword "Marvin".
- **Index 1 (`_unknown_`)**: Probability that the utterance is other human speech (e.g. *dog, happy, stop, yes, no*).
- **Index 2 (`_silence_`)**: Probability that the audio is ambient background noise or silence.

---

## 5. How to Load and Run the Checkpoint in Python

```python
import os
import argparse
import numpy as np
import tensorflow as tf
from irdt.irdtv import irdtv_optimized
import models

tf.compat.v1.disable_eager_execution()

# 1. Build Model Graph
feat_ph = tf.compat.v1.placeholder(tf.float32, [None, 50, 6])
model_args = argparse.Namespace(
    sample_rate=16000, window_size_ms=30, window_stride_ms=10,
    background_frequency=0.8, background_max_volume=0.1, num_mel_bins=64,
    desired_samples=16000, lower_edge_hertz=80.0, upper_edge_hertz=7600.0,
    num_mfccs=40, timeshift_ms=100, wanted_words=["marvin"],
    dataset_path="", split_dir="", steps=3000,
    batch_size=1, buffer_size=1000, workers=1,
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

model = models.TENet6NarrowIRDTModel(model_args)
logits_op = model.build_inference(feat_ph, is_training=False)
probs_tensor = tf.nn.softmax(logits_op)

# 2. Restore Checkpoint Weights
sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.Saver()
checkpoint_prefix = "save/ensemble_w5/seed_101/TENet6NarrowIRDTModel-3000"
saver.restore(sess, checkpoint_prefix)

# 3. Extract Features and Predict
# audio: float32 numpy array normalized to [-1.0, 1.0], exactly 16000 samples
_, feat = irdtv_optimized(audio, [1, 2, 4, 8, 16, 32], 64, 5, 24, False)
X = np.expand_dims(feat, axis=0)  # Shape: (1, 50, 6)

probs = sess.run(probs_tensor, feed_dict={feat_ph: X})[0]
print(f"P(Marvin):           {probs[0]*100:.2f}%")
print(f"P(Unknown Speech):   {probs[1]*100:.2f}%")
print(f"P(Background Noise): {probs[2]*100:.2f}%")

sess.close()
```
