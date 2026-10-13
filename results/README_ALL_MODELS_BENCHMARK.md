# KWS Trained Models & Weights Benchmark Registry

Summary of all trained model variants, parameter scalings, seed initializations, and hardware-quantized exports for Keyword Spotting (KWS).

---

## 1. Channel Width-Scaled Models (`save/tenet_scale_w5/`)

*Evaluated on the 400-sample test set (100 Marvins, 200 Unknown Speech words, 100 Background Noise segments).*

| Model Variant | Width Multiplier | Trainable Params | Total Params | Model Size (Raw / Chkpt) | Test Accuracy | Marvin Recall | Marvin Precision | False Positives | CPU Latency (x86 P50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TENet6-Narrow-1.00x** (Baseline) | 1.00x (16 ch) | 14,384 | 15,856 | 56.19 KB / 61.94 KB | 97.25% | 99.00% (99/100) | 90.83% | 10 / 300 | 1.29 ms |
| **TENet6-Narrow-1.25x** | 1.25x (20 ch) | 21,100 | 22,940 | 82.42 KB / 89.61 KB | 96.75% | 98.00% (98/100) | 89.91% | 11 / 300 | 1.30 ms |
| **TENet6-Narrow-1.50x** | 1.50x (24 ch) | 29,064 | 31,272 | 113.53 KB / 122.16 KB | 96.75% | 96.00% (96/100) | 91.43% | 9 / 300 | 1.30 ms |
| **TENet6-Narrow-1.75x** | 1.75x (28 ch) | 38,276 | 40,852 | 149.52 KB / 159.58 KB | 97.25% | 99.00% (99/100) | 90.83% | 10 / 300 | 1.32 ms |

---

## 2. Multi-Seed Weights & Ensemble Models (`save/TENet6Narrow_MultiSeed/`)

*Trained on 3,000 steps with independent random weight initializations.*

| Model Weights / Checkpoint | Random Seed | Total Params | Best Val Accuracy | Test Accuracy | Marvin Recall | Marvin Precision | False Alarms on Noise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TENet6-Narrow (Seed 42)** | 42 | 15,120 | 96.50% | 97.25% (389/400) | 98.00% (98/100) | 91.59% | 0.00% (0/100) |
| **TENet6-Narrow (Seed 101)** | 101 | 15,120 | 97.00% | 96.75% (387/400) | 99.00% (99/100) | 89.19% | 0.00% (0/100) |
| **TENet6-Narrow (Seed 2024)** | 2024 | 15,120 | 97.00% | 96.25% (385/400) | 97.00% (97/100) | 88.99% | 0.00% (0/100) |
| **3-Seed Soft-Voting Ensemble** | 42 + 101 + 2024 | 45,360 | 97.50% | 97.50% (390/400) | 99.00% (99/100) | 91.67% | 0.00% (0/100) |

---

## 3. Multi-Scale Kernel Fusion & Deep Variants (`save/MTENet_Fusion/` & `save/tenet12/`)

| Model Variant | Description | Trainable Params | Total MACs | Test Accuracy | Checkpoint Size |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MTENet6-Narrow (Fused)** | Multi-branch kernels [3, 5, 7, 9] fused into base kernel | 15,120 | 288,000 | 96.75% | 61.94 KB |
| **TENet12 (Full Deep Model)** | 12 layers, 32 channels (base non-narrow architecture) | 100,512 | 1,920,000 | 96.80% | ~402.0 KB |

---

## 4. Matched DS-CNN Comparative Models (`save/dscnn_irdt/` & `save/dscnn_mfcc/`)

| Model | Acoustic Front-End | Parameters | Arithmetic MACs | Test Accuracy | Marvin Recall | ESP32-S3 Latency | RPi 5 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DS-CNN + iRDT** | Linear iRDT (50 x 6) | 15,121 | 6,445,000 | 95.25% | 95.00% | 2387.49 ms | 0.3189 ms |
| **DS-CNN + MFCC** | 13-MFCC (49 x 13) | 15,121 | 6,445,000 | 86.50% | 84.00% | 3860.35 ms | 1.1823 ms |

---

## 5. Hardware-Quantized Deployment Champion (`best_marvin_model_log_int8.tflite`)

| Parameter / Metric | Quantized On-Device Model Value |
| :--- | :---: |
| **Model Filename** | `best_marvin_model_log_int8.tflite` |
| **Quantization Scheme** | Full Integer (INT8 Weights, Dynamic Range Ops) |
| **Total Model Parameters** | 15,120 (Trainable Conv Backbone Parameters as per Slide 5) |
| **Model Size (.tflite binary)** | 48.78 KB (49,952 Bytes) |
| **Multiply-Accumulate Operations (MACs)** | 288,000 (22.4x fewer than DS-CNN) |
| **Speech Commands Test Accuracy** | 96.50% (as per Slide 4; 3-seed mean: 96.75% ± 0.50%) |
| **Speech Commands Marvin Recall** | 98.00% (as per Slide 4; 98 / 100 on Marvin) |
| **Speech Commands Noise False Alarms** | 0.00% (0 / 100) |
| **ESP32-S3 Total E2E Latency** | 153.20 ms on-chip (147.24 ms NN + 5.95 ms iRDT as per Slide 4) |
| **ESP32-S3 Total Active RAM** | 205.42 KB (all subsystems active as per Slide 4, < 256 KB limit) |
| **ESP32-S3 CPU Utilization** | 9.28% (as per Slide 4; true ambient silence drops to 0.19%) |
| **Raspberry Pi 5 Total E2E Latency** | 0.0650 ms (65.0 us, RTF = 0.000065, 15,383x real-time) |
| **Raspberry Pi 5 CPU Utilization** | 0.0065% |
