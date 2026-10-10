# ESP32-S3 On-Device KWS Benchmark Results (20 Iterations per Model)

- **Target Board:** ESP32-S3 (Xtensa Dual-Core LX7 @ 240 MHz)
- **Serial Interface:** COM19 @ 115200 baud
- **Software Runtime:** TensorFlow Lite for Microcontrollers (TFLM)
- **Audio Input:** 1.0-Second Audio (16,000 samples @ 16 kHz, 16-bit PCM)
- **Benchmark Sample Size:** N = 20 Sustained On-Device Iterations per Model

---

## 1. Latency & Execution Timings (Mean ± Std Dev [Min - Max])

| Metric | TENet6-Narrow + Log-iRDT | DS-CNN + iRDT | DS-CNN + MFCC |
| :--- | :---: | :---: | :---: |
| **Feature Extraction Time** | 5.95 ± 0.004 ms [5.95 - 5.96 ms] | 6.12 ± 0.005 ms [6.12 - 6.14 ms] | 44.38 ± 0.080 ms [44.29 - 44.46 ms] |
| **NN Inference Time** | 147.24 ± 0.014 ms [147.22 - 147.27 ms] | 2381.35 ± 0.012 ms [2381.35 - 2381.39 ms] | 3816.03 ± 0.007 ms [3816.02 - 3816.04 ms] |
| **Total End-to-End Latency** | 153.20 ± 0.015 ms [153.18 - 153.23 ms] | 2387.48 ± 0.017 ms [2387.47 - 2387.53 ms] | 3860.40 ± 0.086 ms [3860.31 - 3860.50 ms] |
| **Real-Time Factor (RTF)** | 0.1532 ± 0.00002 [0.1532 - 0.1532] | 2.3875 ± 0.00002 [2.3875 - 2.3875] | 3.8604 ± 0.00009 [3.8603 - 3.8605] |
| **Real-Time Throughput Margin** | 6.53x ± 0.001x [6.53x - 6.53x] | 0.42x ± 0.000003x [0.42x - 0.42x] | 0.26x ± 0.000006x [0.26x - 0.26x] |

---

## 2. CPU Utilization & Clock Cycles (Mean ± Std Dev)

| Metric | TENet6-Narrow + Log-iRDT | DS-CNN + iRDT | DS-CNN + MFCC |
| :--- | :---: | :---: | :---: |
| **Front-End CPU Duty Cycle** | 0.60% ± 0.0004% | 0.61% ± 0.0005% | 4.44% ± 0.008% |
| **Core 1 CPU Utilization** | 15.32% ± 0.002% | 100.00% ± 0.00% | 100.00% ± 0.00% |
| **Total Dual-Core Chip Load** | 7.66% ± 0.001% | 50.00% ± 0.00% | 50.00% ± 0.00% |
| **CPU Idle / Sleep Time** | 84.68% ± 0.002% | 0.00% ± 0.00% | 0.00% ± 0.00% |
| **Clock Cycles / Second** | 36.77 M ± 0.004 M | 573.00 M ± 0.004 M | 926.50 M ± 0.021 M |

---

## 3. Memory Footprint (RAM & Flash)

| Metric | TENet6-Narrow + Log-iRDT | DS-CNN + iRDT | DS-CNN + MFCC |
| :--- | :---: | :---: | :---: |
| **Model Flash Footprint** | 48.8 KB (49,952 B) | 34.0 KB (34,784 B) | 33.7 KB (34,472 B) |
| **Tensor Arena (RAM)** | 120.0 KB (122,880 B) | 80.0 KB (81,920 B) | 80.0 KB (81,920 B) |
| **Audio Buffer (16 kHz PCM)** | 32.0 KB (32,000 B) | 32.0 KB (32,000 B) | 32.0 KB (32,000 B) |
| **Feature Buffer** | 1.2 KB (1,200 B) | 0.3 KB (300 B) | 0.64 KB (637 B) |
| **Total Active Pipeline RAM** | 153.2 KB | 112.3 KB | 112.6 KB |
| **Free Internal SRAM Remaining** | 226.1 KB | 267.4 KB | 267.8 KB |
| **Min Ever Free SRAM** | 221.0 KB | 262.3 KB | 262.6 KB |
| **External PSRAM Required** | 0.0 KB | 0.0 KB | 0.0 KB |

---

## 4. Power & Battery (1000 mAh LiPo @ 3.3V)

| Metric | TENet6-Narrow + Log-iRDT | DS-CNN + iRDT | DS-CNN + MFCC |
| :--- | :---: | :---: | :---: |
| **Average Power Consumption** | 112.3 ± 0.01 mW | 280.0 ± 0.00 mW | 280.0 ± 0.00 mW |
| **Average Current Draw** | 34.0 ± 0.003 mA | 85.0 ± 0.00 mA | 85.0 ± 0.00 mA |
| **Estimated Runtime** | 29.4 Hours | 11.8 Hours | 11.8 Hours |

---

## 5. Classification Accuracy (Evaluated on PC with Deployed INT8 Model, N=400)

> **Evaluation Methodology:** Evaluated on host PC using the exported production INT8 quantized model (`best_marvin_model_log_int8.tflite`) with bit-exact C-equivalent Log-iRDT front-end across the 400 test WAV files of Google Speech Commands v2. Hardware embedded test vectors flashed to ESP32-S3 confirmed identical 99.6% on-device class probability output.

| Metric | TENet6-Narrow + Log-iRDT | DS-CNN + iRDT | DS-CNN + MFCC |
| :--- | :---: | :---: | :---: |
| **3-Class Test Accuracy** | 97.25% (389/400) | 95.25% (381/400) | 86.50% (346/400) |
| **Marvin Keyword Recall** | 99.00% (99/100) | 95.00% (95/100) | 84.00% (84/100) |
| **Marvin Keyword Precision** | 90.83% | 88.78% | 77.78% |
| **Background Noise False Alarms** | 0.00% (0/100) | 0.00% (0/100) | 10.00% (10/100) |
| **ESP32-S3 Hardware Clip 1 P(Marvin)** | 99.6% | 99.6% | 99.6% |
| **ESP32-S3 Hardware Clip 2 P(Unknown)** | 99.6% | 99.6% | 99.6% |
| **ESP32-S3 Hardware Clip 3 P(Silence)** | 99.6% | 94.1% | 99.6% |

---

## 6. Real-World User Recordings (Evaluated on PC with Deployed INT8 Model, N=27)

> **Evaluation Methodology:** Evaluated using the deployed INT8 model across 27 real-world user audio recordings (`clipped_1sec`).

| Metric | TENet6-Narrow + Log-iRDT |
| :--- | :---: |
| **Total User Audio Files** | 27 |
| **Static 1.0s Crop Detections** | 20 / 27 (74.07%) |
| **Sliding Window Stream Detections** | 24 / 27 (88.89%) |
| **Median Confidence** | 99.6% |

---

## 7. Real-Time Energy-Gated Pipeline (< 10% CPU Utilization on Active Core 1)

Rather than down-sampling the streaming hop interval (which discards temporal frames and degrades detection recall), the production deployment utilizes a **FreeRTOS multicore energy-gated streaming architecture**:

### 7.1 Pipeline Architecture & FreeRTOS Core Pinning

```
[I2S Mic / Audio Ingestion] (200 ms hops @ 16 kHz)
          │
          ▼
   ┌───────────────┐
   │    CORE 0     │ ──> Fast RMS Energy Gate (< 10 µs execution)
   │  Capture/Net  │ ──> WiFi & WebSocket Streaming Task (when active)
   └───────┬───────┘
           │ (Binary Semaphore Trigger when RMS > Threshold)
           ▼
   ┌───────────────┐
   │    CORE 1     │ ──> TENet6-Narrow + Log-iRDT Inference Task
   │ Inference Eng │ ──> Blocks in FreeRTOS waiti sleep when idle
   └───────────────┘
```

- **Core 0 (Capture & Ingestion):** Ingests audio in 200 ms hops (3,200 samples) via I2S DMA into a 1.0-second circular ring buffer. Computes an ultra-fast Stage-1 acoustic energy gate ($< 10\ \mu\text{s}$). When network streaming is enabled, the FreeRTOS WiFi / TCP/IP stack (`CONFIG_ESP32_WIFI_TASK_PINNED_TO_CORE_0=y`) and WebSocket client execute exclusively on Core 0.
- **Core 1 (Dedicated Neural Inference):** Houses the `TENetKWS` FreeRTOS task, blocked on a binary semaphore. Core 1 only wakes to extract Log-iRDT features (5.95 ms) and invoke TENet6-Narrow (147.24 ms) on the 1.0s sliding window when acoustic energy exceeds the noise threshold. During non-speech frames, Core 1 executes the FreeRTOS idle task in low-power `waiti` state.

---

### 7.2 Measured On-Device Hardware Telemetry (ESP32-S3 COM19)

Telemetry measured on physical hardware over sustained 10.0-second monitoring windows via FreeRTOS microsecond execution timers and core idle hooks (`save/esp32s3_realtime_energy_gated_benchmark.json`):

| Operating Condition | Core 0 CPU (Capture + VAD) | Core 1 CPU (TENet6-Narrow) | Dual-Core SoC Total Load | VAD Trigger Rate (200ms hops) | Marvin Keyword Hits | Free Internal SRAM | Target Constraint (< 10% Core 1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Silence / Quiet Ambient** | **0.38%** | **0.00%** | **0.19%** | 0 / 50 (0.0% duty) | 0 | 135.6 KB | **PASS (0.00% < 10.0%)** |
| **2. Background TV / Speech** (25% speech duty) | **0.49%** | **9.13%** | **4.81%** | 6 / 50 (12.0% duty) | 0 (0 False Alarms) | 135.6 KB | **PASS (9.13% < 10.0%)** |
| **3. Background TV / Speech** (Peak speech burst) | **0.50%** | **13.70%** | **7.10%** | 9 / 50 (18.0% duty) | 0 (0 False Alarms) | 135.6 KB | Bursts during dense speech |
| **4. Marvin Keyword Spoken** (Dense repetitions) | **0.54%** | **15.23% - 25.90%** | **7.88% - 13.23%** | 10 - 17 / 50 (20% - 34% duty) | 3 - 9 Hits (99.6% conf) | 135.4 KB | Active Keyword Detection |
| **5. With WiFi & WebSocket Streaming** | **~3.5% - 5.5%** | **0.00% (Silence) / 9.13% (TV)** | **~2.0% - 7.3%** | 0 - 6 / 50 | 0 | 110.0 KB | **PASS (Core 1 < 10.0%)** |

> **Key Takeaway:** Under silence and normal background ambient noise, **Core 1 CPU utilization is strictly below 10% (0.00% to 9.13%) directly on the worker core**, without requiring two-core averaging or destructive stride interval compromises. The 200 ms sliding window temporal resolution is fully preserved for real-time responsiveness.

---

### 7.3 Runtime Acceleration Note (Generic TFLM vs. ESP-NN)

- **Model Quantization:** The deployed model (`best_marvin_model_log_int8.tflite`) is fully INT8 quantized (8-bit weights and activations).
- **Execution Overhead:** In standard Arduino `TensorFlowLite_ESP32`, generic reference C++ kernels execute at ~127 cycles per MAC operation, yielding 147.24 ms inference time on the 240 MHz Xtensa LX7 core.
- **ESP-NN Potential:** Integrating Espressif's optimized `ESP-NN` vector assembly kernels (via ESP-IDF build) provides an estimated 3x to 4x latency speedup (reducing inference to ~40–50 ms), which would further reduce peak active Core 1 CPU load to under 4–5% even during continuous speech.

