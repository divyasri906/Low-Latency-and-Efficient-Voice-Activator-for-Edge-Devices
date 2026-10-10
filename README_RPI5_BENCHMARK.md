# Raspberry Pi 5 On-Device KWS Benchmark Results

- **Target Board:** Raspberry Pi 5 Model B (Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.40 GHz)
- **Host / Interface:** SSH (`10.255.2.225`) | Linux 6.18.34 aarch64 (Debian 12)
- **Software Runtime:** Google LiteRT 2.2.0 + XNNPACK NEON Backend
- **Audio Input:** 1.0-Second Audio (16,000 samples @ 16 kHz, 16-bit PCM)

---

## 1. Latency & Execution Timings (Single-Threaded, 1 Core @ 2.40 GHz)

| Metric | TENet6-Narrow + Log-iRDT | DS-CNN + iRDT | DS-CNN + MFCC |
| :--- | :---: | :---: | :---: |
| **Feature Extraction Time (Mean)** | 0.0378 ms (37.8 us) | 0.0367 ms (36.7 us) | 0.7602 ms (760.2 us) |
| **Feature Extraction Time (P95)** | 0.0397 ms | 0.0398 ms | 0.7681 ms |
| **NN Inference Time (Mean)** | 0.0272 ms (27.2 us) | 0.2822 ms (282.2 us) | 0.4220 ms (422.0 us) |
| **NN Inference Time (P95)** | 0.0301 ms | 0.2925 ms | 0.4311 ms |
| **Total End-to-End Latency** | 0.0650 ms (65.0 us) | 0.3189 ms (318.9 us) | 1.1823 ms (1182.3 us) |
| **End-to-End Latency (P95)** | 0.0699 ms | 0.3298 ms | 1.1974 ms |
| **Real-Time Factor (RTF)** | 0.000065 | 0.000319 | 0.001182 |
| **Real-Time Throughput Margin** | 15,383.22x | 3,135.47x | 845.83x |

---

## 2. Latency & Execution Timings (Multi-Threaded, 4 Cores with XNNPACK)

| Metric | TENet6-Narrow + Log-iRDT | DS-CNN + iRDT | DS-CNN + MFCC |
| :--- | :---: | :---: | :---: |
| **Feature Extraction Time (Mean)** | 0.0371 ms | 0.0401 ms | 0.7618 ms |
| **NN Inference Time (Mean)** | 0.0436 ms | 0.1494 ms | 0.1402 ms |
| **Total End-to-End Latency** | 0.0807 ms (80.7 us) | 0.1895 ms (189.5 us) | 0.9020 ms (902.0 us) |
| **Real-Time Factor (RTF)** | 0.000081 | 0.000189 | 0.000902 |
| **Real-Time Throughput Margin** | 12,391.55x | 5,277.81x | 1,108.60x |

---

## 3. CPU Utilization & Duty Cycle Numbers

| Metric | TENet6-Narrow + Log-iRDT | DS-CNN + iRDT | DS-CNN + MFCC |
| :--- | :---: | :---: | :---: |
| **Live Continuous 1.0s Audio Load (1 Core)** | 0.0065% | 0.0319% | 0.1182% |
| **Live Continuous 1.0s Audio Load (4 Cores)**| 0.0016% | 0.0080% | 0.0295% |
| **Live Responsive 200ms Hop Stream Load** | 0.0325% | 0.1595% | 0.5910% |
| **Live System CPU Idle / Sleep Time** | 99.994% | 99.968% | 99.882% |
| **Max-Throughput Infinite Loop Load (1 Core)**| 151.6% | 93.8% | 93.0% |
| **Max-Throughput Infinite Loop Load (4 Cores)**| 122.3% | 261.9% | 132.8% |
| **Picovoice Porcupine Benchmark Baseline** | 0.6000% | 0.6000% | 0.6000% |
| **CPU Efficiency Advantage vs. Picovoice** | 92.3x less CPU | 18.8x less CPU | 5.1x less CPU |

---

## 4. Memory Footprint (RAM & Storage)

| Metric | TENet6-Narrow + Log-iRDT | DS-CNN + iRDT | DS-CNN + MFCC |
| :--- | :---: | :---: | :---: |
| **Model File Size** | 48.78 KB | 33.97 KB | 33.66 KB |
| **Process Resident Set Size (Peak RSS)** | 44.10 MB | 44.10 MB | 42.56 MB |
| **Current Process RSS** | 45.14 MB | 45.07 MB | 43.17 MB |
| **Total 8 GB System RAM Consumed** | 0.54% | 0.54% | 0.52% |
| **System Free RAM Remaining** | 7.4 GB | 7.4 GB | 7.4 GB |

---

## 5. Classification Accuracy (Speech Commands v2 Test Set, N=400)

| Metric | TENet6-Narrow + Log-iRDT | DS-CNN + iRDT | DS-CNN + MFCC |
| :--- | :---: | :---: | :---: |
| **3-Class Test Accuracy** | 97.25% (389/400) | 95.25% (381/400) | 86.50% (346/400) |
| **Marvin Keyword Recall** | 99.00% (99/100) | 95.00% (95/100) | 84.00% (84/100) |
| **Marvin Keyword Precision** | 90.83% | 88.78% | 77.78% |
| **Background Noise False Alarms** | 0.00% (0/100) | 0.00% (0/100) | 10.00% (10/100) |
| **Hardware Clip 1 P(Marvin)** | 99.61% | 99.61% | 99.61% |
| **Hardware Clip 2 P(Unknown)** | 99.61% | 99.61% | 99.61% |
| **Hardware Clip 3 P(Silence)** | 99.61% | 99.61% | 99.61% |

---

## 6. Real-World User Recordings (`clipped_1sec`, N=27)

| Metric | TENet6-Narrow + Log-iRDT |
| :--- | :---: |
| **Total User Audio Files** | 27 |
| **Static 1.0s Crop Detections** | 20 / 27 (74.07%) |
| **Sliding Window Stream Detections** | 24 / 27 (88.89%) |
| **Median Confidence** | 99.6% |
