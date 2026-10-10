# ESP32-S3 Deployment Guide: DS-CNN with iRDT for Keyword Spotting

This guide walks you through deploying the trained **DS-CNN + iRDT** model onto an **ESP32-S3** microcontroller for ultra-low-power, real-time keyword spotting (`"marvin"`).

---

## 1. Hardware Setup & Pinout

### Components Needed:
1. **ESP32-S3 Development Board** (e.g. ESP32-S3-DevKitC-1 with dual-core Xtensa @ 240 MHz, 512 KB SRAM).
2. **I2S Digital Microphone** (INMP441, SPH0645, or MSM261S).

### Wiring Connection (ESP32-S3 to INMP441):
| INMP441 Pin | ESP32-S3 Pin | Function / Description |
| :--- | :---: | :--- |
| **VDD** | **3.3V** | Power supply (3.3V) |
| **GND** | **GND** | Ground |
| **SD** | **GPIO 6** | Serial Data (I2S `DIN`) |
| **SCK / BCLK** | **GPIO 4** | Bit Clock (I2S `BCLK`) |
| **WS / LRCLK** | **GPIO 5** | Word Select (I2S `WS`) |
| **L/R** | **GND** | Left channel select |

---

## 2. Resource Footprint on ESP32-S3

The ESP32-S3 has **512 KB of internal SRAM** and typically **4 MB to 16 MB of Flash**.

| Resource | DS-CNN-15k INT8 | DS-CNN-32k INT8 | Available on ESP32-S3 | Budget Usage |
| :--- | :---: | :---: | :---: | :---: |
| **Model Flash (`.rodata`)** | **34.8 KB** | **58.0 KB** | 4,096 - 16,384 KB | **< 1.0%** |
| **Audio Ring Buffer (RAM)** | **32.0 KB** *(16k samples)* | **32.0 KB** | 512 KB | **6.2%** |
| **TFLM Tensor Arena (RAM)** | **25.0 KB** | **35.0 KB** | 512 KB | **4.8%** |
| **Total RAM Required** | **~57 KB** | **~67 KB** | 512 KB | **~11.5%** |
| **Free SRAM Remaining** | **> 440 KB** | **> 430 KB** | - | **Plenty for Wi-Fi stack & FreeRTOS** |

---

## 3. Step-by-Step Deployment Steps

### Step 1: Copy Deployment Files to your Project
The following pre-built files are located in `save/dscnn_irdt/`:
1. [`dscnn_marvin_model_data.h`](file:///c:/TENet-kws-irdt/TENet-kws-irdt/save/dscnn_irdt/dscnn_marvin_model_data.h) — C array of the INT8 model (`g_dscnn_model_data[]`).
2. [`irdt_feature_extractor.h`](file:///c:/TENet-kws-irdt/TENet-kws-irdt/save/dscnn_irdt/irdt_feature_extractor.h) — Pure C integer implementation of iRDT with log transform and quantization.
3. [`esp32s3_kws_main.cpp`](file:///c:/TENet-kws-irdt/TENet-kws-irdt/save/dscnn_irdt/esp32s3_kws_main.cpp) — Complete firmware sketch.

### Step 2: Install Libraries in Arduino IDE or ESP-IDF

#### Option A: Using Arduino IDE
1. Install **ESP32 Board Support** via Boards Manager (`esp32` by Espressif, version 2.0.14+ or 3.0+).
2. Install library **`EloquentTinyML`** or **`TensorFlowLite_ESP32`** via Library Manager.
3. Select board: **ESP32S3 Dev Module**.
4. Configure Tools:
   - **Flash Size**: 4MB or 8MB
   - **CPU Frequency**: 240 MHz (WiFi/BT)
   - **PSRAM**: Disabled or OPI PSRAM (not strictly needed, as model fits in internal SRAM)

#### Option B: Using ESP-IDF (Recommended for Production)
1. Add `esp-tflite-micro` component to your `main/idf_component.yml`:
   ```yaml
   dependencies:
     espressif/esp-tflite-micro: "^1.3.1"
   ```
2. Enable ESP-NN acceleration:
   - In `menuconfig` $\rightarrow$ `Component config` $\rightarrow$ `ESP-NN` $\rightarrow$ Select `Optimized Xtensa PIE implementation`.
   - This activates the hardware SIMD vector instructions on the ESP32-S3 Xtensa LX7 core, boosting inference speed by 2.5x to 3x.

---

## 4. Expected Real-Time Timing on ESP32-S3 (@ 240 MHz)

| Pipeline Stage | Algorithm | Duration on ESP32-S3 | Details |
| :--- | :--- | :---: | :--- |
| **Audio Ingest** | I2S DMA Ring Buffer | Background (DMA) | 0% CPU overhead (hardware DMA) |
| **iRDT Feature Extraction** | Integer Laplacian + Log2 | **~3.2 ms** | Pure integer arithmetic (no FFT/filterbanks) |
| **Neural Network Inference** | DS-CNN-15k INT8 (ESP-NN) | **~18.5 ms** | Hardware vector accelerated via Xtensa PIE |
| **Total End-to-End Latency** | Full Wake Word Pipeline | **~21.7 ms** | Real-time factor (RTF) $\approx 0.022$ |
| **Inference Hop** | Every 200 ms | - | Sleeps 178 ms out of every 200 ms |
| **Idle CPU Utilization** | Wake-word Listening | **~10.8%** | Microcontroller remains mostly idle |

---

## 5. Connecting Wake-Up to Cloud ASR

When `p_marvin >= 0.50f` triggers:
1. The ESP32-S3 activates Wi-Fi.
2. Streams subsequent PCM audio chunks over UDP or WebSocket to the remote cloud server (`edge_kws_to_cloud_asr.py`).
3. Latency from keyword ending to cloud reception is **< 15 ms** over local network.
