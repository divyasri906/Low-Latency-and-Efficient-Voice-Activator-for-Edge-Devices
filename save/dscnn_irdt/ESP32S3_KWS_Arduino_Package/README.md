# ESP32-S3 Keyword Spotting (KWS): DS-CNN + iRDT

This package contains two ready-to-flash Arduino sketches for your ESP32-S3:

1. **ESP32S3_KWS_Benchmark/**:
   - Self-contained on-chip benchmark (no external microphone needed).
   - Measures exact microsecond latency for iRDT feature extraction, neural network inference, memory footprint, and verifies prediction accuracy on embedded test samples.
   
2. **ESP32S3_KWS_Live_Mic/**:
   - Production firmware with live I2S microphone (INMP441 / SPH0645) audio streaming.
   - Listens for the wake word \'MARVIN\' and triggers alert upon detection.

---

## Quick Start (Arduino IDE)

### 1. Board & Library Setup
1. In Arduino IDE, go to **Tools -> Board -> Boards Manager**, search for **esp32** by Espressif, and install (version 2.0.14+ or 3.0+).
2. Go to **Sketch -> Include Library -> Manage Libraries**, search for **TensorFlowLite_ESP32** (or **EloquentTinyML**) and install it.

### 2. Open and Upload
1. To run the benchmark:
   - Double-click ESP32S3_KWS_Benchmark/ESP32S3_KWS_Benchmark.ino.
   - Select Board: **ESP32S3 Dev Module**.
   - Select your COM Port.
   - Click **Upload**.
   - Open **Serial Monitor** at **115200 baud** to see real-time benchmark results!

2. To run live microphone detection:
   - Wire your INMP441 microphone (BCLK -> GPIO 4, WS -> GPIO 5, DATA -> GPIO 6, VDD -> 3.3V, GND -> GND).
   - Double-click ESP32S3_KWS_Live_Mic/ESP32S3_KWS_Live_Mic.ino.
   - Click **Upload** and speak \'Marvin\'!
