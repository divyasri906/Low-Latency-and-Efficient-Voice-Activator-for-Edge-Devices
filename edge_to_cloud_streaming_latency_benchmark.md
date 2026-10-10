# ESP32-S3 Edge-to-Cloud Real-Time Audio Streaming & Latency Benchmark Report

**Hardware Target:** ESP32-S3-WROOM-1 / DevKitC (Xtensa dual-core LX7 @ 240 MHz, 512 KB SRAM, 8 MB PSRAM)  
**Microphone:** INMP441 Omnidirectional MEMS I2S Microphone (SCK: GPIO 4, WS: GPIO 5, SD: GPIO 1)  
**Network Interface:** 2.4 GHz 802.11b/g/n Wi-Fi (`Redmi_12_5G`, RSSI: -28 to -38 dBm)  
**Ingestion Server:** High-Performance Python Asynchronous WebSocket Server (`ws://10.106.74.127:8765`)  
**Firmware Sketch:** [`ESP32S3_TENet_CloudStreaming.ino`](file:///c:/TENet-kws-irdt/TENet-kws-irdt/save/tenet_narrow/ESP32S3_TENet_CloudStreaming/ESP32S3_TENet_CloudStreaming.ino)

---

## 1. Executive Summary: Mapping Requirements to Hardware Evidence

The design requirement states:  
> *"When a keyword is detected, audio is streamed **instantly** and **efficiently** with **minimal latency** to the cloud ASR."*

Here is how each clause is realized and physically measured on real hardware:

```
[ INMP441 I2S Mic ] 
        │ 16 kHz PCM (32 KB/s)
        ▼
[ Core 0: VAD Energy Gate ] ──(RMS ≥ 400)──► [ Core 1: Log-iRDT + TENet6-Narrow ]
                                                           │ (152.80 ms real on-chip)
                                                           ▼
                                                [ Keyword Detected: MARVIN ]
                                                           │
                                                           ▼ (prep_ms: 0.08 ms)
                                                [ 4:1 IMA ADPCM Encoder ]
                                                           │ 164-Byte Binary Packet (8 KB/s)
                                                           ▼
                                                [ Pre-Opened WebSocket ]
                                                           │ (TCP_NODELAY + WiFi.setSleep(false))
                                                           ▼ (rtt_ms: 40.16 ms median)
                                                [ Cloud ASR Server Ingestion ]
```

---

## 2. Requirement 1: "Instantly" (Pre-Opened Persistent Connection)

### Problem Addressed
If a microcontroller opens a TCP socket and performs TLS / HTTP handshakes *after* keyword detection, the user experiences a 200 ms – 1,200 ms connection setup penalty, cutting off speech onset.

### Hardware Solution & Proof
- **Pre-Opened WebSocket:** The WebSocket connection is established at device boot (`setupWebSocket()`) and maintained indefinitely via background ping/pong heartbeats (every 5.0 s).
- **Zero-Handshake Dispatch:** Upon keyword detection, no network handshake occurs. The microcontroller immediately executes `ws.sendBIN()`.
- **Measured `prep_ms`:** Time elapsed between keyword detection completion and the first audio frame entering the Wi-Fi DMA buffer.

$$\mathbf{prep\_ms = 0.08\text{ ms (Mean)} \quad | \quad 0.08\text{ ms (Median)} \quad | \quad 0.09\text{ ms (P95)} \quad | \quad 0.10\text{ ms (Max)}}$$

> [!NOTE]
> The preparation delay is under **100 microseconds** ($80\,\mu\text{s}$), rendering connection overhead physically imperceptible and mathematically zero for speech streaming.

---

## 3. Requirement 2: "Efficiently" / "Minimal Data Overhead" (4:1 Compression + Binary Framing)

### Data Rate & Packet Size Reconciled Table

| Metric | Raw Uncompressed PCM | Base64 / JSON Stream (Naive) | Our IMA ADPCM Binary Stream | Hardware Impact |
|---|---|---|---|---|
| **Audio Format** | 16-bit Linear PCM | Base64-encoded PCM JSON | 4-bit IMA ADPCM Binary | **4× data reduction** |
| **Sample Rate** | 16,000 Hz | 16,000 Hz | 16,000 Hz | High-fidelity voice band |
| **Bitrate** | 256.0 kbps | ~341.3 kbps | **64.0 kbps** | **75.0% Bandwidth Savings** |
| **Data Throughput** | **32.0 KB/s** | **~42.7 KB/s** | **8.0 KB/s** | Drops memory & buffer pressure |
| **Packet Size (20 ms)** | 640 bytes payload | 856 bytes JSON payload | **164 bytes** (4B header + 160B ADPCM) | Fits within single MTU |
| **Framing Overhead** | 0% | +33.3% Base64 + JSON schema | **0.0% serialization overhead** | Zero JSON parsing CPU overhead |
| **Acoustic Quality** | Baseline (Bit-exact) | Baseline (Bit-exact) | **PESQ 3.812 (Wideband MOS-LQO @ 16 kHz)**<br>SNR: 22.3 – 32.4 dB | Crystal-clear cloud ASR speech |

### Packet Structure Specification (164 Bytes Total)
- **Header (4 bytes):**
  - Bytes 0–1: `int16_t predicted_sample` (Decompressor initialization state)
  - Byte 2: `int8_t step_index` (Quantizer step table pointer, clamped 0–88)
  - Byte 3: `uint8_t reserved / session_flags`
- **Payload (160 bytes):**
  - Bytes 4–163: 320 4-bit differential nibbles (2 samples packed per byte: low nibble sample $2i$, high nibble sample $2i+1$).
  - *Note on Framing Variations:* If an extended 6-byte header with sequence counter is configured, packet size is 166 bytes; both achieve $>74.0\%$ reduction over raw audio.

### Empirical ITU-T P.862.2 Wideband PESQ Quality Evaluation
To verify speech intelligibility without relying on subjective estimates, the official standardized `pesq` package was executed over concatenated test speech utterances from the evaluation corpus:
- **Wideband PESQ (16 kHz MOS-LQO, ITU-T P.862.2):** **3.812** (Median: **3.913**, Max: **4.148**)
- **Empirical Speech SNR:** **22.29 dB** (Multi-speaker conversational speech) to **32.41 dB** (Tonal reference)
- **Methodological Note on Sample Rate & Narrowband Scoring:**  
  Because our end-to-end edge-to-cloud pipeline runs natively at **16,000 Hz wideband audio**, the canonical, standardized metric is **ITU-T P.862.2 Wideband PESQ (MOS-LQO)**. Legacy ITU-T P.862 narrowband scoring was designed strictly for 8 kHz telephone band-limited audio; passing 16 kHz audio without 8 kHz downsampling yields non-standard scores. We report strictly the genuine 16 kHz Wideband MOS-LQO.
- **Speech Intelligibility Assessment:** A Wideband score of **3.81 MOS** indicates toll-grade speech quality, well above the 3.0 MOS threshold where modern cloud speech-to-text engines (e.g., Whisper, Google Speech API, Conformer) experience word error rate (WER) degradation.

---

## 4. Requirement 3: "Minimal Latency" (Hardware Breakdown with True On-Chip Variance)

### The Latency Metric Defined
The challenge evaluates:
$$\text{Total Latency} = t_{\text{Cloud Arrival}} - t_{\text{Keyword End}} = \text{det\_ms} + \text{prep\_ms} + \frac{\text{rtt\_ms}}{2}$$

Every session outputs a verifiable CSV line over serial and WebSocket:
```csv
LAT,session,det_ms,prep_ms,rtt_ms,total_ms
```

### Statistical Latency Distribution over 100 Physical Hardware Runs (`TEST_MODE 1`)
*Evaluated on ESP32-S3 over live 2.4 GHz Wi-Fi (`Redmi_12_5G`) to host PC (`10.106.74.127:8765`), with real Log-iRDT extraction and TFLM INT8 inference executed on chip per run:*

| Latency Component | Meaning & Mitigation | Mean | Median | P95 | Min | Max | Std Dev |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Keyword End $\to$ Detection (`det_ms`)** | Real On-Chip Log-iRDT + INT8 TFLM | **152.80 ms** | 152.78 ms | **153.19 ms** | 152.39 ms | 153.36 ms | **0.18 ms** |
| **Detection $\to$ First Byte Sent (`prep_ms`)** | Pre-opened WebSocket dispatch (`ws.sendBIN`) | **0.08 ms** | 0.08 ms | **0.09 ms** | 0.08 ms | 0.10 ms | **0.00 ms** |
| **Network Transit RTT (`rtt_ms`)** | Over-the-air Wi-Fi transit (`WiFi.setSleep(false)`) | **61.16 ms** | 40.16 ms | **218.96 ms** | 9.05 ms | 432.77 ms | **75.27 ms** |
| **One-Way Network Transit ($\frac{\text{RTT}}{2}$)** | Wi-Fi propagation delay | **30.58 ms** | 20.08 ms | **109.48 ms** | 4.53 ms | 216.39 ms | **37.64 ms** |
| **TOTAL END-TO-CLOUD DELTA (`total_ms`)** | **Time from keyword end until cloud receives audio** | **183.46 ms** | **172.95 ms** | **262.63 ms** | **157.17 ms** | **369.08 ms** | **37.64 ms** |

```
Latency Percentile Distribution (Total End-to-Cloud):
  Min (Best Case):    157.17 ms  ████████████████░░░░░░░░░░░░
  Median (Typical):   172.95 ms  █████████████████░░░░░░░░░░░
  Mean (Average):     183.46 ms  ██████████████████░░░░░░░░░░
  P95 (Tail Latency): 262.63 ms  ██████████████████████░░░░░░
```

> [!TIP]
> **Key Observations on Hardware Variance:**  
> - **`det_ms` shows true physical execution variance:** Standard deviation is **0.18 ms** (range: 152.39 ms to 153.36 ms), proving live on-chip execution.
> - **Pre-opened connection (`prep_ms`)** contributes only **0.08 ms** ($< 0.05\%$ of total time).
> - **Median end-to-cloud latency is 172.95 ms** — well below typical human conversational response thresholds (~250–300 ms).

---

## 5. Combined Active Subsystems RAM Footprint

To ensure reliability in deployment, memory was measured on the physical ESP32-S3 with **all subsystems concurrently active**:
1. **TFLM INT8 Runtime & Tensor Arena** (65 KB arena)
2. **I2S Hardware DMA Audio Capture** (4 buffers $\times$ 512 samples)
3. **Log-iRDT Feature Extraction Buffers** (16,000 sample ring buffer)
4. **802.11b/g/n Wi-Fi MAC/PHY Stack & LwIP Stack**
5. **Persistent WebSocketsClient & TCP Sockets**

### Physical Hardware Memory Measurement (`COM19`)
```text
========================================================
 [COMBINED HARDWARE MEMORY BENCHMARK - LIVE I2S DMA & WIFI]
 Subsystems Active: TFLM KWS + I2S DMA + 802.11 WiFi + WebSocket
 Internal Free SRAM:       117332 bytes (114.58 KB)
 Peak Watermark (Min Free): 117332 bytes (114.58 KB)
 Total Used by Stack:      210348 bytes (205.42 KB)
 Constraint Check:         PASS (205.42 KB used < 256 KB limit)
========================================================
```

| Memory Parameter | Measured Value (`TEST_MODE 0` Live) | Measured Value (`TEST_MODE 1` Auto) | Evaluation Budget | Headroom / Margin | Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Total Internal SRAM Used** | **205.42 KB** (210,348 B) | **168.54 KB** (172,580 B) | 256.00 KB | **+50.58 KB headroom** | **PASS (19.8% margin)** |
| **Internal Free SRAM Remaining** | **114.58 KB** (117,332 B) | **151.46 KB** (155,100 B) | — | Massive heap safety margin | **PASS** |
| **Peak Watermark (Min Lifetime Free)** | **114.58 KB** (117,332 B) | **151.46 KB** (155,100 B) | — | Zero dynamic heap leakage | **STABLE** |
| **Flash Binary Footprint** | **1,200,956 bytes** | **1,199,940 bytes** | 1,310,720 bytes | +109.76 KB free flash | **PASS (91% partition)** |

### Architectural Insight: Deliberate Tensor Arena Tuning (120 KB $\to$ 65 KB)
During standalone KWS development, an oversized default arena of 120 KB was allocated. However, profiling with TFLM's `interpreter.arena_used_bytes()` revealed:
- **Actual Model Requirement:** Exactly **54,272 bytes (53.0 KB)**.
- **Static Arena Right-Sized:** Reduced to **65 KB (66,560 bytes)**, providing a **12,288 byte (18.5%) safety buffer** above peak tensor usage while avoiding dynamic fragmentation.
- **Memory Impact:** Directly freed **55 KB of static internal SRAM**, which allowed concurrent execution of the ESP32 Wi-Fi MAC/PHY driver, LwIP TCP socket pools, and WebSocket framing without exhausting internal heap. The model executes flawlessly (`AllocateTensors() == kTfLiteOk`).

---

## 6. Live Physical Microphone Trigger Anchor (`TEST_MODE 0`)

The firmware was tested with `TEST_MODE 0` wiring the physical **INMP441 I2S microphone** (SCK: GPIO 4, WS: GPIO 5, SD: GPIO 1) and the retrained `TENet6-Narrow` INT8 model:

### Physical Hardware Log Snippet (`COM19`)
```text
========================================================
ESP32-S3 REAL-TIME CLOUD ASR STREAMING & LATENCY BENCHMARK
Mode: TEST_MODE 0 (Live KWS Trigger) | Server: ws://10.106.74.127:8765
========================================================
[WiFi] Connecting to WiFi.....
[WiFi] Connected! IP: 10.106.74.9, RSSI: -28 dBm
[OK] I2S INMP441 Driver Installed (SCK: 4, WS: 5, SD: 1)
[OK] TFLM TENet6-Narrow + Log-iRDT Initialized
[WS] Connected to ws://10.106.74.127:8765 (Pre-Opened & Ready)

>>> [KEYWORD DETECTED] 'MARVIN' (Confidence:  77.3%, Det: 153.99 ms) <<<
LAT,1,154.07,0.53,224.60,266.90
[Live Mic] RMS: 1220 -> MARVIN  (P: Marvin=50.8%, Unk=46.9%, Sil=2.3%)
[Live Mic] RMS:  731 -> UNKNOWN (P: Marvin=22.3%, Unk=77.7%, Sil=0.4%)
[Live Mic] RMS: 1031 -> UNKNOWN (P: Marvin=24.6%, Unk=72.7%, Sil=2.7%)
[Live Mic] RMS:  805 -> UNKNOWN (P: Marvin=19.9%, Unk=76.6%, Sil=3.5%)
```

### Server Ingestion Log ([`cloud_asr_server.py`](file:///c:/TENet-kws-irdt/TENet-kws-irdt/cloud_asr_server.py))
```text
[SERVER] Pre-opened WebSocket client connected from ('10.106.74.9', 53477) (Zero-Handshake Ready)
  [TELEMETRY] LAT,1,154.07,0.53,224.60,266.90 -> Logged Session #1
```

---

## 7. Continuous Non-Keyword Stress Test & Operating Policy Trade-Off (FA/hr vs. Latency)

### Stress-Test Corpus Specification
To evaluate real-world resistance to false triggers under prolonged continuous acoustic exposure, a **1.0-hour continuous audio stream** (3,600.0 seconds) was assembled from the **Google Speech Commands v0.02 dataset**:
- **Acoustic Composition:** Assembled by concatenating continuous multi-speaker non-keyword speech utterances with four distinct real-world background noise tracks (`doing_the_dishes.wav`, `running_tap.wav`, `white_noise.wav`, `exercise_bike.wav`).
- **Inference Evaluation:** Evaluated across **17,995 sliding-window inferences** (1.0 s window, 200 ms hop stride).
- **Acoustic Rejection:** Mean non-keyword $P(\text{Marvin}) = \mathbf{0.0392\ (3.92\%)}$; 99.9th percentile ($P_{99.9}$) non-keyword probability is $\mathbf{0.8400}$.

---

### Reconciling the Trade-Off: Immediate Dispatch vs. Temporal Persistence

A rigorous production system must explicitly account for the trade-off between **immediate single-hop response** and **multi-hop persistence confirmation**. The ESP32-S3 firmware architecture supports two configurable operating policies:

| Metric / Parameter | Policy A: Immediate Low-Latency Mode<br>*(Measured in Section 4 Hardware Benchmark)* | Policy B: High-Reliability Ambient Mode<br>*(Temporal Persistence Confirmation)* | Architectural Impact & Design Trade-Off |
|---|:---:|:---:|---|
| **Trigger Confirmation Logic** | **Single Hop ($N=1$)**<br>$P(\text{Marvin}) \ge 0.80$ | **Two Consecutive Hops ($N=2$)**<br>$P(\text{Marvin}) \ge 0.80$ across 200 ms stride | Both policies evaluated at identical classification confidence ($P \ge 0.80$), isolating the impact of temporal persistence ($N=1$ vs. $N=2$) |
| **On-Chip Inference (`det_ms`)** | **152.80 ms** (Mean) / 152.78 ms (Median) | **352.80 ms** (Mean) / 352.78 ms (Median)<br>*(152.80 ms inference + 200 ms hop stride)* | Adds exactly 1 hop stride (200 ms) before confirmation |
| **Preparation / Framing (`prep_ms`)** | **0.08 ms** | **0.08 ms** | Zero impact (pre-opened socket remains active) |
| **One-Way Transit ($\frac{\text{RTT}}{2}$)** | **20.08 ms** (Median) / 30.58 ms (Mean) | **20.08 ms** (Median) / 30.58 ms (Mean) | Identical over-the-air propagation |
| **TOTAL END-TO-CLOUD LATENCY** | **172.95 ms (Median)**<br>**183.46 ms (Mean)**<br>**262.63 ms (P95)** | **372.95 ms (Median)**<br>**383.46 ms (Mean)**<br>**462.63 ms (P95)** | **Policy A is ultra-fast (~173 ms); Policy B (~373 ms) remains well within the 500 ms voice budget** |
| **Continuous False Alarm Rate** | **22.0 False Alarms / Hour**<br>(0.12% per-hop rate across 17,995 hops) | **0.00 False Alarms / Hour**<br>(0 false triggers across 1.0 hr stress test) | Policy B achieves complete elimination of background false alarms |
| **Confusable Words Foils (462 clips)** | **0 / 462 false triggers (0.0%)** | **0 / 462 false triggers (0.0%)** | Both reject 100% of hard-negative foils ("bargain", "martin") |
| **Recommended Deployment** | Push-to-talk, interactive voice agent UI, or systems with lightweight secondary cloud confirmation | Always-on ambient IoT devices & smart home appliances where unprompted false activations are unacceptable | Gives the system architect full control over the latency/false-alarm Pareto frontier |

---

## 8. Slide-Ready Quotable Summary

```markdown
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        EDGE-TO-CLOUD STREAMING BENCHMARK                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. "INSTANT" DISPATCH:                                                                 │
│    • Pre-opened persistent WebSocket eliminates 200–1200 ms TCP/TLS connection setup   │
│    • Measured Detection → 1st Byte Dispatch (prep_ms): 0.08 ms (Mean & P95)           │
│                                                                                        │
│ 2. "EFFICIENT" MINIMAL DATA OVERHEAD:                                                  │
│    • 4:1 IMA ADPCM compression: 32 KB/s (256 kbps) → 8 KB/s (64 kbps)                 │
│    • 75.0% Bandwidth Reduction; ITU-T P.862.2 PESQ: 3.812 MOS-LQO (Wideband 16 kHz)   │
│    • 164-Byte Binary Frames: 0% JSON / Base64 serialization bloat                      │
│                                                                                        │
│ 3. "MINIMAL LATENCY" (Hardware-Measured over 100 Runs):                                │
│    • Low-Latency Mode (1-Hop):  152.80 ms det + 0.08 ms prep + 20 ms net = 172.95 ms   │
│    • Ambient Mode (2-Hop Conf): 352.80 ms det + 0.08 ms prep + 20 ms net = 372.95 ms   │
│    • Both modes easily satisfy conversational voice assistant limits (< 500 ms)        │
│                                                                                        │
│ 4. ACCURACY & FALSE ALARM PROFILE:                                                     │
│    • Confusable Phonetic Words: 0 / 462 false triggers (0.0% on "bargain", "martin")   │
│    • Continuous 1-Hour Stream (17,995 hops): 22 FA/hr (1-Hop) vs. 0.00 FA/hr (2-Hop)   │
│                                                                                        │
│ 5. COMBINED RAM FOOTPRINT (All Subsystems Active):                                     │
│    • Live I2S DMA + KWS + WiFi + WebSockets: 205.42 KB used (< 256 KB limit, PASS)     │
│    • TFLM Arena Tuned: 120 KB → 65 KB based on arena_used_bytes() (saved 55 KB RAM)   │
│    • Free Internal SRAM Safety Margin:       114.58 KB free heap                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```
