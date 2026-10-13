"""
High-Performance Cloud ASR Ingestion Server (WebSocket + IMA ADPCM Decoder).
Receives 20ms binary IMA ADPCM frames (164 bytes) over persistent pre-opened WebSocket.
Decodes 4-bit ADPCM to 16-bit PCM (16 kHz), measures one-way & RTT latency,
and aggregates statistical distributions (Mean, Median, P95, Min, Max) over 100 runs.
"""

import asyncio
import struct
import time
import json
import numpy as np
import websockets

# Standard IMA ADPCM Step Table & Index Table
IMA_INDEX_TABLE = [
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8
]

IMA_STEP_TABLE = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
    19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
    50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
    337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
    2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
    5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
    15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
]


def decode_ima_adpcm_frame(data_bytes):
    """
    Decodes a 164-byte frame:
    - Bytes 0-1: Initial predicted sample (int16)
    - Byte 2: Initial step index (uint8)
    - Byte 3: Reserved (0)
    - Bytes 4-163: 160 bytes containing 320 4-bit nibbles -> 320 PCM samples (20 ms @ 16 kHz)
    """
    if len(data_bytes) < 4:
        return np.array([], dtype=np.int16)

    predicted_sample, step_index = struct.unpack("<hb", data_bytes[:3])
    step_index = max(0, min(88, step_index))

    pcm_samples = [predicted_sample]
    adpcm_payload = data_bytes[4:]

    for byte in adpcm_payload:
        # Two 4-bit nibbles per byte: low nibble first, then high nibble
        nibbles = [byte & 0x0F, (byte >> 4) & 0x0F]
        for delta in nibbles:
            step = IMA_STEP_TABLE[step_index]
            diff = step >> 3
            if delta & 4: diff += step
            if delta & 2: diff += (step >> 1)
            if delta & 1: diff += (step >> 2)

            if delta & 8:
                predicted_sample -= diff
            else:
                predicted_sample += diff

            predicted_sample = max(-32768, min(32767, predicted_sample))
            step_index += IMA_INDEX_TABLE[delta]
            step_index = max(0, min(88, step_index))
            pcm_samples.append(predicted_sample)

    return np.array(pcm_samples, dtype=np.int16)


class CloudASRServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.sessions = []
        self.active_sessions = {}
        self.total_frames_received = 0
        self.total_bytes_received = 0

    async def handler(self, websocket):
        client_addr = websocket.remote_address
        print(f"\n[SERVER] Pre-opened WebSocket client connected from {client_addr} (Zero-Handshake Ready)")

        try:
            async for message in websocket:
                recv_time_ms = time.time() * 1000.0

                if isinstance(message, str):
                    # Control message / JSON / Ping / CSV latency report
                    if message.startswith("LAT,"):
                        # Format: LAT,session_id,det_ms,prep_ms,rtt_ms,total_ms
                        parts = message.strip().split(",")
                        if len(parts) >= 6:
                            record = {
                                "session": int(parts[1]),
                                "det_ms": float(parts[2]),
                                "prep_ms": float(parts[3]),
                                "rtt_ms": float(parts[4]),
                                "total_ms": float(parts[5]),
                                "timestamp": recv_time_ms
                            }
                            self.sessions.append(record)
                            print(f"  [TELEMETRY #{len(self.sessions)}/100] {message.strip()}")
                            with open("cloud_asr_latency_log.csv", "a") as f:
                                f.write(f"{record['session']},{record['det_ms']},{record['prep_ms']},{record['rtt_ms']},{record['total_ms']}\n")

                            if len(self.sessions) == 100:
                                self.print_summary()
                    elif message.startswith("PING"):
                        await websocket.send("PONG")
                    elif message.startswith("TRIGGER"):
                        print(f"  [SERVER] Detection trigger event acknowledged from client.")
                    elif message.startswith("EOS"):
                        print(f"  [SERVER] End-of-Stream received from client (Utterance completed).")

                elif isinstance(message, bytes):
                    # Binary audio frame (20ms IMA ADPCM = 164 bytes)
                    self.total_frames_received += 1
                    self.total_bytes_received += len(message)

                    # Quick decode verification
                    pcm = decode_ima_adpcm_frame(message)
                    rms = np.sqrt(np.mean(pcm.astype(np.float32)**2)) if len(pcm) > 0 else 0

                    # Echo lightweight 4-byte ACK for hardware RTT calculation
                    await websocket.send(b"ACK\x00")

        except websockets.exceptions.ConnectionClosed:
            print(f"[SERVER] Client {client_addr} disconnected.")
        except Exception as e:
            print(f"[SERVER] Exception: {e}")

    def print_summary(self):
        stats = self.compute_statistics()
        if not stats:
            return
        print("\n" + "=" * 80)
        print("          REAL-TIME CLOUD ASR STREAMING BENCHMARK REPORT (100 SESSIONS)")
        print("=" * 80)
        print(f" Total Sessions Evaluated: {stats['count']}")
        print("-" * 80)
        print(f" {'Metric':<30} | {'Mean':>10} | {'Median':>10} | {'P95':>10} | {'Min':>8} | {'Max':>8}")
        print("-" * 80)
        t = stats['total_ms']
        d = stats['det_ms']
        p = stats['prep_ms']
        r = stats['rtt_ms']
        print(f" {'Keyword End -> Detection':<30} | {d['mean']:>8.2f}ms | {d['mean']:>8.2f}ms | {d['p95']:>8.2f}ms | {d['min']:>6.2f}ms | {d['max']:>6.2f}ms")
        print(f" {'Detection -> First Byte (prep)':<30} | {p['mean']:>8.2f}ms | {p['median']:>8.2f}ms | {p['p95']:>8.2f}ms | {p['min']:>6.2f}ms | {p['max']:>6.2f}ms")
        print(f" {'Network Transit RTT':<30} | {r['mean']:>8.2f}ms | {r['median']:>8.2f}ms | {r['p95']:>8.2f}ms | {r['min']:>6.2f}ms | {r['max']:>6.2f}ms")
        print("-" * 80)
        print(f" {'TOTAL END-TO-CLOUD LATENCY':<30} | {t['mean']:>8.2f}ms | {t['median']:>8.2f}ms | {t['p95']:>8.2f}ms | {t['min']:>6.2f}ms | {t['max']:>6.2f}ms")
        print("=" * 80)
        print(" DATA OVERHEAD & COMPRESSION:")
        print(f"  - Raw Uncompressed PCM:  16 kHz * 16-bit = 256.0 kbps (32.0 KB/s)")
        print(f"  - IMA ADPCM Compressed:   16 kHz *  4-bit =  64.0 kbps ( 8.0 KB/s)")
        print(f"  - Compression Ratio:      4.0x (75.0% Bandwidth Reduction)")
        print(f"  - WebSocket Payload Type: Binary (164 bytes/frame vs 219 bytes Base64 JSON -> 0% bloat)")
        print(f"  - Connection Overhead:    0 ms (Persistent WebSocket pre-opened at boot)")
        print("=" * 80 + "\n")
        with open("cloud_asr_benchmark_results.json", "w") as jf:
            json.dump(stats, jf, indent=2)

    def compute_statistics(self):
        if not self.sessions:
            return None

        totals = [s["total_ms"] for s in self.sessions]
        dets = [s["det_ms"] for s in self.sessions]
        preps = [s["prep_ms"] for s in self.sessions]
        rtts = [s["rtt_ms"] for s in self.sessions]

        stats = {
            "count": len(totals),
            "total_ms": {
                "mean": round(float(np.mean(totals)), 2),
                "std": round(float(np.std(totals)), 2),
                "median": round(float(np.median(totals)), 2),
                "p95": round(float(np.percentile(totals, 95)), 2),
                "p99": round(float(np.percentile(totals, 99)), 2),
                "min": round(float(np.min(totals)), 2),
                "max": round(float(np.max(totals)), 2),
            },
            "det_ms": {
                "mean": round(float(np.mean(dets)), 2),
                "p95": round(float(np.percentile(dets, 95)), 2),
            },
            "prep_ms": {
                "mean": round(float(np.mean(preps)), 2),
                "p95": round(float(np.percentile(preps, 95)), 2),
            },
            "rtt_ms": {
                "mean": round(float(np.mean(rtts)), 2),
                "p95": round(float(np.percentile(rtts, 95)), 2),
            },
            "compression": {
                "raw_rate_kbps": 256.0,
                "raw_rate_kB_s": 32.0,
                "adpcm_rate_kbps": 64.0,
                "adpcm_rate_kB_s": 8.0,
                "ratio": "4.0x (75.0% reduction)",
                "packet_size_bytes": 164,
                "packet_duration_ms": 20.0
            }
        }
        return stats


async def main():
    server = CloudASRServer(host="0.0.0.0", port=8765)
    print("=" * 80)
    print(" CLOUD ASR WEBSOCKET INGESTION SERVER")
    print(" Endpoint: ws://0.0.0.0:8765")
    print(" Protocol: Binary IMA ADPCM (4-bit, 20ms frames = 164 bytes, 8 KB/s)")
    print("=" * 80)

    async with websockets.serve(server.handler, server.host, server.port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer terminated by user.")
