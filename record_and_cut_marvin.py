"""
Record 5-Second Voice Input and Automatically Center-Cut the 1-Second 'Marvin' Utterance.

Workflow:
1. Records exactly 5.0 seconds from the microphone (16 kHz, 16-bit Mono PCM).
2. Saves the full 5-second raw clip to: recordings/raw/
3. Detects the precise voice activity onset and offset of the spoken word using Short-Time Energy.
4. Centers the detected word and extracts an exact 1.0-second slice (16,000 samples).
5. Saves the calibrated 1.0-second clip to: recordings/clipped_1sec/
6. Runs real-time inference using the trained 3-Seed TENet6-Narrow Ensemble and displays the detection probability.
"""

import os
import time
import wave
import argparse
import datetime
import numpy as np
import sounddevice as sd
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from irdt.irdtv import irdtv_optimized
from run_ensemble_training_and_eval import get_model_probs

SAMPLE_RATE = 16000
RECORD_DURATION = 5.0  # seconds
TARGET_DURATION = 1.0  # seconds
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)  # 16,000 samples


def record_5sec_audio():
    total_samples = int(SAMPLE_RATE * RECORD_DURATION)
    print("\n" + "=" * 60)
    print(" >>> RECORDING 5 SECONDS OF AUDIO <<<")
    print(" Please speak 'MARVIN' clearly at any point during the 5 seconds.")
    print("=" * 60)
    for i in range(3, 0, -1):
        print(f" Starting in {i}...")
        time.sleep(0.7)

    print("\n>>> RECORDING NOW... (Speak 'Marvin'!) <<<")
    audio_int16 = sd.rec(total_samples, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    print(">>> RECORDING FINISHED! <<<\n")
    return audio_int16.flatten()


def detect_and_cut_word(audio_int16, sample_rate=16000, target_samples=16000):
    """
    Energy-based Voice Activity Detector that locates the spoken word
    and extracts an exact 1.0-second (16,000 samples) window centered on it.
    """
    audio_float = audio_int16.astype(np.float32) / 32768.0

    # Sliding window RMS energy: 40ms window, 10ms stride
    win_size = int(sample_rate * 0.040)   # 640 samples
    win_stride = int(sample_rate * 0.010) # 160 samples
    num_frames = (len(audio_float) - win_size) // win_stride + 1

    energies = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * win_stride
        frame = audio_float[start : start + win_size]
        energies[i] = np.sqrt(np.mean(frame ** 2))

    # Noise floor and peak energy
    noise_floor = np.percentile(energies, 20)
    peak_energy = np.max(energies)

    if peak_energy - noise_floor < 0.01:
        print("[WARNING] Very low speech volume detected! Defaulting to middle 1 second.")
        center_sample = len(audio_int16) // 2
    else:
        # Adaptive threshold (20% between noise floor and peak)
        threshold = noise_floor + 0.20 * (peak_energy - noise_floor)
        speech_frames = np.where(energies > threshold)[0]

        if len(speech_frames) == 0:
            center_sample = len(audio_int16) // 2
        else:
            # Find the largest contiguous speech segment
            diffs = np.diff(speech_frames)
            splits = np.where(diffs > 15)[0]  # Allow short pauses up to 150ms
            segments = np.split(speech_frames, splits + 1)
            largest_seg = max(segments, key=lambda s: np.sum(energies[s]))

            onset_frame = largest_seg[0]
            offset_frame = largest_seg[-1]

            onset_sample = onset_frame * win_stride
            offset_sample = offset_frame * win_stride + win_size
            center_sample = (onset_sample + offset_sample) // 2
            
            print(f"  [Detection] Word detected between {onset_sample/sample_rate:.2f}s and {offset_sample/sample_rate:.2f}s (duration: {(offset_sample-onset_sample)/sample_rate*1000:.0f} ms).")
            print(f"  [Alignment] Centering 1.0s window around t = {center_sample/sample_rate:.2f}s.")

    # Calculate 1-second slice bounds
    half_target = target_samples // 2
    start_idx = center_sample - half_target
    end_idx = center_sample + half_target

    # Boundary handling
    if start_idx < 0:
        start_idx = 0
        end_idx = target_samples
    elif end_idx > len(audio_int16):
        end_idx = len(audio_int16)
        start_idx = len(audio_int16) - target_samples

    cut_audio = audio_int16[start_idx:end_idx]
    assert len(cut_audio) == target_samples, f"Shape error: expected {target_samples}, got {len(cut_audio)}"
    return cut_audio, start_idx, end_idx


def save_wav(filename, audio_int16, sample_rate=16000):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    print(f"  --> Saved: {filename} ({len(audio_int16)/sample_rate:.2f}s, {sample_rate} Hz, Mono)")


def run_ensemble_inference(audio_int16, ckpts):
    audio_float = audio_int16.astype(np.float32) / 32768.0
    _, feat = irdtv_optimized(audio_float, [1, 2, 4, 8, 16, 32], 64, 5, 24, False)
    X = np.expand_dims(feat, axis=0)  # Shape: (1, 50, 6)

    probs = []
    for ckpt in ckpts:
        p = get_model_probs(X, ckpt, r"C:\XboxGames\Dataset\speech_commands_v0.02", "configs/v1_irdt")
        probs.append(p)
    p_ens = (probs[0] + probs[1] + probs[2]) / 3.0

    p_marvin = p_ens[0, 0]
    p_unk = p_ens[0, 1]
    p_noise = p_ens[0, 2]

    print("\n" + "=" * 60)
    print(" 3-SEED ENSEMBLE CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"  Confidence Scores:")
    print(f"    - P(Marvin):           {p_marvin * 100:6.2f}%")
    print(f"    - P(Unknown Speech):   {p_unk * 100:6.2f}%")
    print(f"    - P(Background Noise): {p_noise * 100:6.2f}%")
    print("-" * 60)

    # Decisions at standard thresholds
    if p_marvin >= 0.95:
        print("  Status: [STRONG WAKE WORD DETECTED] (P >= 0.95 - Zero False Alarm Regime)")
    elif p_marvin >= 0.80:
        print("  Status: [WAKE WORD DETECTED] (P >= 0.80 - High Recall Regime)")
    elif p_marvin >= 0.50:
        print("  Status: [BORDERLINE DETECTION] (0.50 <= P < 0.80)")
    else:
        print("  Status: [NOT DETECTED / REJECTED]")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Record 5s audio and auto-cut 1s Marvin")
    parser.add_argument("--speaker", default="user", help="Speaker name or ID for file naming")
    parser.add_argument("--out_dir", default="recordings", help="Output directory root")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_filename = os.path.join(args.out_dir, "raw", f"{args.speaker}_5sec_{timestamp}.wav")
    cut_filename = os.path.join(args.out_dir, "clipped_1sec", f"{args.speaker}_marvin_{timestamp}.wav")

    # 1. Record 5 seconds
    audio_5sec = record_5sec_audio()

    # 2. Save full 5-second audio
    print("\n[Step 1] Saving original 5-second recording...")
    save_wav(raw_filename, audio_5sec, SAMPLE_RATE)

    # 3. Detect and cut exactly 1.0 second
    print("\n[Step 2] Automatically locating and extracting 'Marvin'...")
    cut_audio, start_s, end_s = detect_and_cut_word(audio_5sec, SAMPLE_RATE, TARGET_SAMPLES)
    save_wav(cut_filename, cut_audio, SAMPLE_RATE)

    # 4. Live Model Inference
    ckpts = [
        "save/ensemble_w5/seed_42/TENet6NarrowIRDTModel-3000",
        "save/ensemble_w5/seed_101/TENet6NarrowIRDTModel-3000",
        "save/ensemble_w5/seed_2024/TENet6NarrowIRDTModel-3000",
    ]
    if all(os.path.exists(c + ".index") for c in ckpts):
        print("\n[Step 3] Testing 1-second clip with 3-Seed Ensemble...")
        run_ensemble_inference(cut_audio, ckpts)
    else:
        print("\n[Step 3] Checkpoints not found in default paths, skipping inference.")

    print(f"Collection complete! You can run this script repeatedly to collect samples.")


if __name__ == "__main__":
    main()
