import os
import glob
import wave
import numpy as np
import tensorflow as tf
from irdt.irdtv import irdtv_optimized

def main():
    model_path = "best_marvin_model_log_int8.tflite"
    clips_dir = r"c:\TENet-kws-irdt\clipped_1sec"

    print("=" * 95)
    print(" EVALUATING USER 'clipped_1sec' AUDIO CLIPS ON TENet6-Narrow + Log-iRDT")
    print(f" Model File : {model_path}")
    print(f" Clips Dir  : {clips_dir}")
    print("=" * 95)

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    files = sorted(glob.glob(os.path.join(clips_dir, "*.wav")))
    if not files:
        print(f"[!] No WAV files found in {clips_dir}!")
        return

    def eval_audio(arr):
        af = arr.astype(np.float32) / 32768.0
        _, feat = irdtv_optimized(af, [1, 2, 4, 8, 16, 32], 64, 5, 24, log=True)
        interp.set_tensor(inp_idx, np.expand_dims(feat.astype(np.float32), axis=0))
        interp.invoke()
        return interp.get_tensor(out_idx)[0]

    raw_detected_50 = 0
    raw_detected_80 = 0
    streaming_detected_50 = 0
    streaming_detected_80 = 0

    print(f"{'#':<3} | {'File Name':<32} | {'RMS':<6} | {'P(Marvin)':<10} | {'P(Unknown)':<10} | {'Decision':<10} | {'Best P(Marvin) w/ shift':<24}")
    print("-" * 110)

    results = []

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        with wave.open(fpath, 'rb') as wf:
            n_ch = wf.getnchannels()
            n_frames = wf.getnframes()
            buf = wf.readframes(n_frames)
        raw = np.frombuffer(buf, dtype=np.int16)
        if n_ch > 1:
            raw = raw[::n_ch]
        if len(raw) < 16000:
            raw = np.pad(raw, (0, 16000 - len(raw)))
        else:
            raw = raw[:16000]

        rms = float(np.sqrt(np.mean(raw.astype(np.float32)**2)))

        # 1. Exact static 1.0s window
        probs = eval_audio(raw)
        p_m = float(probs[0])
        p_u = float(probs[1])
        p_s = float(probs[2])
        pred = 'MARVIN' if (p_m > p_u and p_m > p_s) else ('UNKNOWN' if p_u > p_s else 'SILENCE')

        if p_m >= 0.50:
            raw_detected_50 += 1
        if p_m >= 0.80:
            raw_detected_80 += 1

        # 2. Sliding window simulation (shifts: -150ms to +150ms in 25ms steps)
        best_p_m = p_m
        for shift_ms in [-150, -100, -75, -50, -25, 25, 50, 75, 100, 150]:
            shift_samples = int(shift_ms * 16)
            if shift_samples < 0:
                s_arr = np.pad(raw[-shift_samples:], (0, -shift_samples))
            else:
                s_arr = np.pad(raw[:16000-shift_samples], (shift_samples, 0))
            p_shifted = eval_audio(s_arr)
            if float(p_shifted[0]) > best_p_m:
                best_p_m = float(p_shifted[0])

        if best_p_m >= 0.50:
            streaming_detected_50 += 1
        if best_p_m >= 0.80:
            streaming_detected_80 += 1

        res_tag = "DETECTED" if p_m >= 0.50 else "MISSED"
        print(f"{i+1:<3} | {fname:<32} | {rms:6.1f} | {p_m*100:6.1f}%    | {p_u*100:6.1f}%    | {res_tag:<10} | {best_p_m*100:6.1f}%")
        results.append({
            "file": fname,
            "rms": rms,
            "p_marvin": p_m,
            "p_unknown": p_u,
            "p_silence": p_s,
            "best_p_marvin_shifted": best_p_m
        })

    total = len(files)
    print("=" * 110)
    print(f"\nRESULTS SUMMARY FOR USER RECORDED 'clipped_1sec' MARVIN CLIPS (N = {total}):")
    print("-" * 75)
    print(f"1. Static Fixed-Crop 1.0s Window Evaluation:")
    print(f"   - Marvin Recall @ threshold tau=0.50 (Standard):  {raw_detected_50}/{total} ({raw_detected_50/total*100:.2f}%)")
    print(f"   - Marvin Recall @ threshold tau=0.80 (High Conf): {raw_detected_80}/{total} ({raw_detected_80/total*100:.2f}%)")
    print("-" * 75)
    print(f"2. Dynamic Sliding Window Evaluation (Live Stream Alignment):")
    print(f"   - Marvin Recall @ threshold tau=0.50 (Standard):  {streaming_detected_50}/{total} ({streaming_detected_50/total*100:.2f}%)")
    print(f"   - Marvin Recall @ threshold tau=0.80 (High Conf): {streaming_detected_80}/{total} ({streaming_detected_80/total*100:.2f}%)")
    print("=" * 75)

if __name__ == '__main__':
    main()
