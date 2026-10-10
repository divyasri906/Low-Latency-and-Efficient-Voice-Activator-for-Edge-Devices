import os
import glob
import wave
import numpy as np
import tensorflow as tf
from irdt.irdtv import irdtv_optimized

def main():
    model_path = 'best_marvin_model_log_int8.tflite'
    print(f"Loading TFLite model from {model_path}...")
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    files = sorted(glob.glob(r'C:\XboxGames\Dataset\speech_commands_v0.02\hard_negatives\*.wav'))
    print(f"Total hard-negative files found: {len(files)}")

    results_by_word = {}
    false_alarms_50 = 0
    false_alarms_80 = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        word = fname.split('_')[0]
        if word not in results_by_word:
            results_by_word[word] = {'total': 0, 'pred_marvin': 0, 'pred_unknown': 0, 'pred_silence': 0, 'probs_m': []}

        with wave.open(fpath, 'rb') as wf:
            buf = wf.readframes(wf.getnframes())
        raw = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        _, feat = irdtv_optimized(raw, [1, 2, 4, 8, 16, 32], 64, 5, 24, log=True)
        interp.set_tensor(inp_idx, np.expand_dims(feat.astype(np.float32), axis=0))
        interp.invoke()
        probs = interp.get_tensor(out_idx)[0]

        p_m, p_u, p_s = float(probs[0]), float(probs[1]), float(probs[2])
        pred = np.argmax(probs)

        results_by_word[word]['total'] += 1
        results_by_word[word]['probs_m'].append(p_m)
        if pred == 0:
            results_by_word[word]['pred_marvin'] += 1
            false_alarms_50 += 1
        elif pred == 1:
            results_by_word[word]['pred_unknown'] += 1
        else:
            results_by_word[word]['pred_silence'] += 1

        if p_m >= 0.80:
            false_alarms_80 += 1

    print("=" * 80)
    print("BASELINE MODEL EVALUATION ON FOILS (WITHOUT HARD-NEGATIVE TRAINING):")
    print(f"Total clips: {len(files)}")
    print(f"Overall False Alarm Rate (tau=0.50): {false_alarms_50}/{len(files)} ({false_alarms_50/len(files)*100:.1f}%)")
    print(f"Overall False Alarm Rate (tau=0.80): {false_alarms_80}/{len(files)} ({false_alarms_80/len(files)*100:.1f}%)")
    print("-" * 80)
    print(f"{'Word':<12} | {'Total':<6} | {'False Trigger (Marvin)':<24} | {'Correct (Unknown)':<18} | {'Avg P(Marvin)':<14}")
    print("-" * 80)
    for word, data in sorted(results_by_word.items()):
        fa_pct = data['pred_marvin'] / data['total'] * 100.0
        unk_pct = data['pred_unknown'] / data['total'] * 100.0
        avg_pm = np.mean(data['probs_m']) * 100.0
        print(f"{word:<12} | {data['total']:6d} | {data['pred_marvin']:3d} ({fa_pct:5.1f}%)               | {data['pred_unknown']:3d} ({unk_pct:5.1f}%)           | {avg_pm:5.1f}%")
    print("=" * 80)

if __name__ == '__main__':
    main()
