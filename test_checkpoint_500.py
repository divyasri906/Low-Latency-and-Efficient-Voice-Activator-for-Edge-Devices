import os
import glob
import wave
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import models
from irdt.irdtv import irdtv_optimized

CKPT_PATH = "save/tenet_log_hardneg/TENet6NarrowIRDTModel-500"
FOILS_DIR = r"C:\XboxGames\Dataset\speech_commands_v0.02\hard_negatives"

def main():
    print("Testing Checkpoint 500 on Hard-Negative Foils...")
    g = tf.Graph()
    with g.as_default():
        feat_ph = tf.compat.v1.placeholder(tf.float32, [1, 50, 6], name='input_features')
        model_args = argparse.Namespace(
            sample_rate=16000, window_size_ms=30, window_stride_ms=10,
            background_frequency=0.8, background_max_volume=0.1, num_mel_bins=64,
            desired_samples=16000, lower_edge_hertz=80.0, upper_edge_hertz=7600.0,
            num_mfccs=40, timeshift_ms=100, wanted_words=['marvin'],
            dataset_path='', split_dir='', steps=3000,
            batch_size=1, buffer_size=1000, workers=1,
            step_save_checkpoint=500, step_logging=50, max_to_keep=5,
            trainable_scopes='', optimizer='adam', lr_list=[0.002, 0.0005, 0.0001],
            boundaries=[1200, 2200], momentum=0.9, weight_decay=4e-5,
            input_file='', save_folder='', checkpoint_path=None,
            arch='TENet6NarrowIRDTModel', dropout=0.0, width_multiplier=1.0,
            num_classes=3, scope_name='', kernel_list=None, irdt_w=64,
            irdt_chan=[1, 2, 4, 8, 16, 32], irdt_win_per_segm=5, irdt_biti=24,
            irdt_log=True, mod='eval', dataset_name='eval',
            window_size_samples=480, window_stride_samples=160, spectrogram_length=98,
        )
        model = models.TENet6NarrowIRDTModel(model_args)
        logits_op = model.build_inference(feat_ph, is_training=False)
        probs_op = tf.nn.softmax(logits_op, name='output_probs')

        sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, CKPT_PATH)

        files = sorted(glob.glob(os.path.join(FOILS_DIR, "*.wav")))
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
            
            p = sess.run(probs_op, feed_dict={feat_ph: np.expand_dims(feat.astype(np.float32), axis=0)})[0]
            p_m, p_u, p_s = float(p[0]), float(p[1]), float(p[2])
            pred = np.argmax(p)

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

        sess.close()

    print("=" * 80)
    print(f"CHECKPOINT 500 EVALUATION ON {len(files)} FOIL CLIPS:")
    print(f"False Trigger Rate @ tau=0.50: {false_alarms_50}/{len(files)} ({false_alarms_50/len(files)*100:.1f}%)")
    print(f"False Trigger Rate @ tau=0.80: {false_alarms_80}/{len(files)} ({false_alarms_80/len(files)*100:.1f}%)")
    print("-" * 80)
    print(f"{'Word':<12} | {'Total':<6} | {'False Trigger (Marvin)':<24} | {'Correct (Unknown)':<18} | {'Avg P(Marvin)':<14}")
    print("-" * 80)
    for word, data in sorted(results_by_word.items()):
        fa_pct = data['pred_marvin'] / data['total'] * 100.0
        unk_pct = data['pred_unknown'] / data['total'] * 100.0
        avg_pm = np.mean(data['probs_m']) * 100.0
        print(f"{word:<12} | {data['total']:6d} | {data['pred_marvin']:3d} ({fa_pct:5.1f}%)               | {data['pred_unknown']:3d} ({unk_pct:5.1f}%)           | {avg_pm:5.2f}%")
    print("=" * 80)

if __name__ == '__main__':
    main()
