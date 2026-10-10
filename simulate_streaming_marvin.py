import os
import wave
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from irdt.irdtv import irdtv_optimized
import models
import config

def build_model_args():
    return argparse.Namespace(
        sample_rate=16000,
        window_size_ms=30,
        window_stride_ms=10,
        background_frequency=0.8,
        background_max_volume=0.1,
        num_mel_bins=64,
        desired_samples=16000,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
        num_mfccs=40,
        timeshift_ms=100,
        wanted_words=['marvin'],
        dataset_path=r'C:\XboxGames\Dataset\speech_commands_v0.02',
        split_dir='configs/v1_irdt',
        steps=3000,
        batch_size=1,
        buffer_size=1000,
        workers=1,
        step_save_checkpoint=500,
        step_logging=50,
        max_to_keep=5,
        trainable_scopes='',
        optimizer='adam',
        lr_list=[0.002, 0.0005, 0.0001],
        boundaries=[1200, 2200],
        momentum=0.9,
        weight_decay=4e-5,
        input_file='',
        save_folder='save/TENet6NarrowIRDTModel',
        checkpoint_path=None,
        arch='TENet6NarrowIRDTModel',
        dropout=0.0,
        width_multiplier=1.0,
        num_classes=3,
        scope_name='',
        kernel_list=None,
        irdt_w=64,
        irdt_chan=[1, 2, 4, 8, 16, 32],
        irdt_win_per_segm=3,
        irdt_biti=24,
        irdt_log=False,
        mod='test',
        dataset_name='test',
        window_size_samples=480,
        window_stride_samples=160,
        spectrogram_length=98,
    )

def load_wav(data_root, rel_path):
    p = os.path.join(data_root, rel_path)
    with wave.open(p, 'rb') as w:
        buf = w.readframes(16000)
        a = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    if len(a) < 16000:
        a = np.pad(a, (0, 16000 - len(a)))
    return a[:16000]

def main():
    data_root = r'C:\XboxGames\Dataset\speech_commands_v0.02'
    model_args = build_model_args()

    raw_audio_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 16000, 1], name="raw_audio")
    feat_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, 83, 6], name="feat_ph")
    model = models.__dict__[model_args.arch](model_args)
    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope(), reuse=tf.compat.v1.AUTO_REUSE):
        model.build(wavs=raw_audio_ph, labels=tf.constant([0], dtype=tf.int64), is_training=False)
        logits_op = model.build_inference(feat_ph, is_training=False)

    sess = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
    sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
    saver = tf.compat.v1.train.Saver(var_list=model.model_variables)
    saver.restore(sess, 'save/TENet6NarrowIRDTModel/best/TENet6NarrowIRDTModel-best')

    # Background audio: continuous running tap
    bg_path = os.path.join(data_root, '_background_noise_', 'running_tap.wav')
    with wave.open(bg_path, 'rb') as w:
        bg_buf = w.readframes(16000 * 25)
        timeline = np.frombuffer(bg_buf, dtype=np.int16).astype(np.float32) / 32768.0
    timeline = timeline[:16000 * 25] * 0.35  # Continuous background noise floor

    def get_first_wav(subfolder):
        d = os.path.join(data_root, subfolder)
        files = [f for f in os.listdir(d) if f.endswith('.wav')]
        return os.path.join(subfolder, files[0])

    # 1. Marvin #1 at 3.0s (ends at 4.0s)
    m1 = load_wav(data_root, get_first_wav('marvin'))
    timeline[16000*3 : 16000*4] += m1 * 0.85

    # 2. Distractor 'learn' at 8.0s (ends at 9.0s)
    l1 = load_wav(data_root, get_first_wav('learn'))
    timeline[16000*8 : 16000*9] += l1 * 0.85

    # 3. Distractor 'down' at 13.0s (ends at 14.0s)
    d1 = load_wav(data_root, get_first_wav('down'))
    timeline[16000*13 : 16000*14] += d1 * 0.85

    # 4. Marvin #2 at 18.0s (ends at 19.0s)
    m_dir = os.path.join(data_root, 'marvin')
    m_files = [f for f in os.listdir(m_dir) if f.endswith('.wav')]
    m2 = load_wav(data_root, os.path.join('marvin', m_files[1]))
    timeline[16000*18 : 16000*19] += m2 * 0.85

    hop = 1600 # 100ms
    win = 16000 # 1000ms
    num_hops = (len(timeline) - win) // hop

    print(f"Timeline: {len(timeline)/16000:.1f}s, {num_hops} sliding hops of 100ms each")
    print("Events: [0-3s BG] -> [3-4s MARVIN1] -> [4-8s BG] -> [8-9s LEARN] -> [9-13s BG] -> [13-14s DOWN] -> [14-18s BG] -> [18-19s MARVIN2] -> [19-25s BG]\n")

    results = []
    for h in range(num_hops):
        t_sec = (h * hop) / 16000.0
        chunk = timeline[h*hop : h*hop + win]
        
        # Sliced active frames
        _, feat = irdtv_optimized(chunk, [1, 2, 4, 8, 16, 32], 64, 3, 24, False)
        f_b = feat[np.newaxis, :, :].astype(np.float32)
        
        # Temporal CV
        speech_ch = feat[:-1, 1:4]
        mu_c = np.mean(speech_ch, axis=0)
        var_c = np.var(speech_ch, axis=0)
        cv = np.mean(np.sqrt(var_c) / (mu_c + 1e-6))
        
        # Inference
        logits = sess.run(logits_op, feed_dict={feat_ph: f_b})[0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        p_marvin = float(probs[0])
        p_unknown = float(probs[1])
        p_noise = float(probs[2])
        
        results.append((t_sec, h, cv, p_marvin, p_unknown, p_noise))

    print(f"{'Time':<8} {'Hop':<6} {'CV':<8} {'p_marvin':<10} {'p_unknown':<10} {'p_noise':<10} {'Event Category'}")
    print("-" * 75)
    
    # Track consecutive frames where p_marvin >= 0.65
    current_run = 0
    runs = []
    
    for t_sec, h, cv, p_m, p_u, p_n in results:
        # Determine event identity
        event = "BACKGROUND"
        if 2.2 <= t_sec <= 3.8: event = "MARVIN #1"
        elif 7.2 <= t_sec <= 8.8: event = "LEARN"
        elif 12.2 <= t_sec <= 13.8: event = "DOWN"
        elif 17.2 <= t_sec <= 18.8: event = "MARVIN #2"
        
        if p_m >= 0.65:
            current_run += 1
        else:
            if current_run > 0:
                runs.append((event, current_run, t_sec - current_run*0.1))
                current_run = 0
                
        if p_m > 0.20 or "MARVIN" in event:
            print(f"{t_sec:<8.2f} {h:<6d} {cv:<8.3f} {p_m:<10.4f} {p_u:<10.4f} {p_n:<10.4f} {event}")

    print("\n" + "=" * 60)
    print("CONSECUTIVE RUNS WHERE p_marvin >= 0.65:")
    print("=" * 60)
    for ev, count, t_start in runs:
        print(f"Event near {t_start:.2f}s: {count} consecutive frames above 0.65 (Duration: {count*100}ms)")

    sess.close()

if __name__ == '__main__':
    main()
