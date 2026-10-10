import os
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import models
from evaluate_optimized_w5 import load_split_data

CKPT_PATH = "save/tenet_log_hardneg/TENet6NarrowIRDTModel-500"
DATASET_PATH = r"C:\XboxGames\Dataset\speech_commands_v0.02"
SPLIT_FILE = "configs/v1_irdt_hardneg/test.txt"

def main():
    print(f"Loading test split from {SPLIT_FILE}...")
    X_test, y_test, _, _ = load_split_data(DATASET_PATH, SPLIT_FILE, w=5, irdt_log=True)

    g = tf.Graph()
    with g.as_default():
        feat_ph = tf.compat.v1.placeholder(tf.float32, [None, 50, 6], name='input_features')
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

        probs = sess.run(probs_op, feed_dict={feat_ph: X_test})
        sess.close()

    preds = np.argmax(probs, axis=1)
    acc = float(np.mean(preds == y_test) * 100.0)

    # Class 0: Marvin, Class 1: Unknown, Class 2: Background
    y_m = (y_test == 0)
    p_m = (preds == 0)
    tp = int(np.sum(y_m & p_m))
    fn = int(np.sum(y_m & ~p_m))
    fp = int(np.sum(~y_m & p_m))
    tn = int(np.sum(~y_m & ~p_m))
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0

    print("=" * 80)
    print("CHECKPOINT 500 TEST SPLIT METRICS:")
    print(f"Total Test Samples : {len(y_test)}")
    print(f"Overall Accuracy   : {acc:.2f}%")
    print(f"Marvin Recall      : {recall:.2f}% ({tp}/{tp+fn})")
    print(f"Marvin Precision   : {precision:.2f}% ({tp}/{tp+fp})")
    print(f"Confusion Matrix   : TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print("=" * 80)

if __name__ == '__main__':
    main()
