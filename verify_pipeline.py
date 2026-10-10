"""
Standalone verification of the iRDT -> TENet6-Narrow integration, using
synthetic random audio/labels (no real dataset required) so it can be run
immediately to check graph-construction and training/checkpoint mechanics
before pointing the real pipeline (main.py) at actual data.

Requires the exact environment this repository specifies: tensorflow==1.15.0
(Python <=3.7). This script uses only APIs already exercised elsewhere in
this exact repository (audio_nets/tenet.py, models.py, helper.py), and was
verified to run correctly end-to-end (all 6 steps below) using a TF1.15-API
compatibility shim (tf_slim + tf.compat.v1 graph mode) under TF2.21/Python
3.12, since a genuine TF1.15 wheel does not exist for that Python version.
Run it in the repo's own stated environment (tensorflow==1.15.0) for the
authoritative result.

Run with:
    python -m verify_pipeline
"""

import argparse
import os
import shutil

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


import models
from irdt.irdtv import irdtv_output_shape


def build_fake_args():
    """Mimics config.update_args()'s output -- only the fields
    TENet6NarrowIRDTModel/preprocess_irdt/build_inference actually read."""
    args = argparse.Namespace()

    args.desired_samples = 16000
    args.sample_rate = 16000

    args.irdt_w = 64
    args.irdt_chan = [1, 2, 4, 8, 16, 32]
    args.irdt_win_per_segm = 3
    args.irdt_biti = 24
    args.irdt_log = False

    args.num_classes = 3
    args.weight_decay = 4e-5
    args.dropout = 0.5
    args.width_multiplier = 1.0
    args.scope_name = ""
    args.kernel_list = None

    return args


def main():
    args = build_fake_args()
    batch_size = 8

    expected_M, expected_m = irdtv_output_shape(
        args.desired_samples, args.irdt_w, args.irdt_chan, args.irdt_win_per_segm)
    print(f"[check] irdtv_output_shape(...) = ({expected_M}, {expected_m})")
    assert (expected_M, expected_m) == (83, 6), (
        f"Expected iRDT output shape (83, 6), got ({expected_M}, {expected_m})"
    )
    print("[check] PASSED: iRDT config produces exactly (83, 6)\n")

    tf.compat.v1.reset_default_graph()

    # -------- stage 1: audio --------
    wavs_ph = tf.compat.v1.placeholder(
        tf.float32, shape=[batch_size, args.desired_samples, 1], name="wavs")
    labels_ph = tf.compat.v1.placeholder(
        tf.int64, shape=[batch_size], name="labels")
    print(f"audio:                [batch,{args.desired_samples}]  "
          f"-> placeholder shape {wavs_ph.shape}")

    # -------- stage 2-6: model (iRDT -> TENet6Narrow -> logits) --------
    model = models.__dict__["TENet6NarrowIRDTModel"](args)
    model.build(wavs=wavs_ph, labels=labels_ph, is_training=True)

    print(f"iRDT:                 [batch,83,6]        -> actual {model.audio.shape}")
    print(f"final logits:          [batch,3]           -> actual {model.logits.shape}")
    assert tuple(model.audio.shape.as_list()) == (batch_size, 83, 6), (
        f"iRDT feature tensor shape mismatch: {model.audio.shape}"
    )
    assert tuple(model.logits.shape.as_list()) == (batch_size, args.num_classes), (
        f"logits shape mismatch: {model.logits.shape}"
    )
    print("[check] PASSED: TENet6NarrowIRDTModel produces [batch, 3] logits\n")

    # -------- optimizer / train op (mirrors helper.py::Trainer exactly) --------
    import tf_slim as slim

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    variables_to_train = tf.compat.v1.trainable_variables()
    train_op = slim.learning.create_train_op(
        model.total_loss, optimizer, global_step=global_step,
        variables_to_train=variables_to_train,
    )

    saver = tf.compat.v1.train.Saver(var_list=model.model_variables, max_to_keep=1)

    ckpt_dir = "/tmp/verify_irdt_tenet_ckpt"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir)
    ckpt_prefix = os.path.join(ckpt_dir, "TENet6NarrowIRDTModel")

    rng = np.random.default_rng(0)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        fake_wavs = rng.uniform(-1, 1, size=(batch_size, args.desired_samples, 1)).astype(np.float32)
        fake_labels = rng.integers(0, args.num_classes, size=(batch_size,)).astype(np.int64)

        # 1. one forward pass
        logits_val = sess.run(model.logits, feed_dict={wavs_ph: fake_wavs, labels_ph: fake_labels})
        print(f"[1] forward pass OK, logits shape = {logits_val.shape}")
        assert logits_val.shape == (batch_size, args.num_classes)

        # 2. loss calculation
        loss_val = sess.run(model.total_loss, feed_dict={wavs_ph: fake_wavs, labels_ph: fake_labels})
        print(f"[2] loss calculation OK, total_loss = {loss_val:.6f}")

        # 3. one training step
        _, loss_after = sess.run(
            [train_op, model.total_loss],
            feed_dict={wavs_ph: fake_wavs, labels_ph: fake_labels},
        )
        print(f"[3] one training step OK, loss after step = {loss_after:.6f}")

        # 4. checkpoint save
        saved_path = saver.save(sess, ckpt_prefix, global_step=1)
        print(f"[4] checkpoint save OK -> {saved_path}")

    # 5. checkpoint restore (fresh graph/session, mirrors helper.py::Base.routine_restore_and_initialize)
    tf.compat.v1.reset_default_graph()
    wavs_ph = tf.compat.v1.placeholder(
        tf.float32, shape=[batch_size, args.desired_samples, 1], name="wavs")
    labels_ph = tf.compat.v1.placeholder(
        tf.int64, shape=[batch_size], name="labels")
    model2 = models.__dict__["TENet6NarrowIRDTModel"](args)
    model2.build(wavs=wavs_ph, labels=labels_ph, is_training=False)
    saver2 = tf.compat.v1.train.Saver(var_list=model2.model_variables)

    with tf.compat.v1.Session() as sess2:
        sess2.run(tf.compat.v1.global_variables_initializer())
        sess2.run(tf.compat.v1.local_variables_initializer())
        saver2.restore(sess2, saved_path)
        print(f"[5] checkpoint restore OK <- {saved_path}")

        # 6. one evaluation pass
        eval_wavs = rng.uniform(-1, 1, size=(batch_size, args.desired_samples, 1)).astype(np.float32)
        eval_labels = rng.integers(0, args.num_classes, size=(batch_size,)).astype(np.int64)
        acc_val, logits_val2 = sess2.run(
            [model2.acc, model2.logits],
            feed_dict={wavs_ph: eval_wavs, labels_ph: eval_labels},
        )
        print(f"[6] evaluation pass OK, acc = {acc_val:.4f}, logits shape = {logits_val2.shape}")

    print("\nALL VERIFICATION STEPS PASSED.")


if __name__ == "__main__":
    main()
