import tensorflow as tf
import numpy as np
import os
import argparse
import config
import models
from irdt.dataset import MarvinAudioWrapper
from irdt.labels import MARVIN_LABEL_PERMUTATION, remap_labels

tf.compat.v1.disable_eager_execution()

def build_mtenet_args(seed=42, kernel_list=[3, 9], mod="eval"):
    return argparse.Namespace(
        sample_rate=16000, window_size_ms=30, window_stride_ms=10, background_frequency=0.8,
        background_max_volume=0.1, num_mel_bins=64, desired_samples=16000, lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0, num_mfccs=40, timeshift_ms=100, wanted_words=["marvin"],
        dataset_path=r"C:\XboxGames\Dataset\speech_commands_v0.02", split_dir="configs/v1_irdt",
        steps=3000, batch_size=100, buffer_size=1000, workers=1, step_save_checkpoint=1000,
        step_logging=100, max_to_keep=5, trainable_scopes="", optimizer="adam",
        lr_list=[0.002, 0.0005, 0.0001], boundaries=[1200, 2200], momentum=0.9, weight_decay=4e-5,
        input_file="", save_folder="save/MTENet_Fusion", checkpoint_path=None, arch="TENet6NarrowIRDTModel",
        dropout=0.0, width_multiplier=1.0, num_classes=3, scope_name="", kernel_list=kernel_list,
        irdt_w=64, irdt_chan=[1, 2, 4, 8, 16, 32], irdt_win_per_segm=3, irdt_biti=24, irdt_log=False,
        mod=mod, dataset_name=mod, window_size_samples=480, window_stride_samples=160,
        spectrogram_length=98, seed=seed,
    )

def test_fusion():
    ckpt_m = "save/MTENet_Fusion/seed_42/best_mtenet/MTENet6NarrowIRDTModel-best"
    reader = tf.compat.v1.train.NewCheckpointReader(ckpt_m)
    var_map = reader.get_variable_to_shape_map()

    # Build target TENet graph
    args_t = build_mtenet_args(kernel_list=None, mod="valid")
    g = tf.Graph()
    with g.as_default():
        sess = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
        dataset = MarvinAudioWrapper(args_t, "valid", False, sess)
        wavs, labels = dataset.get_input_and_output_op()
        labels = remap_labels(labels, MARVIN_LABEL_PERMUTATION)

        model_t = models.__dict__[args_t.arch](args_t)
        model_t.build(wavs=wavs, labels=labels, is_training=False)

        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])

        # Target variables map
        t_vars = {v.name.split(":")[0]: v for v in tf.compat.v1.global_variables() if v.name.startswith("TENet6Narrow")}
        
        assign_ops = []

        # 1. Copy identical non-depthwise variables
        for m_name in var_map.keys():
            if not m_name.startswith("MTENet6Narrow"):
                continue
            t_name = m_name.replace("MTENet6Narrow", "TENet6Narrow")
            if "depthwise" not in m_name and t_name in t_vars:
                val = reader.get_tensor(m_name)
                assign_ops.append(t_vars[t_name].assign(val))

        # 2. Fuse depthwise branches
        # blocks 0, 1, 2; layers 0, 1
        blocks = [0, 1, 2]
        layers = [0, 1]
        eps = 1e-3  # tf.keras / tf.layers batch norm epsilon is typically 1e-3

        for b in blocks:
            for l in layers:
                # Target names
                t_w_name = f"TENet6Narrow/block{b}/depthwise_conv{l}/depthwise_weights"
                t_gamma_name = f"TENet6Narrow/block{b}/depthwise_conv{l}_bn/gamma"
                t_beta_name = f"TENet6Narrow/block{b}/depthwise_conv{l}_bn/beta"
                t_mean_name = f"TENet6Narrow/block{b}/depthwise_conv{l}_bn/moving_mean"
                t_var_name = f"TENet6Narrow/block{b}/depthwise_conv{l}_bn/moving_variance"

                fused_w = np.zeros((9, 1, 48, 1), dtype=np.float32)
                fused_b = np.zeros((48,), dtype=np.float32)

                # Branch 0: kernel 3
                w0 = reader.get_tensor(f"MTENet6Narrow/block{b}/depthwise_conv{l}_0/depthwise_weights") # (3, 1, 48, 1)
                g0 = reader.get_tensor(f"MTENet6Narrow/block{b}/depthwise_conv{l}_0_bn/gamma")
                b0 = reader.get_tensor(f"MTENet6Narrow/block{b}/depthwise_conv{l}_0_bn/beta")
                m0 = reader.get_tensor(f"MTENet6Narrow/block{b}/depthwise_conv{l}_0_bn/moving_mean")
                v0 = reader.get_tensor(f"MTENet6Narrow/block{b}/depthwise_conv{l}_0_bn/moving_variance")

                # Branch 1: kernel 9
                w1 = reader.get_tensor(f"MTENet6Narrow/block{b}/depthwise_conv{l}_1/depthwise_weights") # (9, 1, 48, 1)
                g1 = reader.get_tensor(f"MTENet6Narrow/block{b}/depthwise_conv{l}_1_bn/gamma")
                b1 = reader.get_tensor(f"MTENet6Narrow/block{b}/depthwise_conv{l}_1_bn/beta")
                m1 = reader.get_tensor(f"MTENet6Narrow/block{b}/depthwise_conv{l}_1_bn/moving_mean")
                v1 = reader.get_tensor(f"MTENet6Narrow/block{b}/depthwise_conv{l}_1_bn/moving_variance")

                # Pad branch 0: (3, 1, 48, 1) -> (9, 1, 48, 1) by padding 3 zeros top and 3 bottom
                w0_pad = np.pad(w0, ((3, 3), (0, 0), (0, 0), (0, 0)), mode="constant")

                # Scale by gamma / sqrt(var + eps)
                scale0 = g0 / np.sqrt(v0 + eps)
                scale1 = g1 / np.sqrt(v1 + eps)

                fused_w += w0_pad * scale0.reshape(1, 1, 48, 1)
                fused_w += w1 * scale1.reshape(1, 1, 48, 1)

                fused_b += (b0 - m0 * scale0)
                fused_b += (b1 - m1 * scale1)

                assign_ops.append(t_vars[t_w_name].assign(fused_w))
                assign_ops.append(t_vars[t_beta_name].assign(fused_b))
                assign_ops.append(t_vars[t_gamma_name].assign(np.ones((48,), dtype=np.float32)))
                assign_ops.append(t_vars[t_mean_name].assign(np.zeros((48,), dtype=np.float32)))
                assign_ops.append(t_vars[t_var_name].assign(np.ones((48,), dtype=np.float32) - eps))

        sess.run(assign_ops)

        # Save fused checkpoint
        fused_dir = "save/MTENet_Fusion/seed_42/fused_tenet"
        os.makedirs(fused_dir, exist_ok=True)
        fused_prefix = os.path.join(fused_dir, "TENet6NarrowIRDTModel-fused")
        saver_fused = tf.compat.v1.train.Saver(var_list=[v for v in tf.compat.v1.global_variables() if v.name.startswith("TENet6Narrow")])
        saver_fused.save(sess, fused_prefix)

        # Evaluate validation set
        dataset.init_iterator()
        preds = []
        true_labels = []
        num_batches = (dataset.num_samples - 1) // args_t.batch_size + 1
        for _ in range(num_batches):
            try:
                logits_v, labels_v = sess.run([model_t.logits, labels])
                preds.extend(np.argmax(logits_v, axis=1))
                true_labels.extend(labels_v)
            except tf.errors.OutOfRangeError:
                break
        sess.close()

    acc = np.mean(np.array(preds) == np.array(true_labels))
    print(f"Fused Model Validation Accuracy: {acc*100:.2f}% (Total Samples: {len(true_labels)})")
    print(f"Fused Checkpoint Saved: {fused_prefix}")

if __name__ == "__main__":
    test_fusion()
