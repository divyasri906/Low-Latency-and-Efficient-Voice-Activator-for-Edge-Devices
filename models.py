
import tensorflow as tf
import tf_slim as slim

from audio_nets import tenet, tc_resnet
from irdt.preprocess import preprocess_irdt


# ============================================================
# Available Models
# ============================================================

_available_nets = [
    "TENet6Model",
    "TENet12Model",
    "TENet6NarrowModel",
    "TENet12NarrowModel",
    "TCResNet8Model",
    "TCResNet14Model",
    "TENet6NarrowIRDTModel",
]


# ============================================================
# Audio Feature Extraction
# ============================================================

def _log_mel_spectrogram(audio, **kwargs):
    """
    Convert raw audio into a log-Mel spectrogram.

    TensorFlow 2.x compatible implementation.

    Original TensorFlow 1.x APIs:
        tf.contrib.signal.stft
        tf.contrib.signal.linear_to_mel_weight_matrix

    TensorFlow 2.x replacements:
        tf.signal.stft
        tf.signal.linear_to_mel_weight_matrix
    """

    # --------------------------------------------------------
    # Remove single-channel dimension
    # --------------------------------------------------------
    audio = tf.squeeze(
        audio,
        axis=-1
    )

    # --------------------------------------------------------
    # Short-Time Fourier Transform (STFT)
    # --------------------------------------------------------
    stfts = tf.signal.stft(
        audio,
        frame_length=kwargs["window_size_samples"],
        frame_step=kwargs["window_stride_samples"],
    )

    # --------------------------------------------------------
    # Power spectrogram
    # --------------------------------------------------------
    spectrograms = tf.math.real(
        stfts * tf.math.conj(stfts)
    )

    # --------------------------------------------------------
    # Number of frequency bins
    # --------------------------------------------------------
    num_spectrogram_bins = spectrograms.shape[-1]

    # --------------------------------------------------------
    # Linear frequency -> Mel frequency
    # --------------------------------------------------------
    linear_to_mel_weight_matrix = (
        tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=kwargs["num_mel_bins"],
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=kwargs["sample_rate"],
            lower_edge_hertz=kwargs["lower_edge_hertz"],
            upper_edge_hertz=kwargs["upper_edge_hertz"],
        )
    )

    # --------------------------------------------------------
    # Apply Mel filter bank
    # --------------------------------------------------------
    mel_spectrograms = tf.tensordot(
        spectrograms,
        linear_to_mel_weight_matrix,
        axes=1,
    )

    # --------------------------------------------------------
    # Restore static shape
    # --------------------------------------------------------
    mel_spectrograms.set_shape(
        spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]
        )
    )

    # --------------------------------------------------------
    # Log-Mel spectrogram
    # --------------------------------------------------------
    log_offset = 1e-6

    log_mel_spectrograms = tf.math.log(
        mel_spectrograms + log_offset
    )

    return log_mel_spectrograms


def preprocess_mfcc(inputs, **kwargs):
    """
    Extract MFCC features from raw audio.
    """

    # --------------------------------------------------------
    # Generate log-Mel spectrogram
    # --------------------------------------------------------
    log_mel_spectrograms = _log_mel_spectrogram(
        inputs,
        **kwargs
    )

    # --------------------------------------------------------
    # TensorFlow 2.x MFCC extraction
    # --------------------------------------------------------
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms
    )

    # --------------------------------------------------------
    # Keep requested number of MFCC coefficients
    # --------------------------------------------------------
    mfccs = mfccs[
        ...,
        :kwargs["num_mfccs"]
    ]

    # --------------------------------------------------------
    # Add channel dimension
    # --------------------------------------------------------
    mfccs = tf.expand_dims(
        mfccs,
        axis=-1
    )

    return mfccs


# ============================================================
# Base Audio Network Model
# ============================================================

class AudioNetModel:

    def __init__(self, args):

        self.args = args

        self.weight_decay = (
            self.args.weight_decay
        )

        self.dropout_keep_prob = (
            1 - self.args.dropout
        )

        self.width_multiplier = (
            self.args.width_multiplier
        )

        self.scope_name = (
            self.args.scope_name
        )

        self.fp_penalty_weight = getattr(
            self.args, "fp_penalty_weight", 0.0
        )

    # --------------------------------------------------------
    # Feature preprocessing
    # --------------------------------------------------------

    def preprocess(self, wavs):

        return preprocess_mfcc(
            wavs,
            **vars(self.args)
        )

    # --------------------------------------------------------
    # Build complete model
    # --------------------------------------------------------

    def build(
        self,
        wavs,
        labels,
        is_training
    ):

        # Feature extraction
        self.audio = self.preprocess(
            wavs
        )

        self.labels = labels

        self.is_training = is_training

        # Network inference
        self.logits = self.build_inference(
            self.audio,
            is_training=is_training
        )

        # Variables
        (
            self.model_variables,
            self.model_l2_variables
        ) = self.get_variables(
            self.scope_name
        )

        # Loss
        (
            self.total_loss,
            self.model_loss
        ) = self.build_loss(
            self.logits,
            self.labels
        )

        # Accuracy
        self.acc = self.build_acc(
            self.logits,
            self.labels
        )

    # --------------------------------------------------------
    # Get model variables
    # --------------------------------------------------------

    def get_variables(
        self,
        scope=''
    ):

        def exclude_batch_norm(name):

            return (
                "batch_normalization" not in name
                and
                "BatchNorm" not in name
            )

        def include_model_scope(
            name,
            scope
        ):

            if scope == '':
                return True

            return (
                name
                .split(':')[0]
                .split('/')[0]
                .strip()
                == scope
            )

        # Variables used for restoring checkpoints
        model_variables = [
            v
            for v in slim.get_variables_to_restore()
            if include_model_scope(
                v.name,
                scope
            )
        ]

        # Trainable variables used for L2 regularization
        model_l2_variables = [
            v
            for v in tf.compat.v1.trainable_variables()
            if (
                exclude_batch_norm(v.name)
                and
                include_model_scope(
                    v.name,
                    scope
                )
            )
        ]

        return (
            model_variables,
            model_l2_variables
        )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    def build_loss(
        self,
        logits,
        labels
    ):

        base_loss = (
            tf.compat.v1.losses
            .sparse_softmax_cross_entropy(
                labels=labels,
                logits=logits,
                reduction=tf.compat.v1.losses.Reduction.NONE,
            )
        )

        fp_weight = getattr(self, "fp_penalty_weight", 0.0)
        if fp_weight > 0.0:
            probs = tf.nn.softmax(logits)
            p_marvin = probs[:, 0]
            is_not_marvin = tf.cast(tf.not_equal(labels, 0), tf.float32)
            fp_penalty = fp_weight * is_not_marvin * tf.square(p_marvin)
            model_loss = tf.reduce_mean(base_loss + fp_penalty)
        else:
            model_loss = tf.reduce_mean(base_loss)

        if self.model_l2_variables:

            l2_loss = (
                self.args.weight_decay
                *
                tf.add_n(
                    [
                        tf.nn.l2_loss(
                            tf.cast(
                                v,
                                tf.float32
                            )
                        )
                        for v in self.model_l2_variables
                    ]
                )
            )

        else:

            l2_loss = 0.0

        total_loss = (
            model_loss + l2_loss
        )

        return (
            total_loss,
            model_loss
        )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    def build_acc(
        self,
        logits,
        labels
    ):

        self.preds = tf.math.argmax(
            logits,
            axis=-1
        )

        acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    labels,
                    self.preds
                ),
                tf.float32
            )
        )

        return acc

    # --------------------------------------------------------
    # Network inference
    # --------------------------------------------------------

    def build_inference(
        self,
        inputs,
        is_training=True
    ):

        raise NotImplementedError


# ============================================================
# TCResNet8
# ============================================================

class TCResNet8Model(AudioNetModel):

    def __init__(self, args):

        super().__init__(args)

    def build_inference(
        self,
        inputs,
        is_training
    ):

        with slim.arg_scope(
            tc_resnet.TCResNet_arg_scope(
                is_training=is_training,
                weight_decay=self.weight_decay,
                keep_prob=self.dropout_keep_prob
            )
        ):

            logits = tc_resnet.TCResNet8(
                inputs,
                self.args.num_classes,
                width_multiplier=self.width_multiplier,
                scope=self.scope_name
            )

        return logits


# ============================================================
# TCResNet14
# ============================================================

class TCResNet14Model(AudioNetModel):

    def __init__(self, args):

        super().__init__(args)

    def build_inference(
        self,
        inputs,
        is_training
    ):

        with slim.arg_scope(
            tc_resnet.TCResNet_arg_scope(
                is_training=is_training,
                weight_decay=self.weight_decay,
                keep_prob=self.dropout_keep_prob
            )
        ):

            logits = tc_resnet.TCResNet14(
                inputs,
                self.args.num_classes,
                width_multiplier=self.width_multiplier,
                scope=self.scope_name
            )

        return logits


# ============================================================
# TENet6
# ============================================================

class TENet6Model(AudioNetModel):

    def __init__(self, args):

        super().__init__(args)

    def build_inference(
        self,
        inputs,
        is_training
    ):

        with slim.arg_scope(
            tenet.TENet_arg_scope(
                is_training=is_training,
                weight_decay=self.weight_decay,
                keep_prob=self.dropout_keep_prob
            )
        ):

            logits = tenet.TENet6(
                inputs,
                self.args.num_classes,
                kernel_list=self.args.kernel_list,
                scope=self.scope_name,
                is_training=is_training,
            )

        return logits


# ============================================================
# TENet12
# ============================================================

class TENet12Model(AudioNetModel):

    def __init__(self, args):

        super().__init__(args)

    def build_inference(
        self,
        inputs,
        is_training
    ):

        with slim.arg_scope(
            tenet.TENet_arg_scope(
                is_training=is_training,
                weight_decay=self.weight_decay,
                keep_prob=self.dropout_keep_prob
            )
        ):

            logits = tenet.TENet12(
                inputs,
                self.args.num_classes,
                kernel_list=self.args.kernel_list,
                scope=self.scope_name,
                is_training=is_training,
            )

        return logits


# ============================================================
# TENet6 Narrow
# ============================================================

class TENet6NarrowModel(AudioNetModel):

    def __init__(self, args):

        super().__init__(args)

    def build_inference(
        self,
        inputs,
        is_training
    ):

        with slim.arg_scope(
            tenet.TENet_arg_scope(
                is_training=is_training,
                weight_decay=self.weight_decay,
                keep_prob=self.dropout_keep_prob
            )
        ):

            logits = tenet.TENet6Narrow(
                inputs,
                self.args.num_classes,
                kernel_list=self.args.kernel_list,
                scope=self.scope_name,
                is_training=is_training,
                width_multiplier=getattr(self.args, "width_multiplier", 1.0),
            )

        return logits


# ============================================================
# TENet12 Narrow
# ============================================================

class TENet12NarrowModel(AudioNetModel):

    def __init__(self, args):

        super().__init__(args)

    def build_inference(
        self,
        inputs,
        is_training
    ):

        with slim.arg_scope(
            tenet.TENet_arg_scope(
                is_training=is_training,
                weight_decay=self.weight_decay,
                keep_prob=self.dropout_keep_prob
            )
        ):

            logits = tenet.TENet12Narrow(
                inputs,
                self.args.num_classes,
                kernel_list=self.args.kernel_list,
                scope=self.scope_name,
                is_training=is_training,
            )

        return logits


# ============================================================
# TENet6 Narrow + iRDT
# ============================================================

class TENet6NarrowIRDTModel(
    TENet6NarrowModel
):
    """
    TENet6Narrow using the iRDT feature extractor.

    The TENet architecture is unchanged.

    Only the preprocessing stage is replaced:
        preprocess_mfcc()
    with:
        preprocess_irdt()
    """

    def preprocess(
        self,
        wavs
    ):

        return preprocess_irdt(
            wavs,
            **vars(self.args)
        )