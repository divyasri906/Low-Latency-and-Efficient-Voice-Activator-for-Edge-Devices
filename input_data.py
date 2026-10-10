
import tensorflow as tf
import pandas as pd
import os
import time


SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'
BACKGROUND_DIR = '_background_noise_'


# ---------------------------------------------------------
# Random number helper
# ---------------------------------------------------------
def _gen_random_from_zero(maxval, dtype=tf.float32):
    return tf.random.uniform(
        [],
        maxval=maxval,
        dtype=dtype
    )


# ---------------------------------------------------------
# Generate silence audio
# Output shape: [desired_samples, 1]
# ---------------------------------------------------------
def _gen_empty_audio(desired_samples):
    return tf.zeros(
        [desired_samples, 1],
        dtype=tf.float32
    )


# ---------------------------------------------------------
# Convert background audio to fixed size
#
# IMPORTANT:
# Background WAV files have different lengths.
# Before tf.stack(), every file MUST have the same shape.
# ---------------------------------------------------------
def _fix_background_shape(wav, desired_samples):

    current_length = tf.shape(wav)[0]

    def crop_audio():
        # Randomly select desired_samples from a long
        # background audio file.
        cropped = tf.image.random_crop(
            wav,
            [desired_samples, 1]
        )
        return cropped

    def pad_audio():
        # If background audio is shorter than desired_samples,
        # add zeros at the end.
        padding_length = desired_samples - current_length

        padded = tf.pad(
            wav,
            [
                [0, padding_length],
                [0, 0]
            ],
            mode="CONSTANT"
        )

        return padded

    fixed_wav = tf.cond(
        tf.greater(current_length, desired_samples),
        true_fn=crop_audio,
        false_fn=pad_audio
    )

    # Tell TensorFlow the final static shape.
    fixed_wav.set_shape([desired_samples, 1])

    return fixed_wav


# ---------------------------------------------------------
# Mix background noise with foreground audio
# ---------------------------------------------------------
def _mix_background(
        audio,
        desired_samples,
        background_data,
        is_silent,
        is_training,
        background_frequency,
        background_max_volume,
        **kwargs
):

    # -----------------------------------------------------
    # Foreground audio
    # -----------------------------------------------------
    foreground_wav = tf.cond(
        is_silent,
        true_fn=lambda: _gen_empty_audio(desired_samples),
        false_fn=lambda: tf.identity(audio)
    )

    foreground_wav.set_shape([desired_samples, 1])

    # -----------------------------------------------------
    # IMPORTANT FIX
    #
    # Background files can have different lengths.
    # Normalize EVERY file before tf.stack().
    # -----------------------------------------------------
    fixed_background_data = [
        _fix_background_shape(
            wav,
            desired_samples
        )
        for wav in background_data
    ]

    # Now every tensor has shape:
    #
    # [desired_samples, 1]
    #
    # Therefore stack() is safe.
    background_stack = tf.stack(
        fixed_background_data,
        axis=0
    )

    # Random background index
    random_background_data_idx = _gen_random_from_zero(
        len(fixed_background_data),
        dtype=tf.int32
    )

    # Select one background file
    background_wav = tf.gather(
        background_stack,
        random_background_data_idx
    )

    background_wav.set_shape([desired_samples, 1])

    # -----------------------------------------------------
    # Background volume
    # -----------------------------------------------------
    if is_training:

        background_volume = tf.cond(
            tf.less(
                _gen_random_from_zero(1.0),
                background_frequency
            ),
            true_fn=lambda: _gen_random_from_zero(
                background_max_volume
            ),
            false_fn=lambda: tf.constant(0.0, dtype=tf.float32)
        )

    else:
        background_volume = tf.constant(
            0.0,
            dtype=tf.float32
        )

    # -----------------------------------------------------
    # Apply background volume
    # -----------------------------------------------------
    background_wav = tf.multiply(
        background_wav,
        background_volume
    )

    # -----------------------------------------------------
    # Add foreground + background
    # -----------------------------------------------------
    background_added = tf.add(
        background_wav,
        foreground_wav
    )

    # -----------------------------------------------------
    # Keep audio between -1 and +1
    # -----------------------------------------------------
    augmented_audio = tf.clip_by_value(
        background_added,
        -1.0,
        1.0
    )

    augmented_audio.set_shape(
        [desired_samples, 1]
    )

    return augmented_audio


# ---------------------------------------------------------
# Shift audio
# ---------------------------------------------------------
def _shift_audio(
        audio,
        desired_samples,
        shift_ratio
):

    time_shift = int(
        desired_samples * shift_ratio
    )

    time_shift_amount = tf.random.uniform(
        [],
        minval=-time_shift,
        maxval=time_shift,
        dtype=tf.int32
    )

    time_shift_abs = tf.abs(
        time_shift_amount
    )

    # Positive shift
    def _pos_padding():
        return [
            [time_shift_amount, 0],
            [0, 0]
        ]

    def _pos_offset():
        return [0, 0]

    # Negative shift
    def _neg_padding():
        return [
            [0, time_shift_abs],
            [0, 0]
        ]

    def _neg_offset():
        return [
            time_shift_abs,
            0
        ]

    padded_audio = tf.pad(
        audio,
        tf.cond(
            tf.greater_equal(
                time_shift_amount,
                0
            ),
            true_fn=_pos_padding,
            false_fn=_neg_padding
        ),
        mode="CONSTANT"
    )

    sliced_audio = tf.slice(
        padded_audio,
        tf.cond(
            tf.greater_equal(
                time_shift_amount,
                0
            ),
            true_fn=_pos_offset,
            false_fn=_neg_offset
        ),
        [desired_samples, 1]
    )

    sliced_audio.set_shape(
        [desired_samples, 1]
    )

    return sliced_audio


# ---------------------------------------------------------
# Load WAV file
# ---------------------------------------------------------
def _load_wav_file(
        filename,
        desired_samples=-1
):

    wav_decoder = tf.audio.decode_wav(
        tf.io.read_file(filename),
        desired_channels=1,
        desired_samples=desired_samples
    )

    return wav_decoder.audio


# ---------------------------------------------------------
# Normal audio preprocessing
# ---------------------------------------------------------
def anchored_slice_or_pad(
        filename,
        desired_samples,
        sample_rate,
        **kwargs
):

    is_silent = tf.equal(
        tf.strings.length(filename),
        0
    )

    audio = tf.cond(
        is_silent,
        true_fn=lambda: _gen_empty_audio(
            desired_samples
        ),
        false_fn=lambda: _load_wav_file(
            filename,
            desired_samples
        )
    )

    audio.set_shape(
        [desired_samples, 1]
    )

    # Add background noise
    if "background_data" in kwargs:

        audio = _mix_background(
            audio,
            desired_samples,
            is_silent=is_silent,
            **kwargs
        )

    return audio


# ---------------------------------------------------------
# Audio preprocessing with time shift
# ---------------------------------------------------------
def anchored_slice_or_pad_with_shift(
        filename,
        desired_samples,
        sample_rate,
        **kwargs
):

    is_silent = tf.equal(
        tf.strings.length(filename),
        0
    )

    audio = tf.cond(
        is_silent,
        true_fn=lambda: _gen_empty_audio(
            desired_samples
        ),
        false_fn=lambda: _load_wav_file(
            filename,
            desired_samples
        )
    )

    audio.set_shape(
        [desired_samples, 1]
    )

    # Time shifting for training
    audio = _shift_audio(
        audio,
        desired_samples,
        shift_ratio=0.1
    )

    # Add background noise
    if "background_data" in kwargs:

        audio = _mix_background(
            audio,
            desired_samples,
            is_silent=is_silent,
            **kwargs
        )

    return audio


# =========================================================
# AudioWrapper
# =========================================================
class AudioWrapper:

    def __init__(
            self,
            args,
            mod,
            is_training,
            session
    ):

        assert mod in [
            'train',
            'valid',
            'test'
        ]

        self.args = args
        self.mod = mod
        self.is_training = is_training
        self.session = session

        self.prepare_placeholders()
        self.prepare_dataset()
        self.init_iterator()

    # -----------------------------------------------------
    # Prepare placeholders and dataset information
    # -----------------------------------------------------
    def prepare_placeholders(self):

        # Dataset arguments
        self.dataset_path = self.args.dataset_path

        self.split_file = os.path.join(
            self.args.split_dir,
            self.mod + '.txt'
        )

        self.desired_samples = (
            self.args.desired_samples
        )

        self.sample_rate = (
            self.args.sample_rate
        )

        self.num_classes = (
            self.args.num_classes
        )

        # -------------------------------------------------
        # Words list
        # -------------------------------------------------
        self.prepare_words_list = [
            SILENCE_LABEL,
            UNKNOWN_WORD_LABEL
        ] + self.args.wanted_words

        self.word_to_index = {
            word: i
            for i, word in enumerate(
                self.prepare_words_list
            )
        }

        # -------------------------------------------------
        # Background noise configuration
        # -------------------------------------------------
        self.background_max_volume = tf.constant(
            self.args.background_max_volume,
            dtype=tf.float32
        )

        self.background_frequency = tf.constant(
            self.args.background_frequency,
            dtype=tf.float32
        )

        self.background_data = (
            self.prepare_background_data()
        )

        # -------------------------------------------------
        # Filenames and labels
        # -------------------------------------------------
        self.filenames, self.labels = (
            self.get_filenames_labels()
        )

        self.data = (
            self.filenames,
            self.labels
        )

        self.num_samples = len(
            self.filenames
        )

        # -------------------------------------------------
        # TensorFlow placeholders
        # -------------------------------------------------
        self.filenames_placeholder = (
            tf.compat.v1.placeholder(
                tf.string,
                self.num_samples
            )
        )

        self.labels_placeholder = (
            tf.compat.v1.placeholder(
                tf.int64,
                self.num_samples
            )
        )

        self.placeholders = (
            self.filenames_placeholder,
            self.labels_placeholder
        )

    # -----------------------------------------------------
    # Prepare background WAV files
    # -----------------------------------------------------
    def prepare_background_data(self):

        background_files = []

        background_path = os.path.join(
            self.dataset_path,
            BACKGROUND_DIR
        )

        filenames = os.listdir(
            background_path
        )

        for name in filenames:

            if not name.lower().endswith(
                '.wav'
            ):
                continue

            filename = os.path.join(
                background_path,
                name
            )

            wav = _load_wav_file(
                filename
            )

            background_files.append(
                wav
            )

        return background_files

    # -----------------------------------------------------
    # Prepare tf.data dataset
    # -----------------------------------------------------
    def prepare_dataset(self):

        dataset = tf.data.Dataset.from_tensor_slices(
            self.placeholders
        )

        # Shuffle filenames
        dataset = dataset.shuffle(
            self.num_samples
        )

        # Parse audio
        dataset = dataset.map(
            self._parse_function,
            num_parallel_calls=self.args.workers
        )

        # Prefetch
        dataset = dataset.prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        # Batch
        dataset = dataset.batch(
            self.args.batch_size
        )

        # Repeat training data
        if self.is_training:

            dataset = dataset.shuffle(
                buffer_size=self.args.buffer_size,
                reshuffle_each_iteration=True
            ).repeat(-1)

        self.dataset = dataset

        # TF2-compatible TF1-style iterator
        self.iterator = (
            tf.compat.v1.data.make_initializable_iterator(
                dataset
            )
        )

        self.next_elem = (
            self.iterator.get_next()
        )

    # -----------------------------------------------------
    # Initialize iterator
    # -----------------------------------------------------
    def init_iterator(self):

        self.session.run(
            self.iterator.initializer,
            feed_dict={
                placeholder: variable
                for placeholder, variable
                in zip(
                    self.placeholders,
                    self.data
                )
            }
        )

    # -----------------------------------------------------
    # Get filenames and labels
    # -----------------------------------------------------
    def get_filenames_labels(self):

        df = pd.read_csv(
            self.split_file,
            header=None,
            names=['file']
        )

        # Extract label
        df['label'] = (
            df['file']
            .apply(
                lambda x: x.split('/')[0]
            )
            .apply(
                lambda x:
                    x
                    if x in self.prepare_words_list
                    else UNKNOWN_WORD_LABEL
            )
        )

        # Convert label to integer
        df['index_label'] = (
            df['label']
            .apply(
                lambda x:
                    self.word_to_index[x]
            )
        )

        # Build complete file path
        df['file'] = (
            df['file']
            .apply(
                lambda x:
                    os.path.join(
                        self.dataset_path,
                        x
                    )
            )
        )

        # Silence samples use empty filename
        df.loc[
            df['label'].str.startswith(
                SILENCE_LABEL
            ),
            'file'
        ] = ''

        return (
            list(df['file']),
            list(df['index_label'])
        )

    # -----------------------------------------------------
    # Parse one dataset sample
    # -----------------------------------------------------
    def _parse_function(
            self,
            filename,
            label
    ):

        augmented_audio = (
            self.augment_audio(
                filename,
                self.desired_samples,
                self.sample_rate,
                background_data=self.background_data,
                is_training=self.is_training,
                background_frequency=self.background_frequency,
                background_max_volume=self.background_max_volume,
            )
        )

        return (
            augmented_audio,
            label
        )

    # -----------------------------------------------------
    # Audio augmentation
    # -----------------------------------------------------
    def augment_audio(
            self,
            filename,
            desired_samples,
            sample_rate,
            **kwargs
    ):

        if self.is_training:

            return anchored_slice_or_pad_with_shift(
                filename,
                desired_samples,
                sample_rate,
                **kwargs
            )

        return anchored_slice_or_pad(
            filename,
            desired_samples,
            sample_rate,
            **kwargs
        )

    # -----------------------------------------------------
    # Return dataset input/output operation
    # -----------------------------------------------------
    def get_input_and_output_op(self):

        return self.next_elem
