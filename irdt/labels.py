"""
input_data.py's AudioWrapper always builds its label indices as:
    prepare_words_list = [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words
    word_to_index = {word: i for i, word in enumerate(prepare_words_list)}

For our task (--wanted_words marvin), that gives the FIXED internal order:
    0 = _silence_   (repo's synthetic/background class)
    1 = _unknown_
    2 = marvin

but we need:
    0 = Marvin
    1 = Unknown Speech
    2 = Background

Rather than editing input_data.py's index construction (which every other
model/arch in this repo also depends on), we apply a small index
permutation to the label tensor right after it comes out of the dataset
iterator, in main.py. input_data.py itself is left completely untouched.

permutation[original_index] = desired_index
  original 0 (_silence_) -> desired 2 (Background)
  original 1 (_unknown_) -> desired 1 (Unknown Speech)
  original 2 (marvin)    -> desired 0 (Marvin)
"""

import tensorflow as tf

# Valid ONLY when --wanted_words is set to a single word (our 3-class case).
# For the repo's default 12-class setup this module is not used at all.
MARVIN_LABEL_PERMUTATION = [2, 1, 0]

DESIRED_LABEL_NAMES = ["Marvin", "Unknown Speech", "Background"]


def remap_labels(labels, permutation):
    """
    labels : int64 Tensor, values in [0, len(permutation))
    permutation : list[int], permutation[original_index] = desired_index
    """
    table = tf.constant(permutation, dtype=tf.int64)
    return tf.gather(table, labels)
