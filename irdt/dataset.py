"""
A single, narrow subclass of input_data.py's AudioWrapper, overriding
ONLY get_filenames_labels(). input_data.py itself is NOT modified.

Why this override exists: AudioWrapper.get_filenames_labels() contains

    df.loc[df['label'].str.startswith(SILENCE_LABEL), 'file'] = ''

which unconditionally wipes the file path for any "_silence_"-labeled
split-file row, regardless of what real filename is listed. Combined with
_mix_background()'s is_silent branch (input_data.py), this makes the
repo's "silence"/background class synthetic: all-zero audio, optionally
mixed with noise at <=10% volume with 80% probability DURING TRAINING
ONLY, and exactly all-zero at evaluation time.

Our project's Background class is defined differently: 1000 REAL,
naturally-extracted 1-second segments from _background_noise_ (see
extract_background_segments.py, which physically writes these as
dataset_path/_silence_/bg_XXXX.wav -- real files, not empty placeholders).
For these rows to actually load, get_filenames_labels() must NOT blank
their file paths. Every other line of the original method is unchanged.
"""

import os

import pandas as pd

from input_data import AudioWrapper, UNKNOWN_WORD_LABEL


class MarvinAudioWrapper(AudioWrapper):
    def get_filenames_labels(self):
        df = pd.read_csv(self.split_file, header=None, names=['file'])
        df['label'] = df['file'].apply(lambda x: x.split(
            '/')[0]).apply(lambda x: x if x in self.prepare_words_list else UNKNOWN_WORD_LABEL)
        df['index_label'] = df['label'].apply(lambda x: self.word_to_index[x])
        df['file'] = df['file'].apply(
            lambda x: os.path.join(self.dataset_path, x))

        # Intentionally NOT doing:
        #   df.loc[df['label'].str.startswith(SILENCE_LABEL), 'file'] = ''
        # -- our "_silence_"-labeled rows point at real, physically-extracted
        # background segments (extract_background_segments.py) and must be
        # loaded like any other class, not synthesized. See module docstring.

        return list(df['file']), list(df['index_label'])
