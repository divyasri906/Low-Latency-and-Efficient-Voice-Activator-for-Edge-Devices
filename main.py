import tensorflow as tf

import config
import models
from input_data import AudioWrapper
from helper import Trainer, Evaluator
from irdt.labels import remap_labels, MARVIN_LABEL_PERMUTATION
from irdt.dataset import MarvinAudioWrapper


def _dataset_wrapper_for(args):
    # Only our 3-class iRDT model uses the real-Background dataset path
    # (irdt/dataset.py::MarvinAudioWrapper). Every other --arch keeps using
    # the original, untouched input_data.py::AudioWrapper exactly as before.
    if args.arch == "TENet6NarrowIRDTModel":
        return MarvinAudioWrapper
    return AudioWrapper


def _maybe_remap_labels(args, labels):
    # Only our 3-class iRDT model needs relabeling -- see irdt/labels.py
    # for why input_data.py's fixed [_silence_, _unknown_, wanted_words...]
    # index order doesn't match our required [Marvin, Unknown, Background].
    # Every other --arch is completely unaffected (input_data.py itself is
    # untouched, and this is a no-op unless --arch is our IRDT model).
    if args.arch == "TENet6NarrowIRDTModel":
        return remap_labels(labels, MARVIN_LABEL_PERMUTATION)
    return labels


def train(args):
    is_training = True

    if getattr(args, 'seed', None) is not None:
        import numpy as np
        tf.compat.v1.set_random_seed(args.seed)
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    session = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
    dataset = _dataset_wrapper_for(args)(args, 'train', is_training, session)
    wavs, labels = dataset.get_input_and_output_op()
    labels = _maybe_remap_labels(args, labels)

    model = models.__dict__[args.arch](args)
    model.build(wavs=wavs, labels=labels, is_training=is_training)

    trainer = Trainer(model, session, args, dataset)
    trainer.train()

def evaluate(args):
    is_training = False

    session = tf.compat.v1.Session(config=config.TF_SESSION_CONFIG)
    dataset = _dataset_wrapper_for(args)(args, args.dataset_name, is_training, session)
    wavs, labels = dataset.get_input_and_output_op()
    labels = _maybe_remap_labels(args, labels)

    model = models.__dict__[args.arch](args)
    model.build(wavs=wavs, labels=labels, is_training=is_training)

    evaluator = Evaluator(model, session, args, dataset)
    evaluator.evaluate()

if __name__ == "__main__":
    args = config.arg_config()
    if args.mod == 'train':
        train(args)
    else:
        evaluate(args)
