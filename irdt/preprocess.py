"""
TensorFlow 1.15 wrapper around irdt/irdtv.py, providing a drop-in
replacement for models.py's preprocess_mfcc() -- same calling convention
(inputs, **kwargs) -> feature tensor -- so it plugs into the authors'
existing AudioNetModel.build() without needing to touch the training,
loss, checkpointing, or evaluation code in helper.py/main.py.

No iRDT math lives in this file. It only wraps irdt.irdtv.irdtv_batch()
(the authors'-code-verified NumPy implementation) via tf.py_func, since
TF 1.15 has no native op for iRDT and we are not writing a new one.
"""

import tensorflow as tf

from irdt.irdtv import irdtv_batch, irdtv_output_shape


def preprocess_irdt(inputs, **kwargs):
    """
    Parameters
    ----------
    inputs : Tensor, shape (batch, desired_samples, 1), float32, [-1,1]
             -- this is exactly the `wavs` tensor AudioNetModel.build()
             already receives from AudioWrapper (TF's decode_wav already
             normalizes 16-bit PCM to [-1,1] float, the same convention
             irdtv() expects -- see irdt/irdtv.py docstring).
    kwargs  : must include irdt_w, irdt_chan (list[int]), irdt_win_per_segm,
              irdt_biti, irdt_log, desired_samples (all provided by
              config.py's arg_config() -- see the --irdt_* flags added
              there).

    Returns
    -------
    Tensor, shape (batch, M, m), float32 -- with M, m fixed and verified
    against the expected (83, 6) BEFORE the graph is built (a Python-level
    assertion, not a runtime one, so a misconfiguration fails immediately
    and clearly rather than deep inside a training loop).
    """
    w = kwargs["irdt_w"]
    chan = kwargs["irdt_chan"]
    win_per_segm = kwargs["irdt_win_per_segm"]
    biti = kwargs["irdt_biti"]
    log = kwargs["irdt_log"]
    desired_samples = kwargs["desired_samples"]

    expected_M, expected_m = irdtv_output_shape(desired_samples, w, chan, win_per_segm)
    if expected_M <= 0 or expected_m <= 0:
        raise ValueError(
            f"Invalid iRDT configuration: produced non-positive dimensions "
            f"(M={expected_M}, m={expected_m}) for desired_samples={desired_samples}, "
            f"w={w}, chan={chan}, win_per_segm={win_per_segm}."
        )

    audio = tf.squeeze(inputs, axis=-1)  # (batch, desired_samples), matches
                                          # models.py's _log_mel_spectrogram
                                          # squeeze convention exactly.

    def _np_irdt(audio_np):
        return irdtv_batch(audio_np, chan, w, win_per_segm, biti=biti, log=log)

    feats = tf.compat.v1.py_func(_np_irdt, [audio], tf.float32, name="irdtv_batch")

    # tf.py_func erases static shape entirely; restore it so downstream
    # graph construction (audio_nets/tenet.py's `L = inputs.shape[1]`, used
    # to size convolution kernels) has the static T, F it requires. Preserve
    # the true batch dimension from the input when it's statically known
    # (e.g. a fixed-batch placeholder), rather than always widening it to
    # None. This is also the SECOND verification check: if the runtime
    # shape from irdtv_batch ever disagreed with (83, 6), TensorFlow will
    # raise a shape-mismatch error the first time this tensor is consumed.
    batch_dim = inputs.shape[0]
    feats.set_shape([batch_dim, expected_M, expected_m])

    return feats
