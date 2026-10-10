"""
iRDTv feature extractor -- REUSED VERBATIM from the already-established and
verified implementation (irdt_feature_extractor.py, earlier in this project).

Do not modify this file's algorithm. It is a faithful port of the AUTHORS'
PUBLISHED irdtv() function from:
  https://github.com/radu-dogaru/rdt_transform_for_tiny_ml_signal_classifiers
  Keyword_Spotting/demo_iRDTv_float_tflite_en.ipynb

verified bit-exact (via numba) against the real jitted function across every
win_per_segm in Table I of the iRDT paper (2,3,4,5,6,7), A-type/B-type
channel lists, w=128, and the log=True path. See the project's
iRDT_KWS_specification.md for the full derivation, including two confirmed
paper-vs-code discrepancies (log2(1+x) not 1+log2(x); frame combination is a
SUM not an average, leaving Feat_spec's last row always exactly zero) and
the array-bounds proof for why that last-row-zero behavior is required, not
a bug.

For our fixed configuration:
  w = 64, chan = [1,2,4,8,16,32], m = 6, biti = 24, win_per_segm = 3
  N = 16000  ->  M = 83, m = 6  ->  Feat_spec.shape == (83, 6)
"""

import math
import numpy as np


def irdtv_optimized(signal, chan, w, win_per_segm, biti=24, log=False):
    """
    Vectorized NumPy port of the authors' irdtv(), bit-exact against the
    real numba-jitted function (see module docstring for verification
    provenance). This is the ONLY iRDT implementation used anywhere in this
    project -- do not reimplement iRDT elsewhere.

    Parameters
    ----------
    signal : 1-D array-like, float, normalized to roughly [-1, 1]
             (e.g. int16 PCM / 32768.0, or TensorFlow's decode_wav output,
             which is already in this exact convention).
    chan   : list[int]  -- delay list, e.g. [1,2,4,8,16,32]
    w      : int         -- raw window length
    win_per_segm : int   -- number of raw windows summed per output frame
    biti   : int          -- fixed-point bit width, default 24
    log    : bool          -- if True, apply log2(1 + Feat_spec) (the
                               formula ACTUALLY implemented by the authors'
                               code, not the paper's prose "1+log2(x)")

    Returns
    -------
    M : int
    Feat_spec : np.ndarray, shape (M, m), dtype float64
                (last row is always exactly 0.0 -- by design, see docstring)
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = signal.shape[0]
    m = len(chan)

    multi = 2 ** (biti - 1 - math.log2(w) - 2)
    sig = np.trunc(multi * signal).astype(np.int64)

    windows = N // w
    M = windows // win_per_segm
    limit = w // 4

    spectrum = np.zeros((windows, m), dtype=np.int64)
    t_idx = np.arange(limit, w - limit)
    window_bases = np.arange(windows)[:, None] * w
    n = window_bases + t_idx[None, :]

    for k, delay in enumerate(chan):
        # index clamping (NOT signal padding) -- see project spec doc for
        # the proof that this never changes a value in Feat_spec, since the
        # affected rows are always among the ones the frame-sum step drops.
        lo = np.clip(n - delay, 0, N - 1)
        hi = np.clip(n + delay, 0, N - 1)
        mid = np.clip(n, 0, N - 1)
        lap = sig[lo] + sig[hi] - 2 * sig[mid]
        spectrum[:, k] = np.abs(lap).sum(axis=1)

    Feat_spec = np.zeros((M, m), dtype=np.int64)
    if M - 1 > 0:
        usable = spectrum[1: 1 + (M - 1) * win_per_segm]
        grouped = usable.reshape(M - 1, win_per_segm, m).sum(axis=1)
        Feat_spec[: M - 1] = grouped

    Feat_spec = Feat_spec.astype(np.float64) / multi
    if log:
        Feat_spec = np.log2(1.0 + Feat_spec)

    return M, Feat_spec


def irdtv_output_shape(N, w, chan, win_per_segm):
    """(M, m) for a given signal length N and iRDTv config -- no computation."""
    m = len(chan)
    windows = N // w
    M = windows // win_per_segm
    return M, m


def irdtv_batch(audio_batch, chan, w, win_per_segm, biti=24, log=False):
    """
    Apply irdtv_optimized() independently to each row of a 2-D batch of
    waveforms. Used as the function wrapped by tf.py_func in
    irdt/preprocess.py, since TF1.15 has no native op for iRDT and we do
    NOT want a from-scratch TF reimplementation (per project instruction:
    "do not invent another iRDT implementation").

    Parameters
    ----------
    audio_batch : np.ndarray, shape (batch, N), float32/float64, [-1,1]

    Returns
    -------
    np.ndarray, shape (batch, M, m), dtype float32
    """
    audio_batch = np.asarray(audio_batch)
    batch = audio_batch.shape[0]
    M, m = irdtv_output_shape(audio_batch.shape[1], w, chan, win_per_segm)

    out = np.zeros((batch, M, m), dtype=np.float32)
    for i in range(batch):
        M_i, feat = irdtv_optimized(audio_batch[i], chan, w, win_per_segm,
                                     biti=biti, log=log)
        assert M_i == M, (
            f"irdtv_batch: inconsistent M within a batch ({M_i} != {M}) -- "
            "all inputs in a batch must have the same length."
        )
        out[i] = feat.astype(np.float32)

    return out
