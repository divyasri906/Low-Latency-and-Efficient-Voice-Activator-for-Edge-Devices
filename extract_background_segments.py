"""
Extracts N_BACKGROUND real, randomly-positioned, DISTINCT 1-second segments
from Google Speech Commands' _background_noise_/*.wav files and writes each
as its own physical 16-bit PCM WAV file, at natural (unattenuated) volume.

Why this exists (see prepare_marvin_dataset.py's earlier docstring for the
full discrepancy writeup): input_data.py's AudioWrapper.get_filenames_labels()
unconditionally blanks the file path for any "_silence_"-labeled split-file
row, forcing its synthetic all-zero + low-volume-noise-mix generation path
-- regardless of what real filename is listed. That is the repo's own
existing, tested definition of its own "silence" class, but it does NOT
match our project's definition: "1000 random 1-second segments extracted
from _background_noise_" (i.e. real, full-volume noise recordings).

Also, simply pointing multiple split-file rows at the SAME long noise file
(e.g. _background_noise_/white_noise.wav) would not give genuinely
different segments: TensorFlow's decode_wav(desired_samples=16000) always
takes the FIRST 16000 samples of whatever file it's given, not a random
crop -- so every row referencing the same file would load identical audio.
Hence: pre-extract the random crops offline, once, as distinct files.

Output: <dataset_path>/_silence_/bg_0000.wav ... bg_0999.wav
        (16-bit PCM, 16000 Hz, mono, exactly 16000 samples each)

The "_silence_" folder name is chosen deliberately: it's the exact string
SILENCE_LABEL from input_data.py, so the EXISTING label-detection logic
(`x.split('/')[0]` compared against prepare_words_list) naturally assigns
these rows the correct internal index (0, before our label permutation
remaps it to Background=2) with NO change needed to input_data.py's label
logic -- only its unconditional file-blanking needs to be bypassed, which
is done via irdt/dataset.py::MarvinAudioWrapper (a subclass, not a patch
to input_data.py itself).

Run BEFORE prepare_marvin_dataset.py:
    python extract_background_segments.py --dataset_path /path/to/dataset_v2
"""

import argparse
import glob
import os
import wave

import numpy as np

N_BACKGROUND = 1000
SEGMENT_SAMPLES = 16000
SAMPLE_RATE = 16000
BACKGROUND_DIR = "_background_noise_"
OUT_DIR = "_silence_"


def read_wav_int16_mono(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {sr} != {SAMPLE_RATE}")

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.int64)
    elif sampwidth == 4:
        # some background_noise files ship as 32-bit PCM; downscale to int16
        # range so every extracted segment is written in a consistent format
        audio32 = np.frombuffer(raw, dtype=np.int32).astype(np.int64)
        audio = (audio32 >> 16).astype(np.int64)
    else:
        raise ValueError(f"{path}: unsupported sample width {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio.astype(np.int16)


def write_wav_int16_mono(path, samples_int16):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples_int16.astype(np.int16).tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                         help="Root of Google Speech Commands v0.02")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_background", type=int, default=N_BACKGROUND)
    args = parser.parse_args()

    bg_dir = os.path.join(args.dataset_path, BACKGROUND_DIR)
    source_files = sorted(glob.glob(os.path.join(bg_dir, "*.wav")))
    if not source_files:
        raise ValueError(f"No .wav files found under {bg_dir}")

    print(f"Found {len(source_files)} source noise files under {bg_dir}:")
    sources = {}
    for f in source_files:
        audio = read_wav_int16_mono(f)
        sources[f] = audio
        print(f"  {os.path.basename(f)}: {len(audio)} samples "
              f"({len(audio) / SAMPLE_RATE:.1f} s)")
        if len(audio) < SEGMENT_SAMPLES:
            print(f"  WARNING: {f} is shorter than 1 second, will be skipped")

    usable_sources = {f: a for f, a in sources.items() if len(a) >= SEGMENT_SAMPLES}
    if not usable_sources:
        raise ValueError("No source file is at least 1 second long.")

    out_dir = os.path.join(args.dataset_path, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    files = list(usable_sources.keys())

    # Ensure distinct (start_file, start_offset) pairs so no two extracted
    # segments are byte-identical duplicates.
    seen = set()
    written = 0
    attempts = 0
    max_attempts = args.n_background * 50

    while written < args.n_background and attempts < max_attempts:
        attempts += 1
        f = files[rng.integers(0, len(files))]
        audio = usable_sources[f]
        max_start = len(audio) - SEGMENT_SAMPLES
        start = int(rng.integers(0, max_start + 1))
        key = (f, start)
        if key in seen:
            continue
        seen.add(key)

        segment = audio[start:start + SEGMENT_SAMPLES]
        out_path = os.path.join(out_dir, f"bg_{written:04d}.wav")
        write_wav_int16_mono(out_path, segment)
        written += 1

    if written < args.n_background:
        raise RuntimeError(
            f"Only managed to extract {written}/{args.n_background} distinct "
            f"segments after {attempts} attempts -- source audio may be too "
            f"short/limited for this many distinct 1-second crops."
        )

    print(f"\nWrote {written} real 1-second background segments to {out_dir}")
    print("Each is natural-volume, unattenuated audio directly extracted "
          "from _background_noise_ (no synthesis, no volume scaling).")
    print("\nNext: run prepare_marvin_dataset.py, which will reference "
          f"these files as '{OUT_DIR}/bg_XXXX.wav' rows.")


if __name__ == "__main__":
    main()
