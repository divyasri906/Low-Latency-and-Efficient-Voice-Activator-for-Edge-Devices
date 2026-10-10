"""
Generates configs/v1_irdt/{train,valid,test}.txt in exactly the format
input_data.py::AudioWrapper already expects (word/filename.wav, one per
line, relative to --dataset_path) -- this repo ships no split-generation
code of its own (configs/v1/*.txt are pre-built), so this script is new,
not a modification of anything in the authors' repo.

Implements the class spec from our project:
    0 = Marvin           1000 samples
    1 = Unknown Speech    2000 samples (balanced across a fixed word list)
    2 = Background        1000 samples
    total                 4000 samples, 80/10/10 train/valid/test, stratified

*** Background rows now reference REAL extracted files ***
Run extract_background_segments.py FIRST. It physically writes 1000
distinct, natural-volume, 1-second segments extracted from
_background_noise_ to <dataset_path>/_silence_/bg_0000.wav ... bg_0999.wav.
This script lists those real filenames as "_silence_/bg_XXXX.wav" rows.

For these rows to actually be LOADED (rather than synthesized), training
and evaluation must use irdt.dataset.MarvinAudioWrapper instead of
input_data.AudioWrapper's default get_filenames_labels(), which
unconditionally blanks any "_silence_"-labeled row's file path. main.py
already does this dispatch automatically for --arch TENet6NarrowIRDTModel
-- see main.py::_dataset_wrapper_for(). input_data.py itself is untouched.
"""

import argparse
import os
import random

SILENCE_LABEL = "_silence_"

# From our earlier-established dataset spec: everything in "Unknown Speech"
# is drawn from this fixed 34-word list.
UNKNOWN_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "bed", "bird", "cat", "dog", "happy", "house", "sheila", "tree",
    "wow", "backward", "forward", "follow", "learn", "visual",
]

MARVIN_WORD = "marvin"
N_MARVIN = 1000
N_UNKNOWN = 2000
N_BACKGROUND = 1000
SPLIT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


# Hard negative words identified across seeds
UNIVERSAL_HARD_WORDS = ["dog", "learn", "happy"]
SEED_HARD_WORDS = [
    "on", "nine", "down", "backward", "forward",
    "seven", "off", "left", "right", "stop"
]


def list_word_files(dataset_path, word):
    word_dir = os.path.join(dataset_path, word)
    if not os.path.isdir(word_dir):
        return []
    return sorted(f for f in os.listdir(word_dir) if f.endswith(".wav"))


def sample_marvin(dataset_path, rng):
    files = list_word_files(dataset_path, MARVIN_WORD)
    if len(files) < N_MARVIN:
        raise ValueError(
            f"Only {len(files)} '{MARVIN_WORD}' files found under {dataset_path}, "
            f"need {N_MARVIN}."
        )
    chosen = rng.sample(files, N_MARVIN)
    return [f"{MARVIN_WORD}/{f}" for f in chosen]


def sample_unknown_splits(dataset_path, rng, hard_negative_oversampling=True):
    """
    Samples unknown speech with split-awareness:
    - Held-out valid (200) and test (200) are sampled uniformly across all words FIRST.
      This guarantees valid/test distributions remain natural, balanced, and unskewed.
    - Train (1600) samples are allocated using weighted quotas:
      * Universal hard words (dog, learn, happy): weight = 3 (96 files each)
      * Seed-specific hard words (10 words): weight = 2 (64 files each)
      * Remaining normal words (21 words): weight = 1 (32 files each)
      * 3*3 + 10*2 + 21*1 = 50 units -> 1600 / 50 = 32 per unit. Exact integer match.
    - Zero file overlap between train and valid/test is guaranteed.
    """
    available = {w: list_word_files(dataset_path, w) for w in UNKNOWN_WORDS}
    available = {w: fs for w, fs in available.items() if fs}
    if not available:
        raise ValueError(
            f"None of the UNKNOWN_WORDS were found under {dataset_path}."
        )

    words = list(available.keys())
    n_words = len(words)

    # 1. Held-out validation and test sets (uniform sampling across words)
    val_total = int(round(N_UNKNOWN * SPLIT_RATIOS["valid"]))   # 200
    test_total = int(round(N_UNKNOWN * SPLIT_RATIOS["test"]))  # 200
    train_total = N_UNKNOWN - val_total - test_total            # 1600

    val_chosen = []
    test_chosen = []
    train_pool = {}

    per_w_val = val_total // n_words
    rem_val = val_total % n_words
    per_w_test = test_total // n_words
    rem_test = test_total % n_words

    for i, w in enumerate(words):
        w_files = list(available[w])
        rng.shuffle(w_files)
        nv = per_w_val + (1 if i < rem_val else 0)
        nt = per_w_test + (1 if i < rem_test else 0)
        needed_heldout = nv + nt
        if len(w_files) < needed_heldout + 1:
            raise ValueError(
                f"Word '{w}' only has {len(w_files)} files; need at least {needed_heldout + 1}."
            )
        v_files = w_files[:nv]
        t_files = w_files[nv:nv + nt]
        remaining = w_files[nv + nt:]

        val_chosen.extend(f"{w}/{f}" for f in v_files)
        test_chosen.extend(f"{w}/{f}" for f in t_files)
        train_pool[w] = remaining

    assert len(val_chosen) == val_total
    assert len(test_chosen) == test_total

    # 2. Training set allocation
    train_chosen = []
    if not hard_negative_oversampling:
        # Uniform round-robin
        per_w_tr = train_total // n_words
        rem_tr = train_total % n_words
        for i, w in enumerate(words):
            req = per_w_tr + (1 if i < rem_tr else 0)
            p = train_pool[w]
            if len(p) >= req:
                picked = rng.sample(p, req)
            else:
                picked = list(p) + [rng.choice(p) for _ in range(req - len(p))]
            train_chosen.extend(f"{w}/{f}" for f in picked)
    else:
        # Weighted quota: 3x universal, 2x seed, 1x normal
        weights = {}
        for w in words:
            if w in UNIVERSAL_HARD_WORDS:
                weights[w] = 3
            elif w in SEED_HARD_WORDS:
                weights[w] = 2
            else:
                weights[w] = 1

        total_weight = sum(weights.values())
        unit = train_total // total_weight  # 1600 // 50 = 32
        quotas = {w: weights[w] * unit for w in words}
        rem = train_total - sum(quotas.values())
        for w in sorted(words, key=lambda x: -weights[x])[:rem]:
            quotas[w] += 1

        print("Hard-negative training allocation per word category:")
        print(f"  • Universal hard words (3x, {len(UNIVERSAL_HARD_WORDS)} words): {quotas[UNIVERSAL_HARD_WORDS[0]]} files each")
        print(f"  • Seed-specific hard words (2x, {len(SEED_HARD_WORDS)} words): {quotas[SEED_HARD_WORDS[0]]} files each")
        other_w = [w for w in words if w not in UNIVERSAL_HARD_WORDS and w not in SEED_HARD_WORDS]
        print(f"  • Standard words (1x, {len(other_w)} words): {quotas[other_w[0]]} files each")

        for w in words:
            req = quotas[w]
            p = train_pool[w]
            if len(p) >= req:
                picked = rng.sample(p, req)
            else:
                # With-replacement fallback confined strictly to training pool
                picked = list(p) + [rng.choice(p) for _ in range(req - len(p))]
            train_chosen.extend(f"{w}/{f}" for f in picked)

    assert len(train_chosen) == train_total

    # Verify zero file-level leakage between train and valid/test
    train_set = set(train_chosen)
    val_set = set(val_chosen)
    test_set = set(test_chosen)
    assert len(train_set & val_set) == 0, "DATASET LEAKAGE: train and valid share files!"
    assert len(train_set & test_set) == 0, "DATASET LEAKAGE: train and test share files!"
    assert len(val_set & test_set) == 0, "DATASET LEAKAGE: valid and test share files!"

    rng.shuffle(train_chosen)
    rng.shuffle(val_chosen)
    rng.shuffle(test_chosen)
    return train_chosen, val_chosen, test_chosen


def sample_background(dataset_path, rng):
    # Requires extract_background_segments.py to have already been run --
    # these are REAL files now (natural-volume 1-second noise segments),
    # and their filenames matter (unlike the repo's own placeholder
    # convention): irdt.dataset.MarvinAudioWrapper loads them directly.
    bg_dir = os.path.join(dataset_path, SILENCE_LABEL)
    files = sorted(f for f in os.listdir(bg_dir) if f.endswith(".wav")) \
        if os.path.isdir(bg_dir) else []
    if len(files) < N_BACKGROUND:
        raise ValueError(
            f"Only {len(files)} files found under {bg_dir}, need {N_BACKGROUND}. "
            f"Run extract_background_segments.py --dataset_path {dataset_path} first."
        )
    chosen = rng.sample(files, N_BACKGROUND)
    return [f"{SILENCE_LABEL}/{f}" for f in chosen]


def split_80_10_10(items, rng):
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * SPLIT_RATIOS["train"]))
    n_valid = int(round(n * SPLIT_RATIOS["valid"]))
    train = items[:n_train]
    valid = items[n_train:n_train + n_valid]
    test = items[n_train + n_valid:]
    return train, valid, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True,
                         help="Root of Google Speech Commands v0.02")
    parser.add_argument("--out_dir", default="configs/v1_irdt",
                         help="Where to write train.txt/valid.txt/test.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_hard_negative_oversampling", action="store_true", default=False,
                         help="Disable weighted sampling of hard negative words (fallback to uniform)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    marvin = sample_marvin(args.dataset_path, rng)
    u_tr, u_va, u_te = sample_unknown_splits(
        args.dataset_path,
        rng,
        hard_negative_oversampling=not args.no_hard_negative_oversampling
    )
    background = sample_background(args.dataset_path, rng)

    assert len(marvin) == N_MARVIN
    assert len(u_tr) + len(u_va) + len(u_te) == N_UNKNOWN
    assert len(background) == N_BACKGROUND

    m_tr, m_va, m_te = split_80_10_10(marvin, rng)
    b_tr, b_va, b_te = split_80_10_10(background, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, parts in [("train", (m_tr, u_tr, b_tr)),
                         ("valid", (m_va, u_va, b_va)),
                         ("test", (m_te, u_te, b_te))]:
        lines = parts[0] + parts[1] + parts[2]
        rng.shuffle(lines)
        out_path = os.path.join(args.out_dir, f"{name}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"{out_path}: {len(lines)} lines "
              f"(marvin={len(parts[0])}, unknown={len(parts[1])}, background={len(parts[2])})")

    print("\nTotal: marvin={}, unknown={}, background={}, sum={}".format(
        N_MARVIN, N_UNKNOWN, N_BACKGROUND, N_MARVIN + N_UNKNOWN + N_BACKGROUND
    ))
    print("\nRun training with:")
    print(f"  python -m main --dataset_path {args.dataset_path} --split_dir {args.out_dir} \\")
    print(f"    --arch TENet6NarrowIRDTModel --num_classes 3 --wanted_words {MARVIN_WORD} \\")
    print(f"    --save_folder save/TENet6NarrowIRDTModel/")


if __name__ == "__main__":
    main()
