# iRDT + TENet6-Narrow KWS System — Final Delivery

Built by integrating two official repositories with the smallest possible
diff:
- **TENet**: https://github.com/Interlagos/TENet-kws (cloned, studied, minimally modified)
- **iRDT**: https://github.com/radu-dogaru/rdt_transform_for_tiny_ml_signal_classifiers (already ported/verified in an earlier turn of this project; reused verbatim, not reimplemented)

---

## 1. Complete modified file tree

```
TENet-kws-irdt/
├── LICENSE                        UNCHANGED (authors')
├── README.md                      UNCHANGED (authors')
├── config.py                      MODIFIED  (+13 lines, additive only)
├── main.py                        MODIFIED  (+23 lines, additive only)
├── models.py                      MODIFIED  (+34 lines, additive + 1-line refactor)
├── helper.py                      UNCHANGED (authors')
├── input_data.py                  UNCHANGED (authors') **
├── tenet_fusion.py                UNCHANGED (authors', not used by our task)
├── audio_nets/
│   ├── __init__.py                UNCHANGED (authors')
│   ├── tenet.py                   UNCHANGED (authors') *** source of truth for TENet ***
│   └── tc_resnet.py               UNCHANGED (authors', not used by our task)
├── bash/                          UNCHANGED (authors', reference scripts for other archs)
├── configs/v1/                    UNCHANGED (authors', default 12-class splits, not used by our task)
├── irdt/                          NEW package (our integration code)
│   ├── __init__.py                NEW (empty)
│   ├── irdtv.py                   NEW -- pure-NumPy iRDT, reused verbatim from the
│   │                                    already-verified irdt_feature_extractor.py
│   ├── preprocess.py              NEW -- TF1.15 tf.py_func wrapper around irdtv.py,
│   │                                    drop-in replacement for models.py's preprocess_mfcc()
│   ├── labels.py                  NEW -- index permutation for our label order
│   └── dataset.py                 NEW -- MarvinAudioWrapper(AudioWrapper), overrides
│                                        ONLY get_filenames_labels() (see §Background below)
├── extract_background_segments.py NEW -- offline script: real 1s noise segments -> WAV files
├── prepare_marvin_dataset.py      NEW -- offline script: builds our train/valid/test.txt
├── verify_pipeline.py             NEW -- the 6-step verification harness
└── MODIFIED_FILES.diff            NEW -- exact unified diff of config.py/main.py/models.py
```

`**` `input_data.py` is untouched, per your explicit instruction. Its
unconditional `_silence_`-blanking behavior (see §Background below) is
routed around by a *subclass* in `irdt/dataset.py`, not by editing this file.

---

## 2. New files — complete contents

All four `irdt/*.py` files, `extract_background_segments.py`,
`prepare_marvin_dataset.py`, and `verify_pipeline.py` are included in full
in the delivered archive. Summary of what each does (full source in the
files themselves):

- **`irdt/irdtv.py`** — `irdtv_optimized()`, `irdtv_output_shape()`,
  `irdtv_batch()`. This is *exactly* the NumPy implementation already
  verified bit-exact against the iRDT authors' real numba-jitted
  `irdtv()` in an earlier turn (see that turn's `irdt_feature_extractor.py`)
  — copied in unchanged. No iRDT math was rewritten for this integration.
- **`irdt/preprocess.py`** — `preprocess_irdt(inputs, **kwargs)`. Squeezes
  the wav tensor, asserts (Python-level, at graph-build time) that the
  configured `(w, chan, win_per_segm, desired_samples)` produce exactly
  `(83, 6)`, wraps `irdtv_batch()` via `tf.compat.v1.py_func`, and restores
  static shape via `set_shape()`. This is the *only* new code that touches
  TensorFlow ops for iRDT — no reimplementation of the algorithm itself.
- **`irdt/labels.py`** — `MARVIN_LABEL_PERMUTATION = [2, 1, 0]` and
  `remap_labels()`, converting the repo's fixed internal order
  `[_silence_, _unknown_, marvin]` (indices 0,1,2) to our required
  `[Marvin, Unknown, Background]` (indices 0,1,2).
- **`irdt/dataset.py`** — `MarvinAudioWrapper(AudioWrapper)`, overriding
  only `get_filenames_labels()` to skip the unconditional file-path
  blanking for `_silence_`-labeled rows (see §Background below).
- **`extract_background_segments.py`** — offline, TF-free script that
  reads Google Speech Commands' `_background_noise_/*.wav` files and
  writes 1000 distinct, natural-volume, random 1-second crops as real
  physical WAV files to `<dataset_path>/_silence_/bg_0000.wav`...`bg_0999.wav`.
- **`prepare_marvin_dataset.py`** — offline script producing
  `configs/v1_irdt/{train,valid,test}.txt` per our exact class spec
  (1000 Marvin / 2000 Unknown / 1000 Background, 80/10/10 stratified split).
- **`verify_pipeline.py`** — the 6-step verification harness (§5 below).

---

## 3. Modified files — exact diffs

See `MODIFIED_FILES.diff` in the delivered archive for the full unified
diff. Summary:

- **`config.py`**: adds 5 new `--irdt_*` CLI flags (all default to the
  paper/authors' KWS values: `w=64, chan=[1,2,4,8,16,32], win_per_segm=3,
  biti=24, log=False`), plus one line in `update_args()` to parse
  `--irdt_chan` into a list of ints. No existing line changed.
- **`main.py`**: adds `_dataset_wrapper_for(args)` (dispatches to
  `MarvinAudioWrapper` only for `--arch TENet6NarrowIRDTModel`, otherwise
  the original `AudioWrapper`) and `_maybe_remap_labels(args, labels)`
  (applies our label permutation only for that same arch). `train()`/
  `evaluate()` each get 2 lines changed to call these instead of
  hardcoding `AudioWrapper`.
- **`models.py`**: extracts a 1-line refactor — `build()`'s
  `self.audio = preprocess_mfcc(...)` becomes `self.audio =
  self.preprocess(wavs)`, with `preprocess()` defined as a normal method
  returning `preprocess_mfcc(...)` (byte-for-byte identical behavior for
  every existing model). Adds `TENet6NarrowIRDTModel(TENet6NarrowModel)`,
  which overrides only `preprocess()` to call `irdt.preprocess.preprocess_irdt()`
  — `build_inference()` (i.e. the actual TENet graph) is inherited
  unchanged from `TENet6NarrowModel`, which calls the authors'
  `tenet.TENet6Narrow()` exactly as before. Registers the new class in
  `_available_nets`.

`audio_nets/tenet.py` (the file you specifically asked to be preserved,
including `tenet()`, `TENet6Narrow()`, `TENet_arg_scope()`) has **zero
changes** — confirmed by `git diff` producing no output for that file.

---

## 4. Background class — resolved as instructed

**What changed since last turn:** you're right that the repo's `_silence_`
mechanism doesn't match our spec. Rather than patching `input_data.py`
(shared by every `--arch`), I added a narrow, separate path:

1. **`extract_background_segments.py`** physically writes 1000 distinct
   real 1-second segments, randomly cropped from `_background_noise_`,
   at natural (unattenuated) volume, to `<dataset_path>/_silence_/bg_XXXX.wav`.
   (Random offline extraction, not per-batch cropping, because
   `decode_wav(desired_samples=16000)` always takes the *first* 16000
   samples of whatever file it's given — pointing multiple rows at the
   same long noise file would otherwise yield identical, not random, clips.)

2. **`irdt/dataset.py::MarvinAudioWrapper`** overrides *only*
   `get_filenames_labels()` from the base `AudioWrapper`, to skip this one
   line from the original:
   ```python
   df.loc[df['label'].str.startswith(SILENCE_LABEL), 'file'] = ''
   ```
   Verified directly (see transcript): with the original logic, `_silence_`
   rows' file paths become `''` regardless of what's listed; with our
   override, they resolve to the real, on-disk `bg_XXXX.wav` paths.
   `input_data.py` itself was never edited.

3. **`main.py`** now dispatches to `MarvinAudioWrapper` only when
   `--arch TENet6NarrowIRDTModel`; every other arch keeps using the
   original `AudioWrapper`, byte-for-byte as before.

**One inherited, unmodified behavior worth knowing:** `input_data.py`'s
`_parse_function()` unconditionally passes `background_data=...` to
`_mix_background()` for *every* row (Marvin/Unknown/Background alike),
which additionally overlays a second, low-volume (≤10%) noise layer with
80% probability *during training only* (never at eval time). This is the
repo's standard augmentation, applied uniformly across all three of our
classes — not something added or special-cased for Background. I left it
as-is since changing it would be an unrequested behavior change to shared
augmentation logic.

---

## 5. Exact iRDT → TENet6Narrow tensor flow

```
audio (wavs)              [batch, 16000, 1]     float32, [-1,1]  (TF decode_wav's native output)
   |  irdt.preprocess.preprocess_irdt()
   |    - squeeze          [batch, 16000]
   |    - tf.py_func -> irdt.irdtv.irdtv_batch() (per-example irdtv_optimized())
   v
iRDT features (model.audio) [batch, 83, 6]        float32
   |  audio_nets.tenet.TENet6Narrow() -- AUTHORS' CODE, UNCHANGED
   |    - tf.reshape -> [batch, 83, 1, 6]
   |    - stem: 3x1 conv -> [batch, 83, 1, 16]
   |    - block0 (stride 2,1): depthwise 9x1 -> [batch, 42, 1, 48] -> [batch, 42, 1, 48]
   |    - block1 (stride 2,1): depthwise 9x1 -> [batch, 21, 1, 48] -> [batch, 21, 1, 48]
   |    - block2 (stride 2,1): depthwise 9x1 -> [batch, 11, 1, 48] -> [batch, 11, 1, 48]
   |    - avg_pool -> dropout -> fc (1x1 conv, no bias, no BN) -> squeeze
   v
logits (model.logits)      [batch, 3]             float32  (raw logits, no softmax added)
```

`48` = expand channels = `16 (n_channels) * 3 (n_ratios)`, confirmed from
the actually-executed graph (`depthwise_conv0/depthwise -> [8, 42, 1, 48]` etc.)

---

## 6. Verification results

### 6a. What was actually executed vs. what was not

**Executed and passing** — using a TF1.15-*API* compatibility shim
(`tf_slim` + `tf.compat.v1` graph mode, with a small monkeypatch routing
around one broken legacy lazy-loader specific to this sandbox's
TF2.21/Keras3 build) running under **TF2.21 / Python 3.12**, because no
real TensorFlow 1.15 wheel exists for Python 3.12 and could not be
installed in this sandbox:

- Real graph construction of the actual, unmodified `audio_nets/tenet.py::TENet6Narrow()`
- Real graph construction of `irdt/preprocess.py` + `models.py::TENet6NarrowIRDTModel`
- Real `session.run()` execution of every step below, with real numeric outputs

**This is "tested with the original authors' code" (the actual
`audio_nets/tenet.py` source file, unmodified, actually executed) — it is
explicitly `NOT` "tested under the exact original TF1.15 environment."**
The repo's own stated requirement is `tensorflow==1.15.0`; that specific
package was never installed or run here. You should re-run
`verify_pipeline.py` in a real `tensorflow==1.15.0` environment (§7) before
treating this as fully environment-validated — the shim is a strong proxy
(same op names, same `tf.compat.v1` graph-mode semantics, same `tf_slim`
API `tf.contrib.slim` was built from) but not a substitute for the genuine
target environment.

### 6b. Shapes (from real, executed `session.run` calls)

```
audio:                 [8, 16000, 1]
iRDT:                  [8, 83, 6]
TENet internal input:  [8, 83, 1, 6]        (via tf.reshape inside tenet())
TENet stem output:     [8, 83, 1, 16]
after block0:           [8, 42, 1, 16]  (post-project, pre-block1)
after block1:            [8, 21, 1, 16]
after block2:             [8, 11, 1, 16]
global avg pool:            [8, 16]
final logits:                [8, 3]
```

Depthwise-conv intermediate shapes, read directly off the executed graph:
```
block0/depthwise_conv0/depthwise -> [8, 42, 1, 48]
block0/depthwise_conv1/depthwise -> [8, 42, 1, 48]
block1/depthwise_conv0/depthwise -> [8, 21, 1, 48]
block1/depthwise_conv1/depthwise -> [8, 21, 1, 48]
block2/depthwise_conv0/depthwise -> [8, 11, 1, 48]
block2/depthwise_conv1/depthwise -> [8, 11, 1, 48]
```
Temporal progression: **83 → 42 → 42 → 21 → 21 → 11 → 11** — matches your
expected progression exactly.

### 6c. Parameter count

From the real, executed graph (`model.model_variables`, i.e. every
variable including BN moving-mean/variance):

```
Total parameters:      15,856
Trainable parameters:  14,384
```

(For comparison: an earlier turn's from-scratch Keras reconstruction of
this same architecture — before this turn's authors'-repo-based build —
gave 15,859, a difference of exactly 3. That gap is explained: the real
`fc` layer is `slim.conv2d(..., normalizer_fn=None, ...)` with
`biases_initializer=None` inherited from the global arg_scope, i.e. **no
bias** on the final layer; the earlier Keras `Dense` layer added a
default 3-unit bias. This is a small, concrete example of why using the
actual authors' code, as you required, is more accurate than a
from-scratch reconstruction.)

### 6d. Forward-pass test — EXECUTED, PASSED

```
[1] forward pass OK, logits shape = (8, 3)
[[ 1.5741347  -0.00614589  0.55737054]
 [-0.40128642 -0.0203879   2.2982419 ]
 ... ]
```
All finite, correct shape.

### 6e. One training-step test — EXECUTED, PASSED

```
[2] loss calculation OK, total_loss = 1.570989
[3] one training step OK, loss after step = 1.420569
```
Loss measurably decreased after a single Adam step — confirms gradients
flow correctly through the full `iRDT (tf.py_func) → TENet6Narrow → loss`
graph, including back through the custom `tf.py_func` boundary (which has
no gradient of its own — the iRDT feature extractor is correctly treated
as a fixed, non-differentiable front-end; gradients start at TENet's stem
conv, which is the intended design, identical in spirit to how MFCC is a
non-differentiable front-end for the base model too).

### 6f. Checkpoint save test — EXECUTED, PASSED

```
[4] checkpoint save OK -> /tmp/.../TENet6NarrowIRDTModel-1
```
Real `tf.compat.v1.train.Saver`, real files written to disk.

### 6g. Checkpoint restore test — EXECUTED, PASSED

```
[5] checkpoint restore OK <- /tmp/.../TENet6NarrowIRDTModel-1
```
Restored into a **freshly-built graph in a new session** (not the same
session that trained it) — matches `helper.py`'s own
`routine_restore_and_initialize` pattern.

### 6h. Evaluation test — EXECUTED, PASSED

```
[6] evaluation pass OK, acc = 0.3750, logits shape = (8, 3)
```
(Accuracy is meaningless here — random untrained weights on 1 synthetic
batch — this step verifies the *mechanics*, not model quality.)

### 6i. Dataset-path verification — EXECUTED, PASSED (pandas-level, not full TF)

- `extract_background_segments.py` run against a synthetic
  `_background_noise_` directory: wrote 1000 distinct 16000-sample,
  natural-amplitude (±20000 int16 range) files; confirmed no duplicates
  among sampled files.
- `prepare_marvin_dataset.py` run against a synthetic Speech-Commands-like
  tree: produced 3200/400/400 = 4000 total, with exact 800/1600/800,
  100/200/100, 100/200/100 per-class splits, and balanced sampling across
  all 34 "Unknown" words (42–52 files/word).
- Directly compared original `AudioWrapper.get_filenames_labels()` logic
  vs. `MarvinAudioWrapper`'s: original blanks `_silence_` rows' paths to
  `''`; ours resolves them to real, on-disk file paths (verified with
  `os.path.isfile()` on the actual generated split file).

---

## 7. Exact environment requirements

Per the repo's own `README.md`:
```
tensorflow==1.15.0
pandas
```
TF1.15 requires **Python ≤ 3.7** (no wheel exists for newer Python). A
typical working setup:
```bash
conda create -n tenet-kws python=3.7
conda activate tenet-kws
pip install tensorflow==1.15.0 pandas numpy
```
Our additions require nothing beyond this — `irdt/irdtv.py` uses only
`numpy` (already a TF1.15 dependency) and Python's standard-library `math`.
`extract_background_segments.py` and `prepare_marvin_dataset.py` use only
`numpy`/`wave`/standard library, so they can even be run outside the
TF1.15 env if convenient (they do no TensorFlow work).

---

## 8. Exact commands

### 8a. Prepare the dataset
```bash
# 1. Extract real background segments (writes dataset_path/_silence_/bg_*.wav)
python extract_background_segments.py --dataset_path /path/to/speech_commands_v0.02

# 2. Build our 3-class split files (configs/v1_irdt/{train,valid,test}.txt)
python prepare_marvin_dataset.py --dataset_path /path/to/speech_commands_v0.02
```

### 8b. Train
```bash
python -m main \
  --dataset_path /path/to/speech_commands_v0.02 \
  --split_dir configs/v1_irdt \
  --arch TENet6NarrowIRDTModel \
  --num_classes 3 \
  --wanted_words marvin \
  --save_folder save/TENet6NarrowIRDTModel/
```
(`--irdt_w 64 --irdt_chan 1,2,4,8,16,32 --irdt_win_per_segm 3 --irdt_biti 24`
are all defaults already matching the paper/authors' KWS config — no need
to pass them explicitly unless you want to change the front-end.)

### 8c. Evaluate
```bash
python -m main --mod eval \
  --dataset_path /path/to/speech_commands_v0.02 \
  --split_dir configs/v1_irdt \
  --dataset_name test \
  --arch TENet6NarrowIRDTModel \
  --num_classes 3 \
  --wanted_words marvin \
  --checkpoint_path save/TENet6NarrowIRDTModel/TENet6NarrowIRDTModel-30000
```

### 8d. Restore a checkpoint (implicit in eval above; explicit standalone form)
Any run with `--checkpoint_path <prefix>` restores automatically via
`helper.py::Base.routine_restore_and_initialize()` (unchanged) — e.g. to
resume training from step 1000:
```bash
python -m main \
  --dataset_path /path/to/speech_commands_v0.02 \
  --split_dir configs/v1_irdt \
  --arch TENet6NarrowIRDTModel --num_classes 3 --wanted_words marvin \
  --save_folder save/TENet6NarrowIRDTModel/ \
  --checkpoint_path save/TENet6NarrowIRDTModel/TENet6NarrowIRDTModel-1000
```

### 8e. Run the standalone verification harness (no dataset needed)
```bash
python -m verify_pipeline
```

---

## 9. Provenance — what's from the official repos vs. our adaptation

| Component | Source |
|---|---|
| `audio_nets/tenet.py` (`tenet()`, `TENet6Narrow()`, `TENet_arg_scope()`) | **100% authors' TENet-kws repo, byte-for-byte unchanged** |
| `audio_nets/tc_resnet.py`, `input_data.py`, `helper.py`, `tenet_fusion.py`, `bash/*`, `configs/v1/*` | **100% authors' TENet-kws repo, byte-for-byte unchanged** |
| `models.py`'s `preprocess_mfcc`, `AudioNetModel`, `TENet6Model`/`TENet12Model`/etc. classes | **100% authors' TENet-kws repo, unchanged** (only 1 line refactored inside `build()`, behavior-preserving) |
| `config.py`'s existing args, `main.py`'s existing `train()`/`evaluate()` logic | **100% authors' TENet-kws repo**, additively extended |
| `irdt/irdtv.py`'s core algorithm (`irdtv_optimized`) | **100% authors' iRDT repo's `irdtv()`**, ported to pure NumPy and verified bit-exact against the real `numba`-jitted function in an earlier turn — reused verbatim here, not reimplemented |
| `irdt/preprocess.py` | **our adaptation** — a thin `tf.py_func` bridge, no new math |
| `irdt/labels.py`, `irdt/dataset.py` | **our adaptation** — dataset/label plumbing specific to our 3-class task |
| `TENet6NarrowIRDTModel` class in `models.py` | **our adaptation** — subclass wiring only; calls the authors' `tenet.TENet6Narrow()` for all actual computation |
| `extract_background_segments.py`, `prepare_marvin_dataset.py`, `verify_pipeline.py` | **entirely new**, project-specific tooling; no equivalent existed in either repo |

No new activation function, no new preprocessing step beyond swapping
MFCC→iRDT at the one documented seam, and no change to TENet's own
architecture (channels, strides, ratios, layers, kernel sizes, or the
no-final-ReLU / no-shortcut-ReLU / no-fc-bias behaviors already present in
the authors' code) were introduced anywhere in this integration.
