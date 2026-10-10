"""
Generate standalone, high-performance C++ MFCC Feature Extractor for ESP32-S3.
Produces save/dscnn_mfcc/mfcc_feature_extractor.h.
"""

import os
import math
import json
import numpy as np
import librosa
import scipy.fftpack

OUTPUT_DIR = "save/dscnn_mfcc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load metadata for normalization & quantization constants
with open(os.path.join(OUTPUT_DIR, "dscnn_mfcc_metadata.json"), "r") as f:
    meta = json.load(f)

TRAIN_MEAN = meta["train_mean"]
TRAIN_STD = meta["train_std"]
IN_SCALE = meta["in_scale"]
IN_ZERO_POINT = meta["in_zero_point"]

# MFCC parameters
N_FFT = 512
HOP_LENGTH = 320
N_MELS = 128
N_MFCC = 13
SAMPLE_RATE = 16000
N_FRAMES = 49

# 1. Hann Window (periodic, fftbins=True)
hann_win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / float(N_FFT))

# 2. Mel Filterbank Basis (128, 257)
mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)

# 3. Sparse Mel Basis representation
filter_starts = []
filter_lens = []
filter_offsets = []
flat_weights = []

offset = 0
for m in range(N_MELS):
    row = mel_basis[m]
    nonzero = np.where(row > 1e-12)[0]
    if len(nonzero) == 0:
        filter_starts.append(0)
        filter_lens.append(0)
        filter_offsets.append(offset)
    else:
        start = int(nonzero[0])
        length = int(nonzero[-1] - start + 1)
        w = row[start:start + length].tolist()
        filter_starts.append(start)
        filter_lens.append(length)
        filter_offsets.append(offset)
        flat_weights.extend(w)
        offset += length

print(f"Total Mel filterbank non-zero weights: {len(flat_weights)} floats ({len(flat_weights)*4} bytes)")

# 4. DCT-II Matrix (13, 128)
dct_matrix = np.zeros((N_MFCC, N_MELS), dtype=np.float32)
for c in range(N_MFCC):
    scale = math.sqrt(1.0 / float(N_MELS)) if c == 0 else math.sqrt(2.0 / float(N_MELS))
    for m in range(N_MELS):
        dct_matrix[c, m] = scale * math.cos(math.pi * (2.0 * m + 1.0) * c / (2.0 * float(N_MELS)))

print(f"DCT matrix size: {dct_matrix.size} floats ({dct_matrix.size*4} bytes)")

# 5. Precompute FFT Bit-reversal and Twiddle factors for 512-point FFT
# Cooley-Tukey Radix-2 FFT
def bit_reverse_indices(n):
    bits = int(math.log2(n))
    indices = np.zeros(n, dtype=int)
    for i in range(n):
        rev = 0
        val = i
        for b in range(bits):
            rev = (rev << 1) | (val & 1)
            val >>= 1
        indices[i] = rev
    return indices

bit_rev = bit_reverse_indices(N_FFT)

# Twiddle factors: W_N^k = cos(-2pi k / N) + i * sin(-2pi k / N)
# Only need cos and sin for k from 0 to N/2 - 1
twiddle_cos = np.cos(-2.0 * np.pi * np.arange(N_FFT // 2) / float(N_FFT)).astype(np.float32)
twiddle_sin = np.sin(-2.0 * np.pi * np.arange(N_FFT // 2) / float(N_FFT)).astype(np.float32)

# Generate C++ Header
header_code = f"""// Auto-generated standalone MFCC Feature Extractor for ESP32-S3
// Matches librosa.feature.mfcc(y, sr=16000, n_mfcc=13, n_fft=512, hop_length=320)
#ifndef MFCC_FEATURE_EXTRACTOR_H
#define MFCC_FEATURE_EXTRACTOR_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define MFCC_SAMPLE_RATE     16000
#define MFCC_DESIRED_SAMPLES 16000
#define MFCC_N_FFT           512
#define MFCC_HOP_LENGTH      320
#define MFCC_N_MELS          128
#define MFCC_N_MFCC          13
#define MFCC_N_FRAMES        49

// Normalization & Quantization Parameters
#define MFCC_NORM_MEAN       ({TRAIN_MEAN:.8f}f)
#define MFCC_NORM_STD        ({TRAIN_STD:.8f}f)
#define MFCC_INPUT_SCALE     ({IN_SCALE:.10f}f)
#define MFCC_INPUT_ZERO_PT   ({IN_ZERO_POINT})

// 1. Hann Window (512 points)
static const float g_hann_window[512] = {{
    {', '.join(f'{x:.8f}f' for x in hann_win)}
}};

// 2. Sparse Mel Filterbank ({len(flat_weights)} non-zero weights)
static const uint16_t g_mel_filter_start[128] = {{
    {', '.join(str(x) for x in filter_starts)}
}};

static const uint16_t g_mel_filter_len[128] = {{
    {', '.join(str(x) for x in filter_lens)}
}};

static const uint16_t g_mel_filter_offset[128] = {{
    {', '.join(str(x) for x in filter_offsets)}
}};

static const float g_mel_filter_weights[{len(flat_weights)}] = {{
    {', '.join(f'{x:.8f}f' for x in flat_weights)}
}};

// 3. DCT-II Matrix (13 x 128)
static const float g_dct_matrix[13][128] = {{
"""

for c in range(N_MFCC):
    row_str = ', '.join(f'{x:.8f}f' for x in dct_matrix[c])
    header_code += f"    {{ {row_str} }},\n"

header_code += f"""}};

// 4. In-Place Radix-2 Cooley-Tukey 512-point FFT
static inline void fft_512(float* real, float* imag) {{
    // Bit reversal permutation
    static const uint16_t bit_rev[512] = {{
        {', '.join(str(x) for x in bit_rev)}
    }};

    for (int i = 0; i < 512; ++i) {{
        int j = bit_rev[i];
        if (i < j) {{
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }}
    }}

    // Cooley-Tukey Decimation-in-Time
    for (int s = 1; s <= 9; ++s) {{
        int m = 1 << s;
        int m2 = m >> 1;
        float theta = -2.0f * (float)M_PI / (float)m;
        float wpr = cosf(theta);
        float wpi = sinf(theta);

        for (int k = 0; k < 512; k += m) {{
            float wr = 1.0f;
            float wi = 0.0f;
            for (int j = 0; j < m2; ++j) {{
                int u_idx = k + j;
                int t_idx = u_idx + m2;
                float tr = wr * real[t_idx] - wi * imag[t_idx];
                float ti = wr * imag[t_idx] + wi * real[t_idx];
                real[t_idx] = real[u_idx] - tr;
                imag[t_idx] = imag[u_idx] - ti;
                real[u_idx] += tr;
                imag[u_idx] += ti;

                float next_wr = wr * wpr - wi * wpi;
                wi = wr * wpi + wi * wpr;
                wr = next_wr;
            }}
        }}
    }}
}}

/**
 * Computes 49 x 13 MFCC features directly from 16,000 int16 raw audio samples,
 * standardizes and quantizes to int8 for TFLite Micro ingestion.
 *
 * @param audio_pcm16 Pointer to 16,000 int16_t PCM audio samples
 * @param out_int8    Pointer to output buffer of size (49 * 13) int8 elements
 */
static inline void compute_mfcc_features_int8(const int16_t* audio_pcm16, int8_t* out_int8) {{
    static float real[512];
    static float imag[512];
    static float power[257];
    static float mel_energies[128];
    static float log_mel[128];

    for (int t = 0; t < MFCC_N_FRAMES; ++t) {{
        // Center-padding frame offset: audio starts at t * 320 - 256
        int frame_start = t * MFCC_HOP_LENGTH - (MFCC_N_FFT / 2);

        // Apply Hann window and load to real/imag buffer
        for (int n = 0; n < MFCC_N_FFT; ++n) {{
            int idx = frame_start + n;
            float sample = 0.0f;
            if (idx >= 0 && idx < MFCC_DESIRED_SAMPLES) {{
                sample = (float)audio_pcm16[idx] / 32768.0f;
            }}
            real[n] = sample * g_hann_window[n];
            imag[n] = 0.0f;
        }}

        // 512-point FFT
        fft_512(real, imag);

        // Power spectrum for first 257 bins
        for (int k = 0; k <= 256; ++k) {{
            power[k] = real[k] * real[k] + imag[k] * imag[k];
        }}

        // Mel Filterbank
        float max_log = -1e30f;
        for (int m = 0; m < MFCC_N_MELS; ++m) {{
            int start = g_mel_filter_start[m];
            int len = g_mel_filter_len[m];
            int off = g_mel_filter_offset[m];
            float sum = 0.0f;
            for (int j = 0; j < len; ++j) {{
                sum += power[start + j] * g_mel_filter_weights[off + j];
            }}
            mel_energies[m] = sum;
            float val = 10.0f * log10f(fmaxf(1e-10f, sum));
            log_mel[m] = val;
            if (val > max_log) {{
                max_log = val;
            }}
        }}

        // Apply top_db = 80 dB threshold (matches librosa.power_to_db)
        float thresh = max_log - 80.0f;
        for (int m = 0; m < MFCC_N_MELS; ++m) {{
            if (log_mel[m] < thresh) {{
                log_mel[m] = thresh;
            }}
        }}

        // DCT-II to obtain 13 MFCC coefficients, standardize & quantize to int8
        for (int c = 0; c < MFCC_N_MFCC; ++c) {{
            float mfcc_val = 0.0f;
            const float* dct_row = g_dct_matrix[c];
            for (int m = 0; m < MFCC_N_MELS; ++m) {{
                mfcc_val += dct_row[m] * log_mel[m];
            }}

            // Standardize
            float norm_val = (mfcc_val - MFCC_NORM_MEAN) / MFCC_NORM_STD;

            // Quantize to INT8
            int int_val = (int)roundf(norm_val / MFCC_INPUT_SCALE) + MFCC_INPUT_ZERO_PT;
            if (int_val < -128) int_val = -128;
            if (int_val > 127) int_val = 127;

            out_int8[t * MFCC_N_MFCC + c] = (int8_t)int_val;
        }}
    }}
}}

#ifdef __cplusplus
}}
#endif

#endif // MFCC_FEATURE_EXTRACTOR_H
"""

header_path = os.path.join(OUTPUT_DIR, "mfcc_feature_extractor.h")
with open(header_path, "w") as f:
    f.write(header_code)
print(f"\nGenerated C++ MFCC Feature Extractor: {header_path} ({len(header_code)} bytes)")
