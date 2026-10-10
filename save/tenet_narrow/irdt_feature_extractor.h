#ifndef IRDT_FEATURE_EXTRACTOR_TENET_H
#define IRDT_FEATURE_EXTRACTOR_TENET_H

#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IRDT_SAMPLE_RATE     16000
#define IRDT_DESIRED_SAMPLES 16000
#define IRDT_W               64
#define IRDT_WIN_PER_SEGM    5
#define IRDT_NUM_CHANNELS    6
#define IRDT_NUM_WINDOWS     (IRDT_DESIRED_SAMPLES / IRDT_W)           // 250
#define IRDT_NUM_SEGMENTS    (IRDT_NUM_WINDOWS / IRDT_WIN_PER_SEGM)    // 50
#define IRDT_TOTAL_FEATURES  (IRDT_NUM_SEGMENTS * IRDT_NUM_CHANNELS)   // 300

static const int g_irdt_channels[IRDT_NUM_CHANNELS] = {1, 2, 4, 8, 16, 32};

/**
 * Computes Log-iRDT features directly from 16-bit PCM audio samples.
 * Produces float32 output tensor of shape [1, 50, 6] for TENet6-Narrow.
 * Formula: log2(1.0 + (sum / 32768.0))
 * 
 * @param audio_pcm16  Pointer to 16,000 int16_t raw audio samples.
 * @param out_float    Pointer to output float array of size 300 (50 x 6).
 */
static inline void compute_irdt_features_float(const int16_t* audio_pcm16, float* out_float) {
    static uint32_t spectrum[IRDT_NUM_WINDOWS][IRDT_NUM_CHANNELS];
    
    // Clear spectrum buffer
    for (int w = 0; w < IRDT_NUM_WINDOWS; ++w) {
        for (int k = 0; k < IRDT_NUM_CHANNELS; ++k) {
            spectrum[w][k] = 0;
        }
    }

    // Step 1: Integer Laplacian across raw windows
    const int limit = IRDT_W / 4; // 16
    for (int win = 0; win < IRDT_NUM_WINDOWS; ++win) {
        const int base = win * IRDT_W;
        for (int t = limit; t < (IRDT_W - limit); ++t) {
            const int n = base + t;
            const int mid_val = (int)audio_pcm16[n];

            for (int k = 0; k < IRDT_NUM_CHANNELS; ++k) {
                const int delay = g_irdt_channels[k];
                int lo = n - delay;
                int hi = n + delay;
                if (lo < 0) lo = 0;
                if (hi >= IRDT_DESIRED_SAMPLES) hi = IRDT_DESIRED_SAMPLES - 1;

                const int lap = (int)audio_pcm16[lo] + (int)audio_pcm16[hi] - (mid_val << 1);
                spectrum[win][k] += (uint32_t)abs(lap);
            }
        }
    }

    // Step 2: Frame aggregation & log2(1 + x)
    // Segments 0 to M-2 (0..48) are aggregated across win_per_segm (5) windows
    // Segment M-1 (49) is identically zero by design
    for (int s = 0; s < IRDT_NUM_SEGMENTS; ++s) {
        for (int k = 0; k < IRDT_NUM_CHANNELS; ++k) {
            float val = 0.0f;
            if (s < IRDT_NUM_SEGMENTS - 1) {
                uint32_t sum = 0;
                const int start_win = 1 + s * IRDT_WIN_PER_SEGM;
                for (int j = 0; j < IRDT_WIN_PER_SEGM; ++j) {
                    sum += spectrum[start_win + j][k];
                }
                val = ((float)sum) / 32768.0f;
                val = log2f(1.0f + val);
            }
            out_float[s * IRDT_NUM_CHANNELS + k] = val;
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif // IRDT_FEATURE_EXTRACTOR_TENET_H
