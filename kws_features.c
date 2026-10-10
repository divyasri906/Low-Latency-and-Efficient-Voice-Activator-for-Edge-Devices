#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "save/dscnn_mfcc/ESP32S3_MFCC_Arduino_Package/ESP32S3_MFCC_Benchmark/mfcc_feature_extractor.h"
#include "save/dscnn_irdt/ESP32S3_KWS_Arduino_Package/ESP32S3_KWS_Benchmark/irdt_feature_extractor.h"
#include "save/tenet_narrow/ESP32S3_TENet_Benchmark/irdt_feature_extractor.h"

// Exported C API
void kws_extract_mfcc_int8(const int16_t* audio, int8_t* out) {
    compute_mfcc_features_int8(audio, out);
}

void kws_extract_irdt_int8(const int16_t* audio, int8_t* out) {
    compute_irdt_features_int8(audio, out);
}

void kws_extract_log_irdt_float(const int16_t* audio, float* out) {
    compute_irdt_features_float(audio, out);
}
