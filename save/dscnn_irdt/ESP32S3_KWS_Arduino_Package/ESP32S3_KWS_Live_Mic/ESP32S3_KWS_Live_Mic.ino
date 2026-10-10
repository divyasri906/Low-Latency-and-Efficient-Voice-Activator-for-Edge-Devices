/**
 * ESP32-S3 Keyword Spotting (KWS) with DS-CNN & iRDT Features.
 * 
 * Hardware Requirements:
 * - ESP32-S3 Development Board (e.g. ESP32-S3-DevKitC-1)
 * - I2S Digital Microphone (e.g. INMP441, SPH0645, or MSM261S)
 * 
 * Dependencies (ESP-IDF or Arduino IDE):
 * - TensorFlowLite_ESP32 or tflite-micro
 * - driver/i2s.h (ESP-IDF) or ESP32 I2S library
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "dscnn_marvin_model_data.h"
#include "irdt_feature_extractor.h"

// ============================================================
// I2S Microphone Pin Definitions (Adjust for your board)
// ============================================================
#define I2S_PORT         I2S_NUM_0
#define I2S_PIN_BCLK     4    // Bit Clock (SCK / BCLK)
#define I2S_PIN_WS       5    // Word Select (WS / LRCLK)
#define I2S_PIN_DIN      6    // Serial Data In (SD / DOUT)

// ============================================================
// Detection & Buffer Settings
// ============================================================
#define AUDIO_BUFFER_SAMPLES 16000     // 1.0 second @ 16 kHz
#define HOP_SAMPLES          3200      // 200 ms inference hop interval
#define DETECTION_THRESHOLD  0.50f     // Marvin confidence threshold

static int16_t g_audio_ring_buffer[AUDIO_BUFFER_SAMPLES];
static int16_t g_i2s_dma_chunk[HOP_SAMPLES];

// ============================================================
// TensorFlow Lite for Microcontrollers Memory
// ============================================================
constexpr int kTensorArenaSize = 80 * 1024; // 25 KB is plenty for DS-CNN-15k
static uint8_t tensor_arena[kTensorArenaSize];

static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input_tensor = nullptr;
static TfLiteTensor* g_output_tensor = nullptr;

// ============================================================
// Initialize I2S Peripheral
// ============================================================
bool init_i2s() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_PIN_BCLK,
        .ws_io_num = I2S_PIN_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_PIN_DIN
    };

    if (i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL) != ESP_OK) {
        Serial.println("Error: Failed to install I2S driver");
        return false;
    }
    if (i2s_set_pin(I2S_PORT, &pin_config) != ESP_OK) {
        Serial.println("Error: Failed to set I2S pins");
        return false;
    }
    i2s_zero_dma_buffer(I2S_PORT);
    return true;
}

// ============================================================
// Initialize TensorFlow Lite Micro Model
// ============================================================
bool init_tflm_model() {
    g_model = tflite::GetModel(g_dscnn_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Error: Model schema %d != supported %d\n", g_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Register only required operators to save Flash/RAM
    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMean();      // GlobalAveragePooling2D
    resolver.AddAdd();       // Residual additions
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();

    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, tensor_arena, kTensorArenaSize);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Error: AllocateTensors() failed");
        return false;
    }

    g_input_tensor = g_interpreter->input(0);
    g_output_tensor = g_interpreter->output(0);

    Serial.printf("TFLM Initialized successfully! Tensor Arena: %d KB\n", kTensorArenaSize / 1024);
    return true;
}

// ============================================================
// Setup Function
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n=======================================================");
    Serial.println("  ESP32-S3 TinyML KWS: DS-CNN with iRDT Features");
    Serial.println("=======================================================");

    if (!init_i2s()) {
        Serial.println("Failed to initialize I2S microphone!");
        while (1) delay(100);
    }
    Serial.println("✓ I2S INMP441 Microphone initialized (16 kHz, 16-bit mono)");

    if (!init_tflm_model()) {
        Serial.println("Failed to initialize TFLM model!");
        while (1) delay(100);
    }
    Serial.println("✓ DS-CNN INT8 Model loaded into memory");
    Serial.println("Listening for wake word 'MARVIN'...\n");
}

// ============================================================
// Main Loop
// ============================================================
void loop() {
    size_t bytes_read = 0;

    // 1. Read fresh audio chunk (200 ms = 3200 samples)
    i2s_read(I2S_PORT, g_i2s_dma_chunk, sizeof(g_i2s_dma_chunk), &bytes_read, portMAX_DELAY);

    // 2. Slide circular buffer: shift existing audio left by HOP_SAMPLES
    memmove(g_audio_ring_buffer, 
            g_audio_ring_buffer + HOP_SAMPLES, 
            (AUDIO_BUFFER_SAMPLES - HOP_SAMPLES) * sizeof(int16_t));
    
    // Copy new chunk to end of buffer
    memcpy(g_audio_ring_buffer + (AUDIO_BUFFER_SAMPLES - HOP_SAMPLES), 
           g_i2s_dma_chunk, 
           sizeof(g_i2s_dma_chunk));

    // 3. Compute iRDT features directly into TFLite input tensor
    uint32_t t_start = millis();
    int8_t* model_input = g_input_tensor->data.int8;
    compute_irdt_features_int8(g_audio_ring_buffer, model_input);
    uint32_t t_feat = millis() - t_start;

    // 4. Run Neural Network Inference
    t_start = millis();
    TfLiteStatus invoke_status = g_interpreter->Invoke();
    uint32_t t_nn = millis() - t_start;

    if (invoke_status != kTfLiteOk) {
        Serial.println("Error: Inference invoke failed");
        return;
    }

    // 5. Decode Output Probabilities
    int8_t* model_output = g_output_tensor->data.int8;
    const float out_scale = g_output_tensor->params.scale;
    const int32_t out_zero_point = g_output_tensor->params.zero_point;

    // Dequantize class 0 (Marvin), class 1 (Unknown), class 2 (Background)
    float p_marvin = (model_output[0] - out_zero_point) * out_scale;
    float p_unknown = (model_output[1] - out_zero_point) * out_scale;
    float p_background = (model_output[2] - out_zero_point) * out_scale;

    // 6. Threshold & Action Trigger
    if (p_marvin >= DETECTION_THRESHOLD) {
        Serial.println("*******************************************************");
        Serial.printf(">> [WAKE WORD DETECTED] 'MARVIN' (Confidence: %.1f%%)\n", p_marvin * 100.0f);
        Serial.printf("   Timing: Feat: %d ms | NN: %d ms | Total: %d ms\n", t_feat, t_nn, t_feat + t_nn);
        Serial.println("   --> Triggering Audio Stream to Cloud ASR via Wi-Fi...");
        Serial.println("*******************************************************\n");

        // Optional: Call cloud streaming function here
        // stream_audio_to_cloud_asr(I2S_PORT);
    }
}
