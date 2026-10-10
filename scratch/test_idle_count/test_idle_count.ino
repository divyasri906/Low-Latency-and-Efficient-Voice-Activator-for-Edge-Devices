#include <Arduino.h>
#include "esp_freertos_hooks.h"
#include <esp_timer.h>

static volatile uint32_t idle_count_c0 = 0;
static volatile uint32_t idle_count_c1 = 0;

static volatile int64_t idle_us_c0 = 0;
static volatile int64_t idle_us_c1 = 0;

static int64_t c0_enter = 0;
static int64_t c1_enter = 0;

// High-resolution idle hooks
static bool idle_hook_c0() {
    idle_count_c0++;
    return true;
}

static bool idle_hook_c1() {
    idle_count_c1++;
    return true;
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    esp_register_freertos_idle_hook_for_cpu(idle_hook_c0, 0);
    esp_register_freertos_idle_hook_for_cpu(idle_hook_c1, 1);
    Serial.println("Idle hooks registered.");
}

void loop() {
    static uint32_t last_c0 = 0, last_c1 = 0;
    static int64_t last_time = 0;

    int64_t now = esp_timer_get_time();
    int64_t dt = now - last_time;
    if (dt >= 1000000) { // Every 1s
        uint32_t c0 = idle_count_c0;
        uint32_t c1 = idle_count_c1;
        uint32_t dc0 = c0 - last_c0;
        uint32_t dc1 = c1 - last_c1;
        last_c0 = c0;
        last_c1 = c1;
        last_time = now;

        Serial.printf("[Status] dt=%lld ms | Core 0 idle loops: %u | Core 1 idle loops: %u\n",
                      dt / 1000, dc0, dc1);
    }
    delay(100);
}
