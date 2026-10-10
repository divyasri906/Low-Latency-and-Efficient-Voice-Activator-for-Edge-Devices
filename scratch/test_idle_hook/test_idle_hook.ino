#include <Arduino.h>
#include "esp_freertos_hooks.h"

static volatile uint32_t count0 = 0;
static volatile uint32_t count1 = 0;

static bool idle_hook0() {
    count0++;
    return true;
}

static bool idle_hook1() {
    count1++;
    return true;
}

void setup() {
    Serial.begin(115200);
    esp_register_freertos_idle_hook_for_cpu(idle_hook0, 0);
    esp_register_freertos_idle_hook_for_cpu(idle_hook1, 1);
}

void loop() {
    delay(1000);
    Serial.printf("Core 0 idle ticks: %u, Core 1 idle ticks: %u\n", count0, count1);
}
