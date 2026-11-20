#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/usb_serial_jtag.h"
#include "esp_adc/adc_continuous.h"
#include "esp_check.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "soc/soc_caps.h"

// AD8232: OUTPUT -> GPIO3, LO- -> GPIO6, LO+ -> GPIO7, SDN -> 3V3.
#define ECG_ADC_CHANNEL ADC_CHANNEL_3
#define ECG_LO_MINUS GPIO_NUM_6
#define ECG_LO_PLUS GPIO_NUM_7
#define ECG_SAMPLE_RATE_HZ 1000
#define ECG_FRAME_SAMPLES 32
#define ECG_DMA_BYTES (ECG_FRAME_SAMPLES * SOC_ADC_DIGI_RESULT_BYTES)
#define ECG_QUEUE_FRAMES 16

typedef struct {
    uint32_t epoch;
    uint64_t first_sample;
    uint8_t leads_off;
    uint8_t count;
    uint16_t samples[ECG_FRAME_SAMPLES];
} ecg_frame_t;

static QueueHandle_t s_frames;
static uint32_t s_session;
// Single ISR writer; aligned 32-bit reads are atomic on this single-core target.
static volatile uint32_t s_adc_overflows;
static const char *TAG = "ecg";

static bool on_adc_overflow(adc_continuous_handle_t handle,
                            const adc_continuous_evt_data_t *event, void *context)
{
    (void)handle;
    (void)event;
    (void)context;
    ++s_adc_overflows;
    return false;
}

// Standard IEEE CRC-32, equivalent to Python's zlib.crc32().
static uint32_t crc32(const char *data, size_t length)
{
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < length; ++i) {
        crc ^= (uint8_t)data[i];
        for (unsigned bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320U & (0U - (crc & 1U)));
        }
    }
    return ~crc;
}

static adc_continuous_handle_t init_adc(void)
{
    adc_continuous_handle_t adc = NULL;
    adc_continuous_handle_cfg_t buffer_config = {
        .max_store_buf_size = ECG_DMA_BYTES * 32,
        .conv_frame_size = ECG_DMA_BYTES,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&buffer_config, &adc));
    adc_digi_pattern_config_t pattern = {
        .atten = ADC_ATTEN_DB_12,
        .channel = ECG_ADC_CHANNEL,
        .unit = ADC_UNIT_1,
        .bit_width = ADC_BITWIDTH_12,
    };
    adc_continuous_config_t config = {
        .sample_freq_hz = ECG_SAMPLE_RATE_HZ,
        .conv_mode = ADC_CONV_SINGLE_UNIT_1,
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
        .pattern_num = 1,
        .adc_pattern = &pattern,
    };
    ESP_ERROR_CHECK(adc_continuous_config(adc, &config));
    adc_continuous_evt_cbs_t callbacks = {.on_pool_ovf = on_adc_overflow};
    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(adc, &callbacks, NULL));
    return adc;
}

static void restart_adc(adc_continuous_handle_t adc, uint32_t *epoch,
                        uint64_t *sample_index, uint32_t *seen_overflows)
{
    ESP_ERROR_CHECK(adc_continuous_stop(adc));
    ESP_ERROR_CHECK(adc_continuous_flush_pool(adc));
    ++*epoch;
    *sample_index = 0;
    *seen_overflows = s_adc_overflows;
    // The duration of lost ADC data is unknown. Start a new time segment instead
    // of claiming that samples on either side of an overflow are contiguous.
    ESP_ERROR_CHECK(adc_continuous_start(adc));
}

static void sample_task(void *context)
{
    (void)context;
    adc_continuous_handle_t adc = init_adc();
    uint32_t dma_words[ECG_FRAME_SAMPLES];
    uint32_t epoch = 0;
    uint64_t sample_index = 0;
    uint32_t seen_overflows = 0;
    ESP_ERROR_CHECK(adc_continuous_start(adc));

    while (true) {
        uint32_t bytes_read = 0;
        esp_err_t result = adc_continuous_read(adc, (uint8_t *)dma_words,
                                               sizeof(dma_words), &bytes_read, 100);
        if (seen_overflows != s_adc_overflows) {
            restart_adc(adc, &epoch, &sample_index, &seen_overflows);
            continue;
        }
        if (result == ESP_ERR_TIMEOUT) {
            continue;
        }
        ESP_ERROR_CHECK(result);
        ecg_frame_t frame = {
            .epoch = epoch,
            .first_sample = sample_index,
            // Status is sampled once per block, not precisely at each ADC sample.
            .leads_off = gpio_get_level(ECG_LO_PLUS)
                         | (gpio_get_level(ECG_LO_MINUS) << 1),
        };
        bool valid = bytes_read > 0 && bytes_read % SOC_ADC_DIGI_RESULT_BYTES == 0;
        for (uint32_t offset = 0; valid && offset < bytes_read;
             offset += SOC_ADC_DIGI_RESULT_BYTES) {
            adc_digi_output_data_t sample;
            memcpy(&sample, (uint8_t *)dma_words + offset, sizeof(sample));
            if (sample.type2.unit != 0 || sample.type2.channel != ECG_ADC_CHANNEL) {
                valid = false;
                break;
            }
            frame.samples[frame.count++] = sample.type2.data;
        }
        if (!valid || seen_overflows != s_adc_overflows) {
            restart_adc(adc, &epoch, &sample_index, &seen_overflows);
            continue;
        }
        // Always advance, even if the transport queue is full. The receiver can
        // then detect dropped blocks from the discontinuous sample indices.
        sample_index += frame.count;
        (void)xQueueSend(s_frames, &frame, 0);
    }
}

static void transmit_task(void *context)
{
    (void)context;
    ecg_frame_t frame;
    char line[384];
    while (true) {
        if (xQueueReceive(s_frames, &frame, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        // Leading newline allows recovery if a previous timed-out USB write
        // managed to transmit only a prefix of a packet.
        line[0] = '\n';
        int length = snprintf(line + 1, sizeof(line) - 1,
                              "ECG1,%08" PRIX32 ",%" PRIu32 ",%" PRIu64 ",%u,%u,%u",
                              s_session, frame.epoch, frame.first_sample,
                              ECG_SAMPLE_RATE_HZ, frame.leads_off, frame.count);
        configASSERT(length > 0 && length < sizeof(line) - 1);
        size_t used = (size_t)length + 1;
        for (unsigned i = 0; i < frame.count; ++i) {
            length = snprintf(line + used, sizeof(line) - used, ",%u", frame.samples[i]);
            configASSERT(length > 0 && length < sizeof(line) - used);
            used += (size_t)length;
        }
        uint32_t checksum = crc32(line + 1, used - 1);
        length = snprintf(line + used, sizeof(line) - used, ",%08" PRIX32 "\n", checksum);
        configASSERT(length > 0 && length < sizeof(line) - used);
        used += (size_t)length;
        // USB congestion must never pause ADC sampling. Failed packets are not
        // retried: an incomplete line fails CRC and sample indices expose loss.
        (void)usb_serial_jtag_write_bytes(line, used, pdMS_TO_TICKS(20));
    }
}

void app_main(void)
{
    gpio_config_t leads_config = {
        .pin_bit_mask = (1ULL << ECG_LO_MINUS) | (1ULL << ECG_LO_PLUS),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&leads_config));
    usb_serial_jtag_driver_config_t usb_config = {
        .tx_buffer_size = 2048,
        .rx_buffer_size = 256,
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_config));
    s_session = esp_random();
    s_frames = xQueueCreate(ECG_QUEUE_FRAMES, sizeof(ecg_frame_t));
    ESP_ERROR_CHECK(s_frames ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_LOGI(TAG, "AD8232 GPIO3, %u Hz, USB Serial/JTAG", ECG_SAMPLE_RATE_HZ);
    ESP_ERROR_CHECK(xTaskCreate(transmit_task, "ecg_usb", 4096, NULL, 3, NULL)
                    == pdPASS ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_ERROR_CHECK(xTaskCreate(sample_task, "ecg_adc", 4096, NULL, 5, NULL)
                    == pdPASS ? ESP_OK : ESP_ERR_NO_MEM);
}
