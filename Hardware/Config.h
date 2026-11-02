#ifndef CONFIG_H
#define CONFIG_H

#include <Arduino.h>

// Pin Definitions
#define DHTPIN 15
#define DHTTYPE DHT11
#define MQ135_PIN 34
#define RESET_BUTTON_PIN 4

// Gas Threshold Constants (PPM over baseline)
#define ETHYLENE_THRESHOLD 15.0
#define VOC_THRESHOLD 50.0
#define ALCOHOL_ETHANOL_THRESHOLD 20.0
#define ALDEHYDES_THRESHOLD 10.0

// Data Structure to hold sensor and analysis values
struct SensorData {
  float temperature;
  float humidity;
  
  // Real-time gas concentrations (PPM)
  float ethylene;
  float voc;
  float alcohol_ethanol;
  float aldehydes;

  // Initial Gas Baselines (PPM)
  float base_ethylene;
  float base_voc;
  float base_alcohol_ethanol;
  float base_aldehydes;

  // High/Low Flags
  bool is_ethylene_high;
  bool is_voc_high;
  bool is_alcohol_high;
  bool is_aldehydes_high;

  // Algorithmic Outputs
  float ripeness_scale;      // -1.0 to +1.0
  String storage_condition;
  String time_to_ripen;
};

// Global instance declared in main sketch
extern SensorData currentData;

#endif