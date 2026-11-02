#include "Config.h"

// Calculate Ripeness Scale (-1.0 to +1.0) and Predictions based on baseline differences
void analyzeFruitData(SensorData &data) {
  // Calculate gas increases relative to baseline
  float diffEthylene = data.ethylene - data.base_ethylene;
  if (diffEthylene < 0) diffEthylene = 0;

  // Map Ethylene concentration increase (0 to 50 PPM over base) to ripeness range (-1.0 to +1.0)
  float score = (diffEthylene / 50.0) * 2.0 - 1.0;
  data.ripeness_scale = constrain(score, -1.0, 1.0);

  // Determine Gas Level High/Low Flags based on defined thresholds
  data.is_ethylene_high = (diffEthylene > ETHYLENE_THRESHOLD);
  data.is_voc_high = ((data.voc - data.base_voc) > VOC_THRESHOLD);
  data.is_alcohol_high = ((data.alcohol_ethanol - data.base_alcohol_ethanol) > ALCOHOL_ETHANOL_THRESHOLD);
  data.is_aldehydes_high = ((data.aldehydes - data.base_aldehydes) > ALDEHYDES_THRESHOLD);

  // Predict Estimated Time to Ripen & Suitable Storage Conditions
  if (data.ripeness_scale < -0.3) {
    data.time_to_ripen = "3 to 5 Days";
    data.storage_condition = "Store at room temp (20-22C) with high humidity.";
  } else if (data.ripeness_scale >= -0.3 && data.ripeness_scale <= 0.3) {
    data.time_to_ripen = "0 Days (Just Ripe)";
    data.storage_condition = "Refrigerate at 4-7C to preserve freshness.";
  } else {
    data.time_to_ripen = "Overripe (Consume Immediately)";
    data.storage_condition = "Keep cool; consume or process immediately.";
  }
}