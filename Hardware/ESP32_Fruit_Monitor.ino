#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DHT.h>
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <BluetoothSerial.h>

#include "Config.h"

// Declare external analysis function
void analyzeFruitData(SensorData &data);

// Wi-Fi Credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ESP32-CAM Endpoint URL (Fetches image from CAM module)
const char* cam_ip = "192.168.1.50"; 

DHT dht(DHTPIN, DHTTYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);
BluetoothSerial SerialBT;
WebServer server(80);

SensorData currentData;

// Buffer to store the last downloaded image frame in bytes
uint8_t* lastImageBuf = NULL;
size_t lastImageLen = 0;

// Verdict pushed from the Python app (POST /verdict). Shown on the LCD for
// VERDICT_HOLD_MS then the normal T/H/ripeness display resumes.
char verdictLine1[17] = "";
char verdictLine2[17] = "";
unsigned long lastVerdictTime = 0;
const unsigned long VERDICT_HOLD_MS = 3000;

// Web Page HTML Endpoint
void handleRoot() {
  String html = "<!DOCTYPE html><html><head><title>Fruit Monitor Dashboard</title>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<style>body{font-family:Arial;margin:20px;background:#f4f4f9;} h2{color:#333;} .card{background:#fff;padding:15px;margin-bottom:15px;border-radius:8px;box-shadow:0 2px 5px rgba(0,0,0,0.1);}</style>";
  html += "</head><body><h2>Fruit Ripeness Monitor Dashboard</h2>";
  
  html += "<div class='card'>";
  html += "<p><b>Temperature:</b> " + String(currentData.temperature, 1) + " C</p>";
  html += "<p><b>Humidity:</b> " + String(currentData.humidity, 1) + " %</p>";
  html += "<p><b>Ripeness Score:</b> " + String(currentData.ripeness_scale, 2) + "</p>";
  html += "<p><b>Estimated Time to Ripen:</b> " + currentData.time_to_ripen + "</p>";
  html += "<p><b>Storage Condition:</b> " + currentData.storage_condition + "</p>";
  html += "</div>";

  html += "<div class='card'><h3>Gas Status</h3>";
  html += "<p>Ethylene: " + String(currentData.is_ethylene_high ? "HIGH" : "LOW") + "</p>";
  html += "<p>VOC: " + String(currentData.is_voc_high ? "HIGH" : "LOW") + "</p>";
  html += "<p>Alcohol & Ethanol: " + String(currentData.is_alcohol_high ? "HIGH" : "LOW") + "</p>";
  html += "<p>Aldehydes: " + String(currentData.is_aldehydes_high ? "HIGH" : "LOW") + "</p>";
  html += "</div>";

  html += "<a href='/download_csv'><button style='padding:10px 15px;margin-right:10px;'>Download CSV Data</button></a>";
  html += "<a href='/download_jpg'><button style='padding:10px 15px;'>Download Captured Image</button></a>";
  html += "</body></html>";

  server.send(200, "text/html", html);
}

// Download Telemetry as CSV File Endpoint
void handleDownloadCSV() {
  String csvData = "Temperature(C),Humidity(%),EthyleneStatus,VOCStatus,AlcoholStatus,AldehydesStatus,RipenessScale,TimeToRipen,StorageCondition,EthylenePPM,VOCPPM,AlcoholEthanolPPM,AldehydesPPM\n";
  csvData += String(currentData.temperature, 1) + ",";
  csvData += String(currentData.humidity, 1) + ",";
  csvData += String(currentData.is_ethylene_high ? "HIGH" : "LOW") + ",";
  csvData += String(currentData.is_voc_high ? "HIGH" : "LOW") + ",";
  csvData += String(currentData.is_alcohol_high ? "HIGH" : "LOW") + ",";
  csvData += String(currentData.is_aldehydes_high ? "HIGH" : "LOW") + ",";
  csvData += String(currentData.ripeness_scale, 2) + ",";
  csvData += "\"" + currentData.time_to_ripen + "\",";
  csvData += "\"" + currentData.storage_condition + "\",";
  // Raw gas concentrations (PPM) — let the software side calibrate in-repo.
  csvData += String(currentData.ethylene, 1) + ",";
  csvData += String(currentData.voc, 1) + ",";
  csvData += String(currentData.alcohol_ethanol, 1) + ",";
  csvData += String(currentData.aldehydes, 1) + "\n";

  server.sendHeader("Content-Disposition", "attachment; filename=fruit_data.csv");
  server.send(200, "text/csv", csvData);
}

// POST /verdict — receive the two LCD lines from the Python app.
// The client sends application/x-www-form-urlencoded fields `line1`/`line2`
// (the WebServer.h `arg()` contract — no JSON library needed on-device).
// Flash on the LCD for VERDICT_HOLD_MS; each line must be ≤16 chars (16×2).
void handleVerdict() {
  String l1 = server.hasArg("line1") ? server.arg("line1") : "";
  String l2 = server.hasArg("line2") ? server.arg("line2") : "";
  if (l1.length() > 16 || l2.length() > 16) {
    server.send(400, "text/plain", "line too long");
    return;
  }
  strncpy(verdictLine1, l1.c_str(), sizeof(verdictLine1) - 1);
  verdictLine1[sizeof(verdictLine1) - 1] = 0;
  strncpy(verdictLine2, l2.c_str(), sizeof(verdictLine2) - 1);
  verdictLine2[sizeof(verdictLine2) - 1] = 0;
  lastVerdictTime = millis();

  server.send(200, "text/plain", "ok");
}

// Download Captured JPG Image Endpoint
void handleDownloadJPG() {
  if (lastImageBuf != NULL && lastImageLen > 0) {
    server.sendHeader("Content-Disposition", "attachment; filename=fruit_sample.jpg");
    server.send_P(200, "image/jpeg", (const char*)lastImageBuf, lastImageLen);
  } else {
    server.send(404, "text/plain", "No image captured yet");
  }
}

// Download image over HTTP from ESP32-CAM
void fetchCameraImage() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(String("http://") + cam_ip + "/capture");
    int httpCode = http.GET();

    if (httpCode == HTTP_CODE_OK) {
      int len = http.getSize();
      if (len > 0) {
        if (lastImageBuf != NULL) free(lastImageBuf);
        lastImageBuf = (uint8_t*)malloc(len);
        if (lastImageBuf) {
          WiFiClient * stream = http.getStreamPtr();
          stream->readBytes(lastImageBuf, len);
          lastImageLen = len;
        }
      }
    }
    http.end();
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(RESET_BUTTON_PIN, INPUT_PULLUP);

  dht.begin();
  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();

  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi");

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  SerialBT.begin("ESP32_Fruit_Monitor");

  Serial.println();
  Serial.println("ESP32 Fruit Monitor — point the app at:");
  Serial.print("http://");
  Serial.println(WiFi.localIP());

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Calibrating Gas");

  // Calibrate initial baselines from MQ135 sensor over 2 seconds
  int rawBase = analogRead(MQ135_PIN);
  float vBase = (rawBase / 4095.0) * 3.3;
  currentData.base_voc = vBase * 200.0;
  currentData.base_ethylene = vBase * 50.0;
  currentData.base_alcohol_ethanol = vBase * 100.0;
  currentData.base_aldehydes = vBase * 30.0;
  delay(2000);

  // Setup Web Server Endpoints
  server.on("/", handleRoot);
  server.on("/download_csv", handleDownloadCSV);
  server.on("/download_jpg", handleDownloadJPG);
  server.on("/verdict", HTTP_POST, handleVerdict);
  server.begin();

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Ready");
  delay(1500);
  lcd.clear();
}

unsigned long lastSenseTime = 0;

void loop() {
  server.handleClient();

  // Reset Button Check
  if (digitalRead(RESET_BUTTON_PIN) == LOW) {
    delay(50);
    if (digitalRead(RESET_BUTTON_PIN) == LOW) {
      lcd.clear();
      lcd.print("Rebooting...");
      delay(1500);
      ESP.restart();
    }
  }

  // Real-time measurement loop running every 1 second
  if (millis() - lastSenseTime >= 1000) {
    lastSenseTime = millis();

    currentData.temperature = dht.readTemperature();
    currentData.humidity = dht.readHumidity();

    int rawGas = analogRead(MQ135_PIN);
    float voltage = (rawGas / 4095.0) * 3.3;

    // Measured continuous gas concentrations
    currentData.voc = voltage * 200.0;
    currentData.ethylene = voltage * 50.0;
    currentData.alcohol_ethanol = voltage * 100.0;
    currentData.aldehydes = voltage * 30.0;

    // Run Ripeness & Condition Analysis Function
    analyzeFruitData(currentData);

    // Update LCD Screen — show the pushed verdict for VERDICT_HOLD_MS,
    // otherwise the normal Temperature/Humidity/Ripeness display.
    if (verdictLine1[0] != 0 && (millis() - lastVerdictTime) < VERDICT_HOLD_MS) {
      lcd.setCursor(0, 0);
      lcd.print(verdictLine1);
      lcd.print(" ");                    // pad to clear overflow
      lcd.setCursor(0, 1);
      lcd.print(verdictLine2);
      lcd.print(" ");
    } else {
      lcd.setCursor(0, 0);
      if (isnan(currentData.temperature) || isnan(currentData.humidity)) {
        lcd.print("Sensor Err    ");
      } else {
        lcd.print("T:");
        lcd.print(currentData.temperature, 1);
        lcd.print((char)223);
        lcd.print("C H:");
        lcd.print((int)currentData.humidity);
        lcd.print("% ");
      }

      lcd.setCursor(0, 1);
      lcd.print("Ripeness:");
      if (currentData.ripeness_scale >= 0.0) lcd.print("+");
      lcd.print(currentData.ripeness_scale, 2);
      lcd.print("   ");
    }

    // Bluetooth Output Stream (Line-by-Line Format)
    if (SerialBT.hasClient()) {
      SerialBT.print("Temperature(C) -> ");
      SerialBT.println(currentData.temperature, 1);

      SerialBT.print("Humidity(%) -> ");
      SerialBT.println(currentData.humidity, 1);

      SerialBT.println("Gases:-");
      SerialBT.print("   Ethylene: ");
      SerialBT.println(currentData.is_ethylene_high ? "high" : "low");

      SerialBT.print("   VOC: ");
      SerialBT.println(currentData.is_voc_high ? "high" : "low");

      SerialBT.print("   Alcohol and Ethanol: ");
      SerialBT.println(currentData.is_alcohol_high ? "high" : "low");

      SerialBT.print("   Aldehydes: ");
      SerialBT.println(currentData.is_aldehydes_high ? "high" : "low");

      SerialBT.print("Ripeness scale: ");
      SerialBT.println(currentData.ripeness_scale, 2);

      SerialBT.print("Suitable Storage conditions: ");
      SerialBT.println(currentData.storage_condition);

      SerialBT.print("Estimated time to ripen: ");
      SerialBT.println(currentData.time_to_ripen);
      SerialBT.println();
    }
  }

  // Periodic image fetch loop (e.g., fetch photo every 15 seconds)
  static unsigned long lastCamTime = 0;
  if (millis() - lastCamTime >= 15000) {
    lastCamTime = millis();
    fetchCameraImage();
  }
}