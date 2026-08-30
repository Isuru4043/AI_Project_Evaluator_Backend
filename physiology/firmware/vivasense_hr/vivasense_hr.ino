// VivaSense heart-rate band - MAX30102 + SSD1306 OLED + ESP32 DevKit v1
//
// DISPLAY LIBRARY: Adafruit SSD1306 + Adafruit GFX.
//   U8g2 was dropped because instantiating any U8g2 SSD1306 object hung the
//   Arduino build indefinitely at "ResolveLibrary(SPI.h)" on this toolchain -
//   reproducible with hardware I2C, software I2C, and after reinstalling the
//   library, while U8g2 with no display object compiled fine. Adafruit's
//   driver builds cleanly with the same sketch, so the OLED layer was ported
//   and nothing else was touched.
//
// WHAT THIS FIRMWARE DOES, AND WHY IT IS SHAPED THIS WAY
//
// 1. It speaks BLE using the standard Heart Rate Service (0x180D).
//    Not a custom protocol: the standard characteristic already carries
//    RR-intervals and a sensor-contact bit, and any client - nRF Connect, a
//    phone HR app, the station sidecar - can read it. The band is debuggable
//    without our software.
//
// 2. It transmits INTER-BEAT INTERVALS, not the smoothed BPM.
//    This is the important one. Nervousness is read from heart-rate
//    VARIABILITY - how much successive beats differ - and the display filter
//    chain (median of 9, then exponential smoothing) exists precisely to
//    remove that variation so the number on screen sits still. Sending only
//    the smoothed value would send a number with the signal already filtered
//    out of it.
//
//    So the two paths are separated:
//      display  -> plausibility + median-deviation + median + EMA  (calm)
//      BLE      -> plausibility only                               (faithful)
//    Artifact rejection for the variability maths happens server-side, where
//    it can be tuned without reflashing and where the rejection rate is
//    recorded as a signal-quality figure.
//
// 3. Sensor contact is reported in the standard flags byte, driven by the
//    existing FINGER_THRESHOLD. The server discards windows without contact
//    rather than reading them as a calm heart.
//
// 4. Battery Service (0x180F), so the examiner learns the band is dying
//    before it dies mid-viva.
//
// The OLED keeps showing the smoothed BPM: it is the student's own reading,
// and calm is the right property for a number a person is looking at.
//
// LIBRARIES
//   Adafruit SSD1306
//   Adafruit GFX Library
//   SparkFun MAX3010x Pulse and Proximity Sensor Library
//   NimBLE-Arduino  (1.4.x - far smaller than Bluedroid; DevKit v1 flash is
//                    tight once the sensor, display and a BLE stack are in)

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "MAX30105.h"
#include "heartRate.h"

#include <NimBLEDevice.h>

// --- Display ---------------------------------------------------------------
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1      // no reset pin wired
#define OLED_ADDR     0x3C

// I2C pins, shared by the OLED (0x3C) and the MAX30102 (0x57).
#define I2C_SDA 21
#define I2C_SCL 22

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
bool displayReady = false;    // a dead screen must not stop the band streaming

// Adafruit GFX places the cursor at the TOP-LEFT of the text cell, whereas
// U8g2's drawStr placed it on the BASELINE. These are the ported positions:
// each y is the old baseline minus the font's height, so the layout lands
// where it did before.
#define ROW_TOP      12       // was baseline 20
#define ROW_BOTTOM   32       // was baseline 40
#define LABEL_TOP     4       // "BPM" label, was baseline 15
#define BIGNUM_TOP   24       // big digits, was baseline 55 with a 28px font
#define STATUS_X     96       // link indicator, bottom-right
#define STATUS_TOP   54       // was baseline 62

// Default GFX font is 6x8 px per character at size 1, scaling linearly:
// size 4 gives 24x32 px digits, the closest match to the 28px font U8g2 used.
#define TEXT_SMALL 1
#define TEXT_BIG   4

// --- Identity --------------------------------------------------------------
// Advertised name the station scans for. The suffix is filled from the MAC at
// boot so several bands can coexist in one room and each stays identifiable
// across sessions.
char deviceName[24] = "VivaSense-HR";

// --- Standard GATT UUIDs ---------------------------------------------------
#define HR_SERVICE_UUID        "180D"
#define HR_MEASUREMENT_UUID    "2A37"
#define HR_BODY_SENSOR_LOC     "2A38"
#define BATTERY_SERVICE_UUID   "180F"
#define BATTERY_LEVEL_UUID     "2A19"

// Wire a divider from the cell to this pin to report real battery level.
// Left undefined the band advertises a static value, which is honest enough
// for a bench build but tells the examiner nothing.
// #define BATTERY_ADC_PIN 34
#define BATTERY_DIVIDER_RATIO 2.0f
#define BATTERY_FULL_V        4.20f
#define BATTERY_EMPTY_V       3.30f

MAX30105 particleSensor;

NimBLECharacteristic* hrmChar = nullptr;
NimBLECharacteristic* battChar = nullptr;
bool bleConnected = false;

// --- Display filter chain --------------------------------------------------
const byte IBI_SIZE = 9;
long  ibiBuffer[IBI_SIZE];
byte  ibiSpot = 0;
byte  ibiCount = 0;
long  lastBeat = 0;

float smoothedBpm = 0;
int   displayBpm = 0;
const float EMA_ALPHA = 0.25;

// --- Transmit queue --------------------------------------------------------
// Intervals accepted on plausibility alone, waiting for the next notification.
// A BLE notification fits in the default 23-byte MTU: 2 header bytes leave
// room for 9 intervals, and at 180 bpm only 3 arrive per second.
const byte RR_QUEUE_SIZE = 8;
uint16_t rrQueue[RR_QUEUE_SIZE];   // in 1/1024 s, the unit the spec requires
byte     rrCount = 0;

// --- Limits ----------------------------------------------------------------
const long  FINGER_THRESHOLD = 50000;
const long  IBI_MIN_MS = 333;     // 180 BPM
const long  IBI_MAX_MS = 1500;    // 40 BPM
const float OUTLIER_TOLERANCE = 0.30;   // display path only
const byte  MIN_BEATS_TO_DISPLAY = 5;

unsigned long lastDisplayUpdate = 0;
unsigned long lastNotify = 0;
unsigned long lastBattery = 0;
unsigned long lastSerialUpdate = 0;
const unsigned long DISPLAY_INTERVAL_MS = 1000;
const unsigned long NOTIFY_INTERVAL_MS  = 1000;
const unsigned long BATTERY_INTERVAL_MS = 60000;
const unsigned long SERIAL_INTERVAL_MS  = 500;

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* server) {
    bleConnected = true;
  }
  void onDisconnect(NimBLEServer* server) {
    bleConnected = false;
    // Advertise again immediately: a station that reconnects mid-viva should
    // resume without anyone touching the band.
    NimBLEDevice::startAdvertising();
  }
};

// --- Small display helpers -------------------------------------------------
// One place where the GFX state (size, colour, cursor) is set, so no draw call
// can inherit a size left behind by the previous frame.
void drawText(int16_t x, int16_t y, uint8_t size, const char* text) {
  display.setTextSize(size);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(x, y);
  display.print(text);
}

void showMessage(const char* line1, const char* line2) {
  if (!displayReady) return;
  display.clearDisplay();
  drawText(0, ROW_TOP, TEXT_SMALL, line1);
  if (line2 != nullptr) drawText(0, ROW_BOTTOM, TEXT_SMALL, line2);
  display.display();
}

void resetFilter() {
  for (byte i = 0; i < IBI_SIZE; i++) ibiBuffer[i] = 0;
  ibiSpot = 0;
  ibiCount = 0;
  smoothedBpm = 0;
  displayBpm = 0;
  lastBeat = 0;
  rrCount = 0;
}

long medianIBI() {
  if (ibiCount == 0) return 0;
  long tmp[IBI_SIZE];
  for (byte i = 0; i < ibiCount; i++) tmp[i] = ibiBuffer[i];
  for (byte i = 1; i < ibiCount; i++) {
    long key = tmp[i];
    int j = i - 1;
    while (j >= 0 && tmp[j] > key) { tmp[j + 1] = tmp[j]; j--; }
    tmp[j + 1] = key;
  }
  return tmp[ibiCount / 2];
}

void queueRR(long ibiMs) {
  if (rrCount >= RR_QUEUE_SIZE) return;   // drop rather than overrun
  rrQueue[rrCount++] = (uint16_t)((ibiMs * 1024L) / 1000L);
}

uint8_t readBatteryPercent() {
#ifdef BATTERY_ADC_PIN
  int raw = analogRead(BATTERY_ADC_PIN);
  float volts = (raw / 4095.0f) * 3.3f * BATTERY_DIVIDER_RATIO;
  float pct = (volts - BATTERY_EMPTY_V) / (BATTERY_FULL_V - BATTERY_EMPTY_V);
  if (pct < 0) pct = 0;
  if (pct > 1) pct = 1;
  return (uint8_t)(pct * 100.0f);
#else
  return 100;   // no divider wired; see BATTERY_ADC_PIN above
#endif
}

// Build and send a Heart Rate Measurement per the Bluetooth SIG layout:
//   flags, heart rate, then the RR-intervals collected since the last notify.
void notifyHeartRate(bool fingerPresent) {
  if (hrmChar == nullptr || !bleConnected) { rrCount = 0; return; }

  uint8_t payload[2 + RR_QUEUE_SIZE * 2];
  uint8_t index = 0;

  // bit0 clear -> 8-bit rate (never exceeds 255 here)
  // bit1 contact detected, bit2 contact supported, bit4 RR present
  uint8_t flags = 0x04;
  if (fingerPresent) flags |= 0x02;
  if (rrCount > 0)   flags |= 0x10;

  payload[index++] = flags;
  payload[index++] = (uint8_t)(displayBpm > 255 ? 255 : displayBpm);

  for (byte i = 0; i < rrCount; i++) {
    payload[index++] = (uint8_t)(rrQueue[i] & 0xFF);
    payload[index++] = (uint8_t)(rrQueue[i] >> 8);
  }

  hrmChar->setValue(payload, index);
  hrmChar->notify();
  rrCount = 0;
}

void setupBle() {
  uint64_t mac = ESP.getEfuseMac();
  snprintf(deviceName, sizeof(deviceName), "VivaSense-HR-%04X",
           (uint16_t)(mac & 0xFFFF));

  NimBLEDevice::init(deviceName);
  NimBLEDevice::setPower(ESP_PWR_LVL_P9);   // exam room, not a wearable on the move

  NimBLEServer* server = NimBLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  NimBLEService* hrService = server->createService(HR_SERVICE_UUID);
  hrmChar = hrService->createCharacteristic(
      HR_MEASUREMENT_UUID, NIMBLE_PROPERTY::NOTIFY);

  // Body Sensor Location: 3 == finger. Advisory, but clients display it.
  NimBLECharacteristic* loc = hrService->createCharacteristic(
      HR_BODY_SENSOR_LOC, NIMBLE_PROPERTY::READ);
  uint8_t finger = 3;
  loc->setValue(&finger, 1);
  hrService->start();

  NimBLEService* battService = server->createService(BATTERY_SERVICE_UUID);
  battChar = battService->createCharacteristic(
      BATTERY_LEVEL_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
  uint8_t level = readBatteryPercent();
  battChar->setValue(&level, 1);
  battService->start();

  NimBLEAdvertising* advertising = NimBLEDevice::getAdvertising();
  advertising->addServiceUUID(HR_SERVICE_UUID);
  advertising->setScanResponse(true);
  NimBLEDevice::startAdvertising();

  Serial.print("BLE advertising as ");
  Serial.println(deviceName);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  // Bring the bus up ourselves so both devices share these pins and this
  // clock. display.begin() is told NOT to initialise Wire (last argument
  // false), because its own call takes no pin arguments and would drop the
  // bus back to the default 100 kHz.
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  displayReady = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR, false, false);
  if (!displayReady) {
    // The band's real job is streaming beats; losing the screen is a
    // degradation, not a failure, so this reports and carries on.
    Serial.println("SSD1306 not found at 0x3C - continuing without display.");
  } else {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    drawText(0, ROW_TOP, TEXT_SMALL, "Initializing...");
    display.display();
  }

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not found. Check wiring.");
    showMessage("Sensor not found", nullptr);
    while (1) delay(1000);
  }

  byte ledBrightness = 0x3F;
  byte sampleAverage = 4;
  byte ledMode       = 2;      // Red + IR
  int  sampleRate    = 400;    // detection quality depends on this staying high
  int  pulseWidth    = 411;
  int  adcRange      = 4096;

  particleSensor.setup(ledBrightness, sampleAverage, ledMode,
                       sampleRate, pulseWidth, adcRange);

  resetFilter();
  setupBle();

  showMessage("Place finger", "on sensor");
}

void loop() {
  long irValue = particleSensor.getIR();
  bool fingerPresent = (irValue >= FINGER_THRESHOLD);

  if (!fingerPresent) {
    if (ibiCount > 0) resetFilter();
  }
  else if (checkForBeat(irValue)) {
    unsigned long now = millis();

    if (lastBeat > 0) {
      long ibi = now - lastBeat;
      bool plausible = (ibi >= IBI_MIN_MS && ibi <= IBI_MAX_MS);

      if (plausible) {
        // BLE path: send it as measured. The variability between these values
        // IS the signal, so nothing that smooths is applied before transmit.
        queueRR(ibi);

        // Display path: additionally reject values far from the running
        // median, so the number on the OLED stays readable.
        bool steady = true;
        if (ibiCount >= 3) {
          long med = medianIBI();
          float deviation = fabs((float)(ibi - med)) / (float)med;
          if (deviation > OUTLIER_TOLERANCE) steady = false;
        }

        if (steady) {
          ibiBuffer[ibiSpot++] = ibi;
          ibiSpot %= IBI_SIZE;
          if (ibiCount < IBI_SIZE) ibiCount++;

          long med = medianIBI();
          if (med > 0) {
            float bpm = 60000.0 / (float)med;
            if (smoothedBpm == 0) smoothedBpm = bpm;
            else smoothedBpm = EMA_ALPHA * bpm + (1.0 - EMA_ALPHA) * smoothedBpm;
            displayBpm = (int)(smoothedBpm + 0.5);
          }
        }
      }
    }
    lastBeat = now;
  }

  // Bench telemetry over USB serial. This is the fastest way to tell whether
  // the SENSOR is working when the radio is not yet: it needs no BLE, no
  // station and no backend. `rr` is how many intervals are queued for the next
  // notification - if that stays 0 while beats are counting up, the transmit
  // path is the problem rather than the optics.
  if (millis() - lastSerialUpdate >= SERIAL_INTERVAL_MS) {
    lastSerialUpdate = millis();
    Serial.print("IR=");
    Serial.print(irValue);
    Serial.print(", medIBI=");
    Serial.print(medianIBI());
    Serial.print("ms, BPM=");
    Serial.print(displayBpm);
    Serial.print(", beats=");
    Serial.print(ibiCount);
    Serial.print(", rr=");
    Serial.print(rrCount);
    Serial.print(bleConnected ? ", BLE=up" : ", BLE=down");
    if (!fingerPresent) Serial.print("  [no finger]");
    Serial.println();
  }

  if (millis() - lastNotify >= NOTIFY_INTERVAL_MS) {
    lastNotify = millis();
    notifyHeartRate(fingerPresent);
  }

  if (millis() - lastBattery >= BATTERY_INTERVAL_MS) {
    lastBattery = millis();
    if (battChar != nullptr) {
      uint8_t level = readBatteryPercent();
      battChar->setValue(&level, 1);
      if (bleConnected) battChar->notify();
    }
  }

  if (millis() - lastDisplayUpdate >= DISPLAY_INTERVAL_MS) {
    lastDisplayUpdate = millis();

    if (displayReady) {
      display.clearDisplay();

      if (!fingerPresent) {
        drawText(0, ROW_TOP, TEXT_SMALL, "Place finger");
        drawText(0, ROW_BOTTOM, TEXT_SMALL, "on sensor");
      } else if (ibiCount < MIN_BEATS_TO_DISPLAY) {
        drawText(0, ROW_TOP, TEXT_SMALL, "Stabilising...");
        char progress[16];
        snprintf(progress, sizeof(progress), "%d/%d beats",
                 ibiCount, MIN_BEATS_TO_DISPLAY);
        drawText(0, ROW_BOTTOM, TEXT_SMALL, progress);
      } else {
        drawText(0, LABEL_TOP, TEXT_SMALL, "BPM");
        char bpmStr[8];
        snprintf(bpmStr, sizeof(bpmStr), "%d", displayBpm);
        drawText(0, BIGNUM_TOP, TEXT_BIG, bpmStr);
      }

      // Link state, so a failed pairing is visible on the band itself rather
      // than only in a log on the station.
      drawText(STATUS_X, STATUS_TOP, TEXT_SMALL, bleConnected ? "BLE" : "---");

      display.display();
    }
  }
}
