#include <mecanum_driver.h>
#include <Adafruit_NeoPixel.h>

// ================= MOTOR DEFINITIONS =================
robot::motor front_right(40, 41, 8);
robot::motor front_left (38, 39, 2);
robot::motor rear_right (36, 37, 10);
robot::motor rear_left  (34, 35, 11);

// ================= RGB LED + BUZZER =================
const int RGB_PIN = 6;
const int BUZZER_PIN = 7;
const int NUM_LEDS = 30;

Adafruit_NeoPixel strip(NUM_LEDS, RGB_PIN, NEO_GRB + NEO_KHZ800);

// ================= ULTRASONIC PINS =================
const int TRIG_F = 28; const int ECHO_F = 29;
const int TRIG_L = 24; const int ECHO_L = 25;
const int TRIG_R = 26; const int ECHO_R = 27;

// ================= BATTERY MONITOR =================
const int BATTERY_PIN = A0;
const float BAT_R1 = 33000.0;
const float BAT_R2 = 10000.0;
const float BAT_VREF = 5.30;

const unsigned long BATTERY_INTERVAL = 1000;
const unsigned long SENSOR_TELEMETRY_INTERVAL = 250;

// ================= DISTANCE THRESHOLDS =================
const float STOP_DIST = 38.0;
const float SIDE_STOP_DIST = 30.0;

// ================= SPEEDS =================
const int MAX_FWD_PWM = 100;
const int MID_FWD_PWM = 100;
const int LOW_FWD_PWM = 80;

const int BACK_PWM    = 70;
const int STRAFE_FAST = 145;
const int STRAFE_MID  = 145;
const int STRAFE_LOW  = 100;

// Reduced rotation speeds to reduce overshoot
const int ROTATE_PWM      = 95;
const int ROTATE_PWM_LOW  = 115;
const int ROTATE_PWM_MED  = 130;
const int ROTATE_PWM_HIGH = 140;

// ================= TIMINGS =================
const unsigned long SENSOR_INTERVAL = 45;
const unsigned long SENSOR_GAP_US = 700;
const unsigned long PI_CMD_TIMEOUT_MS = 160;

// Lower timeout = faster response if sensor misses echo
const unsigned long ULTRASONIC_TIMEOUT_US = 7000;

// ================= PI COMMANDS =================
enum PiCommand {
  PI_NONE,
  PI_STOP,
  PI_FORWARD,
  PI_BACKWARD,

  PI_STRAFE_LEFT,
  PI_STRAFE_RIGHT,

  PI_ROTATE_LEFT,
  PI_ROTATE_RIGHT,

  PI_ROTATE_LEFT_LOW,
  PI_ROTATE_LEFT_MED,
  PI_ROTATE_LEFT_HIGH,

  PI_ROTATE_RIGHT_LOW,
  PI_ROTATE_RIGHT_MED,
  PI_ROTATE_RIGHT_HIGH
};

PiCommand currentPiCmd = PI_STOP;

unsigned long lastPiCmdTime = 0;
unsigned long lastSensorTime = 0;
unsigned long lastBatteryTime = 0;
unsigned long lastSensorTelemetryTime = 0;

// ================= INDICATOR STATES =================
enum IndicatorState {
  IND_FREE,
  IND_LOCKED,
  IND_LOST,
  IND_UNLOCK_FLASH
};

IndicatorState indicatorState = IND_FREE;

unsigned long indicatorStartTime = 0;
unsigned long ledTimer = 0;
unsigned long buzzerTimer = 0;

int freeLedIndex = 0;
bool lostLedOn = false;
int lostBuzzerStep = 0;

// ================= SENSOR VALUES =================
float distF = 999;
float distL = 999;
float distR = 999;

// ================= BATTERY VALUES =================
float batteryVoltage = 0.0;
int batteryPercentage = 0;

// ================= BATTERY FUNCTIONS =================
int getBatteryPercentage(float voltage) {
  if (voltage >= 16.8) return 100;
  if (voltage <= 12.0) return 0;

  if (voltage >= 16.0) return 90 + (voltage - 16.0) * (10.0 / 0.8);
  if (voltage >= 15.2) return 75 + (voltage - 15.2) * (15.0 / 0.8);
  if (voltage >= 14.8) return 60 + (voltage - 14.8) * (15.0 / 0.4);
  if (voltage >= 14.0) return 40 + (voltage - 14.0) * (20.0 / 0.8);
  if (voltage >= 13.2) return 20 + (voltage - 13.2) * (20.0 / 0.8);

  return (voltage - 12.0) * (20.0 / 1.2);
}

float getBatteryVoltage(int rawADC) {
  float Vout = (rawADC / 1023.0) * BAT_VREF;
  return Vout * (BAT_R1 + BAT_R2) / BAT_R2;
}

float readBatteryVoltageAveraged() {
  const int samples = 6;
  long total = 0;

  for (int i = 0; i < samples; i++) {
    total += analogRead(BATTERY_PIN);
    delayMicroseconds(200);
  }

  return getBatteryVoltage(total / samples);
}

void sendBatteryTelemetry() {
  Serial.print("BAT,");
  Serial.print(batteryVoltage, 2);
  Serial.print(",");
  Serial.println(batteryPercentage);
}

void updateBatteryTelemetry() {
  unsigned long now = millis();

  if (now - lastBatteryTime >= BATTERY_INTERVAL) {
    lastBatteryTime = now;
    batteryVoltage = readBatteryVoltageAveraged();
    batteryPercentage = getBatteryPercentage(batteryVoltage);
    sendBatteryTelemetry();
  }
}

void sendSensorTelemetry() {
  Serial.print("SENS,");
  Serial.print(distF, 1);
  Serial.print(",");
  Serial.print(distL, 1);
  Serial.print(",");
  Serial.println(distR, 1);
}

void updateSensorTelemetry() {
  unsigned long now = millis();

  if (now - lastSensorTelemetryTime >= SENSOR_TELEMETRY_INTERVAL) {
    lastSensorTelemetryTime = now;
    sendSensorTelemetry();
  }
}

// ================= RGB HELPERS =================
void clearStrip() {
  strip.clear();
  strip.show();
}

void setAll(uint32_t color) {
  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, color);
  }
  strip.show();
}

void setIndicatorState(IndicatorState newState) {
  if (indicatorState == newState) return;

  indicatorState = newState;
  indicatorStartTime = millis();
  ledTimer = millis();
  buzzerTimer = millis();
  freeLedIndex = 0;
  lostLedOn = false;
  lostBuzzerStep = 0;

  digitalWrite(BUZZER_PIN, LOW);
  clearStrip();

  if (newState == IND_LOCKED) {
    setAll(strip.Color(0, 255, 0));
  }
}

void updateIndicators() {
  unsigned long now = millis();

  switch (indicatorState) {
    case IND_FREE:
      digitalWrite(BUZZER_PIN, LOW);

      if (now - ledTimer >= 35) {
        ledTimer = now;

        for (int i = 0; i < NUM_LEDS; i++) {
          strip.setPixelColor(i, strip.Color(15, 0, 0));
        }

        int led1 = freeLedIndex;
        int led2 = (freeLedIndex - 1 + NUM_LEDS) % NUM_LEDS;
        int led3 = (freeLedIndex - 2 + NUM_LEDS) % NUM_LEDS;

        strip.setPixelColor(led3, strip.Color(60, 0, 0));
        strip.setPixelColor(led2, strip.Color(130, 0, 0));
        strip.setPixelColor(led1, strip.Color(255, 0, 0));

        strip.show();

        freeLedIndex = (freeLedIndex + 1) % NUM_LEDS;
      }
      break;

    case IND_LOCKED:
      digitalWrite(BUZZER_PIN, LOW);
      break;

    case IND_LOST:
      if (now - ledTimer >= 250) {
        ledTimer = now;
        lostLedOn = !lostLedOn;

        if (lostLedOn) setAll(strip.Color(0, 255, 0));
        else clearStrip();
      }

      switch (lostBuzzerStep) {
        case 0:
          digitalWrite(BUZZER_PIN, HIGH);
          buzzerTimer = now;
          lostBuzzerStep = 1;
          break;

        case 1:
          if (now - buzzerTimer >= 120) {
            digitalWrite(BUZZER_PIN, LOW);
            buzzerTimer = now;
            lostBuzzerStep = 2;
          }
          break;

        case 2:
          if (now - buzzerTimer >= 120) {
            digitalWrite(BUZZER_PIN, HIGH);
            buzzerTimer = now;
            lostBuzzerStep = 3;
          }
          break;

        case 3:
          if (now - buzzerTimer >= 120) {
            digitalWrite(BUZZER_PIN, LOW);
            buzzerTimer = now;
            lostBuzzerStep = 4;
          }
          break;

        case 4:
          if (now - buzzerTimer >= 2000) {
            lostBuzzerStep = 0;
          }
          break;
      }
      break;

    case IND_UNLOCK_FLASH:
      digitalWrite(BUZZER_PIN, LOW);

      if (now - ledTimer >= 120) {
        ledTimer = now;
        lostLedOn = !lostLedOn;

        if (lostLedOn) setAll(strip.Color(255, 0, 0));
        else clearStrip();
      }

      if (now - indicatorStartTime >= 1200) {
        setIndicatorState(IND_FREE);
      }
      break;
  }
}

// ================= ULTRASONIC HELPERS =================
void sort3(float &a, float &b, float &c) {
  if (a > b) { float t = a; a = b; b = t; }
  if (b > c) { float t = b; b = c; c = t; }
  if (a > b) { float t = a; a = b; b = t; }
}

float getDistRaw(int trig, int echo) {
  digitalWrite(trig, LOW);
  delayMicroseconds(2);

  digitalWrite(trig, HIGH);
  delayMicroseconds(10);
  digitalWrite(trig, LOW);

  long dur = pulseIn(echo, HIGH, ULTRASONIC_TIMEOUT_US);

  if (dur == 0) return 999.0;

  float d = (dur * 0.0343f) / 2.0f;

  if (d < 2 || d > 400) return 999.0;

  return d;
}

float getDistMedian(int trig, int echo) {
  float a = getDistRaw(trig, echo);
  delayMicroseconds(SENSOR_GAP_US);

  float b = getDistRaw(trig, echo);
  delayMicroseconds(SENSOR_GAP_US);

  float c = getDistRaw(trig, echo);

  sort3(a, b, c);
  return b;
}

void updateSensors() {
  static int sensorIndex = 0;

  if (sensorIndex == 0) {
    distF = getDistMedian(TRIG_F, ECHO_F);
  } 
  else if (sensorIndex == 1) {
    distL = getDistMedian(TRIG_L, ECHO_L);
  } 
  else {
    distR = getDistMedian(TRIG_R, ECHO_R);
  }

  sensorIndex = (sensorIndex + 1) % 3;
}

// ================= MOTOR COMMANDS =================
void stopAll() {
  front_right.halt();
  front_left.halt();
  rear_right.halt();
  rear_left.halt();
}

void setForward(int spd) {
  front_left.move_cclockwise(spd);
  front_right.move_cclockwise(spd);
  rear_left.move_cclockwise(spd);
  rear_right.move_cclockwise(spd);
}

void setBackward(int spd) {
  front_left.move_clockwise(spd);
  front_right.move_clockwise(spd);
  rear_left.move_clockwise(spd);
  rear_right.move_clockwise(spd);
}

void setStrafeRight(int spd) {
  front_left.move_cclockwise(spd);
  front_right.move_clockwise(spd);
  rear_left.move_clockwise(spd);
  rear_right.move_cclockwise(spd);
}

void setStrafeLeft(int spd) {
  front_left.move_clockwise(spd);
  front_right.move_cclockwise(spd);
  rear_left.move_cclockwise(spd);
  rear_right.move_clockwise(spd);
}

void setRotateRight(int spd) {
  front_left.move_cclockwise(spd);
  rear_left.move_cclockwise(spd);
  front_right.move_clockwise(spd);
  rear_right.move_clockwise(spd);
}

void setRotateLeft(int spd) {
  front_left.move_clockwise(spd);
  rear_left.move_clockwise(spd);
  front_right.move_cclockwise(spd);
  rear_right.move_cclockwise(spd);
}

int forwardSpeedFromDistance(float d) {
  if (d > 120) return MAX_FWD_PWM;
  if (d > 75)  return MID_FWD_PWM;
  if (d > STOP_DIST) return LOW_FWD_PWM;
  return 0;
}

// ================= SERIAL READ =================
void readPiSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    unsigned long now = millis();

    switch (c) {
      case 'F':
        currentPiCmd = PI_FORWARD;
        lastPiCmdTime = now;
        break;

      case 'B':
        currentPiCmd = PI_BACKWARD;
        lastPiCmdTime = now;
        break;

      case 'S':
        currentPiCmd = PI_STOP;
        lastPiCmdTime = now;
        break;

      case 'J':
        currentPiCmd = PI_STRAFE_LEFT;
        lastPiCmdTime = now;
        break;

      case 'M':
        currentPiCmd = PI_STRAFE_RIGHT;
        lastPiCmdTime = now;
        break;

      case 'L':
        currentPiCmd = PI_ROTATE_LEFT;
        lastPiCmdTime = now;
        break;

      case 'R':
        currentPiCmd = PI_ROTATE_RIGHT;
        lastPiCmdTime = now;
        break;

      case 'Q':
        currentPiCmd = PI_ROTATE_LEFT_LOW;
        lastPiCmdTime = now;
        break;

      case 'W':
        currentPiCmd = PI_ROTATE_LEFT_MED;
        lastPiCmdTime = now;
        break;

      case 'E':
        currentPiCmd = PI_ROTATE_LEFT_HIGH;
        lastPiCmdTime = now;
        break;

      case 'I':
        currentPiCmd = PI_ROTATE_RIGHT_LOW;
        lastPiCmdTime = now;
        break;

      case 'O':
        currentPiCmd = PI_ROTATE_RIGHT_MED;
        lastPiCmdTime = now;
        break;

      case 'P':
        currentPiCmd = PI_ROTATE_RIGHT_HIGH;
        lastPiCmdTime = now;
        break;

      case 'A':
        setIndicatorState(IND_FREE);
        break;

      case 'K':
        setIndicatorState(IND_LOCKED);
        break;

      case 'X':
        setIndicatorState(IND_LOST);
        break;

      case 'U':
        setIndicatorState(IND_UNLOCK_FLASH);
        break;

      default:
        break;
    }
  }
}

// ================= APPLY PI COMMAND =================
void applyPiCommand() {
  if ((millis() - lastPiCmdTime) > PI_CMD_TIMEOUT_MS) {
    currentPiCmd = PI_STOP;
  }

  switch (currentPiCmd) {
    case PI_FORWARD:
      if (distF > STOP_DIST) {
        int spd = forwardSpeedFromDistance(distF);
        if (spd > 0) setForward(spd);
        else stopAll();
      } else {
        stopAll();
      }
      break;

    case PI_BACKWARD:
      setBackward(BACK_PWM);
      break;

    case PI_STRAFE_LEFT:
      if (distL > SIDE_STOP_DIST) setStrafeLeft(STRAFE_MID);
      else stopAll();
      break;

    case PI_STRAFE_RIGHT:
      if (distR > SIDE_STOP_DIST) setStrafeRight(STRAFE_MID);
      else stopAll();
      break;

    case PI_ROTATE_LEFT:
      setRotateLeft(ROTATE_PWM);
      break;

    case PI_ROTATE_RIGHT:
      setRotateRight(ROTATE_PWM);
      break;

    case PI_ROTATE_LEFT_LOW:
      setRotateLeft(ROTATE_PWM_LOW);
      break;

    case PI_ROTATE_LEFT_MED:
      setRotateLeft(ROTATE_PWM_MED);
      break;

    case PI_ROTATE_LEFT_HIGH:
      setRotateLeft(ROTATE_PWM_HIGH);
      break;

    case PI_ROTATE_RIGHT_LOW:
      setRotateRight(ROTATE_PWM_LOW);
      break;

    case PI_ROTATE_RIGHT_MED:
      setRotateRight(ROTATE_PWM_MED);
      break;

    case PI_ROTATE_RIGHT_HIGH:
      setRotateRight(ROTATE_PWM_HIGH);
      break;

    case PI_STOP:
    case PI_NONE:
    default:
      stopAll();
      break;
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  pinMode(BATTERY_PIN, INPUT);

  strip.begin();
  strip.setBrightness(80);
  clearStrip();

  pinMode(TRIG_F, OUTPUT); pinMode(ECHO_F, INPUT);
  pinMode(TRIG_L, OUTPUT); pinMode(ECHO_L, INPUT);
  pinMode(TRIG_R, OUTPUT); pinMode(ECHO_R, INPUT);

  digitalWrite(TRIG_F, LOW);
  digitalWrite(TRIG_L, LOW);
  digitalWrite(TRIG_R, LOW);

  stopAll();
  delay(300);

  updateSensors();
  updateSensors();
  updateSensors();

  batteryVoltage = readBatteryVoltageAveraged();
  batteryPercentage = getBatteryPercentage(batteryVoltage);
  lastBatteryTime = millis();
  sendBatteryTelemetry();

  lastPiCmdTime = millis();
  currentPiCmd = PI_STOP;

  indicatorState = IND_LOCKED;
  setIndicatorState(IND_FREE);
}

// ================= MAIN LOOP =================
void loop() {
  unsigned long now = millis();

  readPiSerial();
  applyPiCommand();

  if (now - lastSensorTime >= SENSOR_INTERVAL) {
    lastSensorTime = now;
    updateSensors();
  }

  readPiSerial();
  applyPiCommand();

  updateBatteryTelemetry();
  updateSensorTelemetry();
  updateIndicators();
}
