# Person-Following Rover

An IoT-enabled autonomous rover that detects, locks onto, and follows a selected person using computer vision, gesture recognition, ultrasonic safety sensing, and a cloud dashboard.

This project was developed as a Senior Design Project for **Design and Implementation of an IoT-Enabled Rover for Person Following**.

---

## Project Overview

The rover uses a two-layer control architecture:

- **Raspberry Pi 5** handles camera input, YOLOv8n person detection, ByteTrack tracking, MediaPipe gesture recognition, dashboard communication, and high-level movement decisions.
- **Arduino Mega 2560** handles motor control, ultrasonic obstacle sensing, NeoPixel/buzzer indicators, and battery telemetry.

The user can lock onto the rover using a **peace-sign gesture** and unlock using an **open-palm gesture**. The system can also be monitored and manually controlled through a cloud-hosted dashboard.

---

## Main Features

- Real-time person detection using **YOLOv8n**
- Target tracking using **ByteTrack**
- Gesture-based target locking and unlocking using **MediaPipe Hands**
- Autonomous person following
- Mecanum-wheel movement
- Obstacle safety using three ultrasonic sensors
- Arduino-based local safety override
- Battery voltage and percentage monitoring
- NeoPixel LED strip and buzzer state indicators
- Cloud dashboard for monitoring and manual override
- Live video streaming

---

## System Architecture

| Layer | Main Components | Function |
|---|---|---|
| Perception and Decision Layer | Raspberry Pi 5, USB camera, YOLOv8n, ByteTrack, MediaPipe | Detects people, tracks the selected target, recognizes gestures, and sends movement commands |
| Actuation and Sensing Layer | Arduino Mega 2560, L298N drivers, DC motors, ultrasonic sensors, NeoPixel strip, buzzer | Executes motor commands, reads sensors, handles obstacle safety, and displays rover state |
| IoT Layer | Cloud dashboard, HTTP communication, video stream | Provides telemetry, live monitoring, and manual override |

---

## Hardware Used

- Raspberry Pi 5
- Arduino Mega 2560
- USB camera
- Four DC / mecanum wheels
- Two L298N dual H-bridge motor drivers
- Three HC-SR04 ultrasonic sensors
- 30-LED NeoPixel strip
- Piezo buzzer
- 4S2P 18650 lithium-ion battery pack
- XL4015 step-down buck converter
- 10,000 mAh USB-C power bank for Raspberry Pi 5

---

## Software Used

### Raspberry Pi

- Raspberry Pi OS 64-bit
- Python 3.11
- OpenCV
- Ultralytics YOLOv8
- ByteTrack
- MediaPipe
- PySerial

### Arduino

- Arduino IDE
- Arduino Mega 2560
- Adafruit NeoPixel library
- Mecanum motor driver library

---

## Key Implementation Details

| Parameter | Value |
|---|---:|
| Camera resolution | 480 × 270 |
| Camera frame rate | 60 FPS |
| YOLO input size | 288 × 288 |
| Person confidence threshold | 0.28 |
| Target center smoothing factor | 0.35 |
| Horizontal deadzone | 18% on each side |
| Stop threshold | 190 px bounding-box height |
| Rotation pulse range | 15 ms to 250 ms |
| Rotation settle time | 180 ms |
| Lost-state timeout | 15 s |
| Lost-state strafe delay | 5 s |
| Obstacle stop distance | 38 cm |
| Arduino command timeout | 160 ms |
| Battery telemetry interval | 1 s |
| Ultrasonic telemetry interval | 250 ms |
| Dashboard telemetry interval | 60 s |

---

## Rover States

| State | Description | Indicator |
|---|---|---|
| FREE | No target is locked | Red moving LED animation |
| LOCKED | Rover is following the selected target | Solid green LED |
| LOST | Target is temporarily lost | Flashing green LED with double beep |
| UNLOCKED | Target has been released | Flashing red LED, then returns to FREE |

---

## Serial Communication

The Raspberry Pi communicates with the Arduino through USB serial at **115,200 baud**.

### Raspberry Pi to Arduino

| Command | Character |
|---|---:|
| Forward | `F` |
| Backward | `B` |
| Stop | `S` |
| Strafe left | `J` |
| Strafe right | `M` |
| Rotate left low / medium / high | `Q / W / E` |
| Rotate right low / medium / high | `I / O / P` |
| FREE indicator | `A` |
| LOCKED indicator | `K` |
| LOST indicator | `X` |
| UNLOCK indicator | `U` |

### Arduino to Raspberry Pi

| Message | Format |
|---|---|
| Battery telemetry | `BAT,voltage,percentage` |
| Ultrasonic telemetry | `SENS,front,left,right` |

---

## Power System

The rover uses two separate power sources:

| Power Source | Supplies |
|---|---|
| 4S2P 18650 lithium-ion battery pack | Motors, Arduino, ultrasonic sensors, LED strip, buzzer |
| 10,000 mAh USB-C power bank | Raspberry Pi 5 and USB camera |

The split power design prevents motor voltage drops and current spikes from affecting the Raspberry Pi.

Estimated runtime:

- Main 4S2P battery pack: about **3.8 hours**
- Raspberry Pi power bank: about **2.1 hours**
- Complete prototype: about **2.1 hours**, limited by the power bank

---

## Dashboard

The rover is connected to a cloud-hosted dashboard:

```text
https://ff15.cam
```

Dashboard functions include:

- Battery monitoring
- CPU temperature monitoring
- Live video stream
- Manual override controls
- Switching back to autonomous mode

---

## Repository Contents

This repository includes:

- Arduino code
- Raspberry Pi camera and control code
- Dashboard-related code
- Rover schematic
- System flowchart and PlantUML files
- Final documentation
- Project figures and supporting files

---

## Testing Summary

The prototype was tested for:

- Person detection and lock acquisition
- Person following at walking speed
- Rotation and lateral tracking
- Obstacle stopping at 38 cm
- Target loss and recovery behavior
- Manual override through the dashboard
- Battery life above 1 hour
- Dashboard telemetry and video streaming

---

## Limitations

- GPS was not implemented because the rover is mainly designed for indoor environments.
- Video streaming delay depends on cloud and network conditions.
- Target reacquisition is limited to short-term target loss.
- Ultrasonic sensors do not fully cover the rover edges.
- Very close obstacles may cause unreliable ultrasonic readings.

---

## Future Work

Possible improvements include:

- Reducing dashboard video delay using local or edge streaming
- Adding indoor localization using UWB, Bluetooth beacons, Wi-Fi positioning, visual SLAM, or AprilTags
- Improving obstacle coverage using LiDAR, depth cameras, or time-of-flight sensors
- Improving target reacquisition after longer loss periods
- Using an AI accelerator or stronger processor
- Adding stronger sensor fusion and multi-person tracking

---

## Authors

- Mira Hindawi
- Hamza Bodair

Supervisor:
- Prof. Esam Qaralleh

---

## Links

Repository:

```text
https://github.com/hamztion/Person-following-Rover-Project.git
```

Dashboard:

```text
https://ff15.cam
```
