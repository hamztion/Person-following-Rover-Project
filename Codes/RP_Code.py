#!/usr/bin/env python3

import atexit
import signal
import sys
import fcntl
import gc
import json
import math
import os
import subprocess
import socket
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from types import SimpleNamespace

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import requests
import serial
from ultralytics import YOLO

# ---------------- DASHBOARD CONFIG ----------------
CONFIG_DIR = Path.home() / "rover" / "rover-pi-client"
ENV_FILE = CONFIG_DIR / ".env"

def read_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        if not path.exists():
            return values

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    except Exception as e:
        print(f"[DASHBOARD] Failed to read {path}: {e}")

    return values


ENV_VALUES = read_env_file(ENV_FILE)


def config_value(key: str, default: Any = "") -> str:
    return str(os.getenv(key, ENV_VALUES.get(key, default)))


def config_float(key: str, default: Any) -> float:
    try:
        return float(config_value(key, default))
    except (TypeError, ValueError):
        return float(default)


def config_int(key: str, default: Any) -> int:
    try:
        return int(float(config_value(key, default)))
    except (TypeError, ValueError):
        return int(default)


def config_bool(key: str, default: Any) -> bool:
    value = config_value(key, default).strip().lower()
    return value not in ("0", "false", "no", "off")


DASHBOARD_URL = config_value("DASHBOARD_URL", "").rstrip("/")
API_TOKEN = config_value("API_TOKEN", config_value("ROVER_TOKEN", ""))
ROVER_ID = config_value("ROVER_ID", "")

COMMAND_POLL_INTERVAL = config_float("COMMAND_POLL_INTERVAL", 0.25)
DASHBOARD_REQUEST_TIMEOUT = config_float("DASHBOARD_REQUEST_TIMEOUT", 5.0)
HEARTBEAT_INTERVAL = config_float("HEARTBEAT_INTERVAL", 10.0)
TELEMETRY_INTERVAL = config_float("TELEMETRY_INTERVAL", 60.0)
MANUAL_MOVE_HOLD_SECONDS = config_float("MANUAL_MOVE_HOLD_SECONDS", 0.6)
MANUAL_STOP_HOLD_SECONDS = config_float("MANUAL_STOP_HOLD_SECONDS", 0.25)

STREAM_URL = config_value("STREAM_URL", "").strip()
STREAM_ENABLED = config_bool("STREAM_ENABLED", "true")
STREAM_MODE = config_value("STREAM_MODE", "relay").strip().lower()
STREAM_PUBLIC_URL = config_value("STREAM_PUBLIC_URL", "").strip()
ROVER_IP = config_value("ROVER_IP", config_value("PI_IP", "")).strip()
STREAM_PORT = config_value("STREAM_PORT", "8081").strip()
STREAM_PATH = config_value("STREAM_PATH", "/video").strip() or "/video"
PUBLIC_STREAM_URL = STREAM_PUBLIC_URL


def derive_public_stream_url() -> Optional[str]:
    public_url = STREAM_PUBLIC_URL.strip()
    if public_url:
        return public_url

    explicit = STREAM_URL.strip()
    if explicit.startswith(("http://", "https://")):
        return explicit

    if ROVER_IP:
        host = ROVER_IP.replace("http://", "").replace("https://", "").rstrip("/")
        port = STREAM_PORT or "8081"
        path = STREAM_PATH if STREAM_PATH.startswith("/") else f"/{STREAM_PATH}"
        return f"http://{host}:{port}{path}"

    return None


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return ROVER_IP or ""

# ---------------- FOLLOWER CONFIG ----------------
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200

last_send_time = 0.0
SEND_REPEAT_INTERVAL = 0.12
SERIAL_KEEPALIVE_INTERVAL = config_float("SERIAL_KEEPALIVE_INTERVAL", 0.05)

MODEL_PATH = config_value("MODEL_PATH", "/home/hamzamira/models/yolov8n_288.onnx")
CAM_INDEX = config_int("CAM_INDEX", 10)
CAMERA_FALLBACKS = (0, 1, 2)

LOST_TIMEOUT = 15.0
SMOOTHING_FACTOR = 0.35
DEADZONE_WIDTH = 0.18
MAX_CAMERA_FAILS = 20

LOCK_HOLD_SECONDS = 0.20
UNLOCK_HOLD_SECONDS = 1.2

PERSON_CONF = config_float("PERSON_CONF", 0.28)
YOLO_IMGSZ = config_int("YOLO_IMGSZ", 288)  
YOLO_MAX_DET = config_int("YOLO_MAX_DET", 4)
TRACKER_CONFIG = config_value("TRACKER_CONFIG", "bytetrack.yaml")
HAND_PROCESS_SCALE = config_float("HAND_PROCESS_SCALE", 0.45)
MAX_HANDS_FREE = config_int("MAX_HANDS_FREE", 4)
MAX_HANDS_LOCKED = config_int("MAX_HANDS_LOCKED", 2)
HAND_DETECTION_CONF = config_float("HAND_DETECTION_CONF", 0.50)
HAND_TRACKING_CONF = config_float("HAND_TRACKING_CONF", 0.50)
HAND_CROP_PADDING_FRAC = config_float("HAND_CROP_PADDING_FRAC", 0.12)

FRAME_W = config_int("FRAME_W", 480)
FRAME_H = config_int("FRAME_H", 270)
CAMERA_FPS = config_int("CAMERA_FPS", 60)
CAMERA_READ_THROTTLE_FPS = config_float("CAMERA_READ_THROTTLE_FPS", 0)
CAMERA_THREADED = config_bool("CAMERA_THREADED", "false")
CAMERA_READ_TIMEOUT = config_float("CAMERA_READ_TIMEOUT", 1.0)


TRACK_EVERY_N_FRAMES_FREE = config_int("TRACK_EVERY_N_FRAMES_FREE", 3)
TRACK_EVERY_N_FRAMES_LOCKED = config_int("TRACK_EVERY_N_FRAMES_LOCKED", 1)
HANDS_EVERY_N_FRAMES_FREE = config_int("HANDS_EVERY_N_FRAMES_FREE", 2)
HANDS_EVERY_N_FRAMES_LOCKED = config_int("HANDS_EVERY_N_FRAMES_LOCKED", 3)


STOP_HEIGHT_THRESHOLD = 190
FORWARD_HEIGHT_THRESHOLD = 235
HEIGHT_SMOOTH_ALPHA = 0.18

MIN_PERSON_HEIGHT_PX = 40
MIN_VISIBLE_HEIGHT_FRAC = 0.75
MIN_VISIBLE_AREA_FRAC = 0.65

PEACE_MIN_PERSON_HEIGHT_PX = 35

BOX_SIDE_RATIO = 0.42
BOX_CENTER_Y_RATIO = 0.38


ROTATE_PULSE_TIME_FAR = 0.015
ROTATE_PULSE_TIME_MID = 0.025
ROTATE_PULSE_TIME_CLOSE = 0.250

# Height range used for pulse scaling.
ROTATE_HEIGHT_FAR = 150
ROTATE_HEIGHT_CLOSE = 230

ROTATE_SETTLE_TIME = 0.180


FAR_CURVE_HEIGHT_THRESHOLD = config_int("FAR_CURVE_HEIGHT_THRESHOLD", 170)


CENTER_PREFERENCE_WEIGHT = 0.65
SIZE_PREFERENCE_WEIGHT = 0.35


LOCK_HAND_REGION_FRAC = 0.80
UNLOCK_HAND_REGION_FRAC = 0.60


POST_LOCK_STABLE_FRAMES = 3


TARGET_INVALID_GRACE_TIME = 1.20
ROTATION_TARGET_GRACE_TIME = 1.80

AUTO_REACQUIRE_SECONDS = config_float("AUTO_REACQUIRE_SECONDS", 4.0)


LOST_STRAFE_AFTER_SECONDS = config_float("LOST_STRAFE_AFTER_SECONDS", 5.0)
LOST_SIDE_STOP_CM = config_float("LOST_SIDE_STOP_CM", 30.0)
LOST_STRAFE_PULSE_SECONDS = config_float("LOST_STRAFE_PULSE_SECONDS", 0.45)
LOST_STRAFE_CHECK_SECONDS = config_float("LOST_STRAFE_CHECK_SECONDS", 0.25)
ARDUINO_TELEMETRY_MAX_AGE = config_float("ARDUINO_TELEMETRY_MAX_AGE", 2.0)

REACQUIRE_CENTER_DISTANCE_FRAC = 0.28
REACQUIRE_HEIGHT_DIFF_FRAC = 0.35
REACQUIRE_MIN_HIST_CORR = 0.55
REACQUIRE_SCORE_GAP = 0.15


OCCLUSION_TARGET_OVERLAP_FRAC = 0.22
OCCLUSION_OTHER_OVERLAP_FRAC = 0.22
OCCLUSION_IOU_FRAC = 0.15
OCCLUSION_LOST_TIME = 0.65

IDENTITY_CENTER_JUMP_FRAC = config_float("IDENTITY_CENTER_JUMP_FRAC", 0.30)
IDENTITY_HEIGHT_DIFF_FRAC = config_float("IDENTITY_HEIGHT_DIFF_FRAC", 0.45)
IDENTITY_MIN_HIST_CORR = config_float("IDENTITY_MIN_HIST_CORR", 0.45)


TARGET_HIST_UPPER_FRAC = 0.65
TARGET_HIST_UPDATE_ALPHA = 0.08


PRINT_INTERVAL = 1.0

# --- Speed monitoring from Arduino encoder telemetry ---
SPEED_REQUIREMENT_MPS = config_float("SPEED_REQUIREMENT_MPS", 0.50)
SPEED_PRINT_ONLY_WHEN_FORWARD = config_bool("SPEED_PRINT_ONLY_WHEN_FORWARD", "true")

# ---------------- GLOBALS ----------------
cap = None
ser = None
last_sent = None
last_action = "STOP"

MOVEMENT_SERIAL_CHARS = set("FBSJMLRQWEIOP")
current_serial_movement_char = "S"
serial_command_lock = threading.Lock()
serial_sender_thread = None
opened_captures = []
opened_captures_lock = threading.Lock()


ROVER_LOCK_FILE = Path("/tmp/person_following_rover_camera.lock")
rover_lock_handle = None


arduino_battery_voltage: Optional[float] = None
arduino_battery_percentage: Optional[int] = None
arduino_battery_time = 0.0
arduino_dist_f: Optional[float] = None
arduino_dist_l: Optional[float] = None
arduino_dist_r: Optional[float] = None
arduino_sensor_time = 0.0

arduino_speed_mps: Optional[float] = None
arduino_speed_cmps: Optional[float] = None
arduino_speed_kmh: Optional[float] = None
arduino_speed_rpm: Optional[float] = None
arduino_speed_pulses: Tuple[int, int, int, int] = (0, 0, 0, 0)
arduino_speed_time = 0.0

speed_forward_sample_count = 0
speed_forward_sum_mps = 0.0
speed_forward_max_mps = 0.0
speed_forward_last_report_time = 0.0

last_indicator_sent = None
unlock_indicator_until = 0.0

dashboard_session = requests.Session()
stream_session = requests.Session()
telemetry_session = requests.Session()
shutdown_event = threading.Event()
serial_lock = threading.Lock()
manual_lock = threading.Lock()
manual_mode_enabled = False
manual_override_until = 0.0
manual_active_cmd: Optional[str] = None
manual_speed = 50
manual_last_command_id = None
last_dashboard_manual_flag: Optional[bool] = None

latest_frame_lock = threading.Lock()
latest_stream_frame = None
latest_stream_frame_seq = 0
latest_stream_jpeg: Optional[bytes] = None
latest_stream_jpeg_seq = -1
last_stream_frame_capture_time = 0.0
STREAM_PUBLISH_FPS = config_float("STREAM_PUBLISH_FPS", 6)
STREAM_CAPTURE_FPS = config_float("STREAM_CAPTURE_FPS", STREAM_PUBLISH_FPS)
STREAM_CONNECT_TIMEOUT = config_float("STREAM_CONNECT_TIMEOUT", 3.0)
STREAM_READ_TIMEOUT = config_float("STREAM_READ_TIMEOUT", 5.0)
STREAM_JPEG_QUALITY = config_int("STREAM_JPEG_QUALITY", 40)
STREAM_FRAME_WIDTH = config_int("STREAM_FRAME_WIDTH", FRAME_W)
STREAM_FRAME_HEIGHT = config_int("STREAM_FRAME_HEIGHT", 0)
STREAM_RESEND_STALE_FRAMES = config_bool("STREAM_RESEND_STALE_FRAMES", "true")
STREAM_FAILURE_BACKOFF = config_float("STREAM_FAILURE_BACKOFF", 0.25)

HLS_OUTPUT_DIR = Path(config_value("HLS_OUTPUT_DIR", "/tmp/rover-hls"))
HLS_PLAYLIST_NAME = config_value("HLS_PLAYLIST_NAME", "stream.m3u8")
HLS_PUBLIC_URL = config_value("HLS_PUBLIC_URL", "").strip()
HLS_FPS = config_float("HLS_FPS", 15)
HLS_SEGMENT_SECONDS = config_float("HLS_SEGMENT_SECONDS", 0.5)
HLS_LIST_SIZE = config_int("HLS_LIST_SIZE", 3)
HLS_BITRATE = config_value("HLS_BITRATE", "800k")
HLS_ENCODER = config_value("HLS_ENCODER", "libx264")
HLS_UPLOAD_POLL_INTERVAL = config_float("HLS_UPLOAD_POLL_INTERVAL", 0.05)


loop_fps_ewma = 0.0
last_loop_time = 0.0


latest_hls_frame_lock = threading.Lock()
latest_hls_frame = None
latest_hls_frame_seq = 0
last_hls_frame_time = 0.0
hls_upload_state: Dict[str, Tuple[float, int]] = {}

# ---------------- MODELS ----------------
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(config_int("OPENCV_THREADS", 1))
except Exception:
    pass

try:
    import torch

    torch.set_num_threads(config_int("TORCH_THREADS", 2))
except Exception:
    pass

model = YOLO(MODEL_PATH)
try:
    model.fuse()
except Exception:
    pass

mp_hands = mp.solutions.hands
hands_free = mp_hands.Hands(
    max_num_hands=MAX_HANDS_FREE,
    min_detection_confidence=HAND_DETECTION_CONF,
    min_tracking_confidence=HAND_TRACKING_CONF,
)
hands_locked = mp_hands.Hands(
    max_num_hands=MAX_HANDS_LOCKED,
    min_detection_confidence=HAND_DETECTION_CONF,
    min_tracking_confidence=HAND_TRACKING_CONF,
)

# ---------------- STATE ----------------
FREE, LOCKED, LOST, DISABLED = "FREE", "LOCKED", "LOST", "DISABLED"
state = FREE

target_track_id = None
lost_timer = None
lost_recovery_cmd = None
lost_recovery_timer_seen = None
lost_recovery_phase = "IDLE"   # IDLE, STRAFE, CHECK, STOP
lost_recovery_phase_until = 0.0
smooth_tx = None
smooth_height = None
camera_fail_count = 0

peace_hold_start = None
peace_candidate_track = None
open_palm_hold_start = None

frame_count = 0
cached_person_boxes = {}
cached_hand_results = None

target_stable_count = 0
target_invalid_since = None
target_hist = None
auto_reacquire_until = 0.0

pulse_active = False
pulse_cmd = None
pulse_start_time = 0.0
pulse_duration = 0.0
settle_until = 0.0

last_print_time = 0.0
last_ambiguous_peace_print = 0.0


# ---------------- DASHBOARD HELPERS ----------------
def load_dashboard_config():
    """Load dashboard config from ~/rover/rover-pi-client/.env."""
    global DASHBOARD_URL, API_TOKEN, ROVER_ID
    global STREAM_URL, STREAM_ENABLED, STREAM_MODE, STREAM_PUBLIC_URL, PUBLIC_STREAM_URL, ROVER_IP, STREAM_PORT

    DASHBOARD_URL = config_value("DASHBOARD_URL", DASHBOARD_URL).rstrip("/")
    API_TOKEN = config_value("API_TOKEN", config_value("ROVER_TOKEN", API_TOKEN))
    ROVER_ID = config_value("ROVER_ID", ROVER_ID)
    STREAM_URL = config_value("STREAM_URL", STREAM_URL).strip()
    STREAM_MODE = config_value("STREAM_MODE", STREAM_MODE).strip().lower()
    STREAM_PUBLIC_URL = config_value("STREAM_PUBLIC_URL", STREAM_PUBLIC_URL).strip()
    PUBLIC_STREAM_URL = STREAM_PUBLIC_URL
    ROVER_IP = config_value("ROVER_IP", config_value("PI_IP", ROVER_IP)).strip()
    STREAM_PORT = config_value("STREAM_PORT", STREAM_PORT).strip()
    STREAM_ENABLED = config_bool("STREAM_ENABLED", STREAM_ENABLED)

    if DASHBOARD_URL and API_TOKEN and ROVER_ID:
        print(f"[DASHBOARD] Config loaded: {DASHBOARD_URL}, rover {ROVER_ID}")
    else:
        print("[DASHBOARD] Missing DASHBOARD_URL/API_TOKEN/ROVER_ID; manual dashboard control disabled.")

    if hls_stream_enabled():
        print(f"[STREAM] HLS upload mode enabled: {HLS_PUBLIC_URL or dashboard_public_url('/rover/hls/' + HLS_PLAYLIST_NAME)}")
    elif frame_relay_enabled():
        print(f"[STREAM] Dashboard frame relay enabled: {DASHBOARD_URL}/api/rover/frame")
    elif STREAM_MODE == "direct":
        print(f"[STREAM] Direct stream mode enabled: {derive_public_stream_url() or '(missing STREAM_URL)'}")
    else:
        print("[STREAM] Stream publishing disabled.")


def dashboard_configured():
    return bool(DASHBOARD_URL and API_TOKEN and ROVER_ID)


def frame_relay_enabled() -> bool:
    return STREAM_ENABLED and STREAM_MODE not in ("direct", "hls", "off", "false", "none")


def hls_stream_enabled() -> bool:
    return STREAM_ENABLED and STREAM_MODE == "hls"


def dashboard_api_url(path: str) -> str:
    base = DASHBOARD_URL.rstrip("/")
    if base.endswith("/api"):
        return f"{base}{path}"
    return f"{base}/api{path}"


def dashboard_public_url(path: str) -> str:
    base = DASHBOARD_URL.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    return f"{base}{path}"


def dashboard_headers(content_type=False):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "X-Rover-Id": ROVER_ID,
        "Accept": "application/json",
    }
    if content_type:
        headers["Content-Type"] = "application/json"
    return headers


def send_heartbeat():
    try:
        response = dashboard_session.post(
            dashboard_api_url("/rover/heartbeat"),
            json={},
            headers=dashboard_headers(content_type=True),
            timeout=DASHBOARD_REQUEST_TIMEOUT,
        )
        if response.status_code not in (200, 201):
            print(f"[DASHBOARD] Heartbeat failed: {response.status_code}")
    except Exception as e:
        print(f"[DASHBOARD] Heartbeat error: {e}")


def fetch_pending_commands():
    global manual_mode_enabled
    try:
        response = dashboard_session.get(
            dashboard_api_url("/rover/commands/pending"),
            headers=dashboard_headers(),
            timeout=DASHBOARD_REQUEST_TIMEOUT,
        )
        if response.status_code == 200:
            data = response.json()
            is_manual_mode = data.get("is_manual_mode", False)
            
            global last_dashboard_manual_flag
            with manual_lock:
                curr_manual_state = manual_mode_enabled

            if last_dashboard_manual_flag != is_manual_mode:
                print(f"[DASHBOARD] DB manual flag changed: is_manual_mode={is_manual_mode}, Pi State={curr_manual_state}")
                last_dashboard_manual_flag = is_manual_mode

            if is_manual_mode and not curr_manual_state:
                print("[DASHBOARD] Database indicates manual mode is enabled. Switching to manual.")
                enter_manual_mode()
            elif not is_manual_mode and curr_manual_state:
                print("[DASHBOARD] Database indicates manual mode is disabled. Switching to automatic.")
                exit_manual_mode()

            return data.get("commands", [])

        print(f"[DASHBOARD] Command fetch failed: {response.status_code} {response.text[:160]}")
    except Exception as e:
        print(f"[DASHBOARD] Command fetch error: {e}")

    return []


def mark_command_complete(cmd_id, status, message):
    if cmd_id is None:
        return

    try:
        payload = {
            "status": status,
            "response": message if isinstance(message, str) else json.dumps(message),
        }
        response = dashboard_session.post(
            dashboard_api_url(f"/rover/commands/{cmd_id}/complete"),
            json=payload,
            headers=dashboard_headers(content_type=True),
            timeout=DASHBOARD_REQUEST_TIMEOUT,
        )
        if response.status_code not in (200, 201):
            print(f"[DASHBOARD] Complete failed for command {cmd_id}: {response.status_code} {response.text[:160]}")
    except Exception as e:
        print(f"[DASHBOARD] Complete error for command {cmd_id}: {e}")


def update_stream_url(stream_url: str) -> bool:
    if not dashboard_configured() or not stream_url:
        return False

    try:
        response = dashboard_session.patch(
            dashboard_api_url("/rover/settings"),
            json={"stream_url": stream_url},
            headers=dashboard_headers(content_type=True),
            timeout=DASHBOARD_REQUEST_TIMEOUT,
        )
        if response.status_code in (200, 201):
            print(f"[DASHBOARD] Stream URL updated: {stream_url}")
            return True

        print(f"[DASHBOARD] Stream URL update failed: {response.status_code} {response.text[:160]}")
    except Exception as e:
        print(f"[DASHBOARD] Stream URL update error: {e}")

    return False


def update_rover_network_info(stream_url: str, ip_address: str, stream_port: int) -> bool:
    if not dashboard_configured():
        return False

    payload = {
        "stream_url": stream_url,
        "ip_address": ip_address,
        "stream_port": stream_port,
    }

    try:
        response = dashboard_session.patch(
            dashboard_api_url("/rover/settings"),
            json=payload,
            headers=dashboard_headers(content_type=True),
            timeout=DASHBOARD_REQUEST_TIMEOUT,
        )
        if response.status_code in (200, 201):
            print(f"[DASHBOARD] Network info updated: {ip_address}:{stream_port} -> {stream_url}")
            return True

        print(f"[DASHBOARD] Network info update failed: {response.status_code} {response.text[:160]}")
    except Exception as e:
        print(f"[DASHBOARD] Network info update error: {e}")

    return False


def read_sysfs(path: Path, default: str = "0") -> str:
    try:
        if path.exists():
            return path.read_text().strip()
    except Exception:
        pass
    return default


def get_temperature_data() -> Dict[str, Any]:
    temp_data: Dict[str, Any] = {}

    try:
        thermal_file = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_file.exists():
            temp_data["cpu_temp"] = round(int(thermal_file.read_text().strip()) / 1000, 2)
    except Exception:
        pass

    if "cpu_temp" not in temp_data:
        try:
            result = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                temp_str = result.stdout.split("=")[1].replace("'C", "").strip()
                temp_data["cpu_temp"] = float(temp_str)
        except Exception:
            pass

    try:
        with open("/proc/loadavg", "r") as f:
            load_avg = float(f.read().split()[0])
            temp_data["motor_temp"] = round(35 + (load_avg * 5), 2)
    except Exception:
        temp_data["motor_temp"] = 40

    if "cpu_temp" not in temp_data:
        temp_data["cpu_temp"] = 45

    if "ambient_temp" not in temp_data:
        temp_data["ambient_temp"] = temp_data.get("motor_temp", temp_data["cpu_temp"])

    return temp_data


def get_battery_data() -> Dict[str, Any]:
    battery_info: Dict[str, Any] = {}


    if (
        arduino_battery_voltage is not None
        and arduino_battery_percentage is not None
        and (time.time() - arduino_battery_time) <= ARDUINO_TELEMETRY_MAX_AGE
    ):
        return {
            "percentage": int(arduino_battery_percentage),
            "voltage": round(float(arduino_battery_voltage), 2),
            "current": 0,
            "charging": False,
            "source": "arduino",
        }

    power_supply_path = Path("/sys/class/power_supply")

    try:
        if power_supply_path.exists():
            for battery_dir in power_supply_path.iterdir():
                if not battery_dir.name.startswith("BAT"):
                    continue

                capacity_file = battery_dir / "capacity"
                voltage_file = battery_dir / "voltage_now"
                current_file = battery_dir / "current_now"

                try:
                    battery_info["percentage"] = int(read_sysfs(capacity_file, "100"))
                except Exception:
                    battery_info["percentage"] = 100

                try:
                    voltage_uv = int(read_sysfs(voltage_file, "0"))
                    battery_info["voltage"] = round(voltage_uv / 1000000, 2)
                except Exception:
                    battery_info["voltage"] = 12.0

                try:
                    current_ua = int(read_sysfs(current_file, "0"))
                    battery_info["current"] = round(current_ua / 1000000, 2)
                except Exception:
                    battery_info["current"] = 2.5

                battery_info["charging"] = read_sysfs(battery_dir / "status", "Unknown") == "Charging"
                break
    except Exception:
        pass

    if not battery_info:
        battery_info = {
            "percentage": 85,
            "voltage": 12.0,
            "current": 2.5,
            "charging": False,
        }

    return battery_info


def get_gps_data() -> Dict[str, Any]:
    try:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            sock.connect(("localhost", 2947))
            sock.send(b'?WATCH={"enable":true,"json":true}\n')
            data = sock.recv(4096).decode(errors="ignore")

            for line in data.splitlines():
                if '"class":"TPV"' not in line:
                    continue
                report = json.loads(line)
                return {
                    "latitude": report.get("lat", 0),
                    "longitude": report.get("lon", 0),
                    "altitude": report.get("alt", 0),
                    "speed": report.get("speed", 0),
                    "heading": report.get("track", 0),
                    "satellites": report.get("satellites", 0),
                    "accuracy": report.get("eph", 0),
                }
        finally:
            sock.close()
    except Exception:
        pass

    return {
        "latitude": 0,
        "longitude": 0,
        "altitude": 0,
        "speed": 0,
        "heading": 0,
        "satellites": 0,
        "accuracy": 0,
    }


def send_telemetry(telemetry_type: str, data: Dict[str, Any]) -> bool:
    if not dashboard_configured():
        return False

    try:
        payload = {
            "type": telemetry_type,
            "data": data,
            "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        response = telemetry_session.post(
            dashboard_api_url("/telemetry"),
            json=payload,
            headers=dashboard_headers(content_type=True),
            timeout=DASHBOARD_REQUEST_TIMEOUT,
        )
        if response.status_code in (200, 201):
            return True

        print(f"[DASHBOARD] Telemetry failed ({telemetry_type}): {response.status_code} {response.text[:160]}")
    except Exception as e:
        print(f"[DASHBOARD] Telemetry error ({telemetry_type}): {e}")

    return False


def get_ultrasonic_data() -> Dict[str, Any]:
    fresh = (time.time() - arduino_sensor_time) <= ARDUINO_TELEMETRY_MAX_AGE
    return {
        "front_cm": arduino_dist_f if fresh else None,
        "left_cm": arduino_dist_l if fresh else None,
        "right_cm": arduino_dist_r if fresh else None,
        "fresh": fresh,
        "source": "arduino",
    }


def get_speed_data() -> Dict[str, Any]:
    fresh = (
        arduino_speed_mps is not None
        and (time.time() - arduino_speed_time) <= ARDUINO_TELEMETRY_MAX_AGE
    )

    return {
        "mps": arduino_speed_mps if fresh else None,
        "cmps": arduino_speed_cmps if fresh else None,
        "kmh": arduino_speed_kmh if fresh else None,
        "rpm": arduino_speed_rpm if fresh else None,
        "fresh": fresh,
        "source": "arduino_encoder",
        "requirement_mps": SPEED_REQUIREMENT_MPS,
        "meets_requirement": (
            bool(fresh)
            and arduino_speed_mps is not None
            and arduino_speed_mps >= SPEED_REQUIREMENT_MPS
        ),
    }


def collect_and_send_tracking_data():
    send_telemetry("gps", get_gps_data())
    send_telemetry("battery", get_battery_data())
    send_telemetry("temperature", get_temperature_data())
    send_telemetry("ultrasonic", get_ultrasonic_data())
    send_telemetry("speed", get_speed_data())


def speed_to_turn_command(side: str, speed: int) -> str:
    speed = clamp(int(speed), 0, 100)
    if side == "left":
        if speed <= 35:
            return "ROTATE_LEFT_LOW"
        if speed <= 70:
            return "ROTATE_LEFT_MED"
        return "ROTATE_LEFT_HIGH"

    if speed <= 35:
        return "ROTATE_RIGHT_LOW"
    if speed <= 70:
        return "ROTATE_RIGHT_MED"
    return "ROTATE_RIGHT_HIGH"


def manual_move_command(direction: str, speed: int) -> Optional[str]:
    direction = str(direction).lower()
    if direction == "forward":
        return "FORWARD"
    if direction == "backward":
        return "BACKWARD"
    if direction == "left":
        return "STRAFE_LEFT"
    if direction == "right":
        return "STRAFE_RIGHT"
    return None


def reset_to_unlocked_state(show_free_indicator: bool = True):

    global state, target_track_id, lost_timer, smooth_tx, smooth_height
    global peace_hold_start, peace_candidate_track, open_palm_hold_start
    global target_stable_count, target_invalid_since, target_hist
    global auto_reacquire_until
    global lost_recovery_cmd, lost_recovery_timer_seen, lost_recovery_phase, lost_recovery_phase_until
    global pulse_active, pulse_cmd, pulse_start_time, pulse_duration, settle_until
    global cached_hand_results, unlock_indicator_until

    state = FREE
    target_track_id = None
    lost_timer = None
    lost_recovery_cmd = None
    lost_recovery_timer_seen = None
    lost_recovery_phase = "IDLE"
    lost_recovery_phase_until = 0.0
    smooth_tx = None
    smooth_height = None

    peace_hold_start = None
    peace_candidate_track = None
    open_palm_hold_start = None

    target_stable_count = 0
    target_invalid_since = None
    target_hist = None

    auto_reacquire_until = 0.0

    pulse_active = False
    pulse_cmd = None
    pulse_start_time = 0.0
    pulse_duration = 0.0
    settle_until = 0.0

    cached_hand_results = None
    unlock_indicator_until = 0.0

    send_cmd("STOP")

    if show_free_indicator:
        send_indicator("FREE")


def reset_autonomous_state_for_manual():

    reset_to_unlocked_state(show_free_indicator=True)


def enter_manual_mode():
    global manual_mode_enabled, manual_override_until, manual_active_cmd

    reset_to_unlocked_state(show_free_indicator=True)
    clear_manual_override(stop=True)

    with manual_lock:
        manual_mode_enabled = True
        manual_active_cmd = None
        manual_override_until = 0.0

    send_cmd("STOP")
    send_indicator("FREE")


def exit_manual_mode():

    global manual_mode_enabled, manual_override_until, manual_active_cmd
    global auto_reacquire_until

    clear_manual_override(stop=True)

    with manual_lock:
        manual_mode_enabled = False
        manual_active_cmd = None
        manual_override_until = 0.0

    reset_to_unlocked_state(show_free_indicator=True)
    auto_reacquire_until = 0.0
    send_cmd("STOP")
    send_indicator("FREE")


def set_manual_override(cmd: Optional[str], hold_seconds: float):
    global manual_mode_enabled, manual_override_until, manual_active_cmd

    now = time.time()

    with manual_lock:
        already_manual = manual_mode_enabled

    if not already_manual:
        reset_autonomous_state_for_manual()

    with manual_lock:
        manual_mode_enabled = True
        manual_active_cmd = cmd
        manual_override_until = now + hold_seconds

    send_cmd(cmd or "STOP")


def clear_manual_override(stop=True):
    global manual_override_until, manual_active_cmd

    with manual_lock:
        manual_active_cmd = None
        manual_override_until = 0.0

    if stop:
        send_cmd("STOP")


def apply_manual_override(now: float) -> bool:
    global manual_override_until, manual_active_cmd

    expired = False
    active_cmd = None
    manual_enabled = False

    with manual_lock:
        manual_enabled = manual_mode_enabled
        if manual_enabled and manual_active_cmd is not None and now < manual_override_until:
            active_cmd = manual_active_cmd
        elif manual_enabled and manual_active_cmd is not None and now >= manual_override_until:
            manual_active_cmd = None
            manual_override_until = 0.0
            expired = True

    if not manual_enabled:
        return False

    if active_cmd is not None:
        send_cmd(active_cmd)
        return True

    if expired or manual_enabled:
        send_cmd("STOP")

    return True


def handle_dashboard_command(cmd: Dict[str, Any]):
    global manual_speed, manual_last_command_id

    cmd_id = cmd.get("id")
    cmd_type = cmd.get("type")
    payload = cmd.get("payload") or {}

    if cmd_id == manual_last_command_id:
        return
    manual_last_command_id = cmd_id

    print(f"[DASHBOARD] Command {cmd_id}: {cmd_type} {payload}")

    try:
        if cmd_type == "manual_override":
            enter_manual_mode()
            mark_command_complete(cmd_id, "executed", "Manual override enabled; autonomous following stopped")
            return

        if cmd_type == "auto_follow":
            exit_manual_mode()
            mark_command_complete(cmd_id, "executed", "Automatic person following enabled")
            return

        if cmd_type == "move":
            direction = payload.get("direction", "forward")
            speed = int(payload.get("speed", manual_speed))
            manual_speed = clamp(speed, 0, 100)

            manual_cmd = manual_move_command(direction, manual_speed)
            if manual_cmd is None:
                raise ValueError(f"Unsupported manual direction: {direction}")

            set_manual_override(manual_cmd, MANUAL_MOVE_HOLD_SECONDS)
            mark_command_complete(
                cmd_id,
                "executed",
                f"Manual override: {direction} at speed {manual_speed} -> {manual_cmd}",
            )
            return

        if cmd_type == "rotate":
            direction = str(payload.get("direction", "clockwise")).lower()
            speed = int(payload.get("speed", manual_speed))
            manual_speed = clamp(speed, 0, 100)

            if direction in ("clockwise", "right", "cw"):
                manual_cmd = speed_to_turn_command("right", manual_speed)
            elif direction in ("counterclockwise", "counter", "left", "ccw"):
                manual_cmd = speed_to_turn_command("left", manual_speed)
            else:
                raise ValueError(f"Unsupported rotate direction: {direction}")

            set_manual_override(manual_cmd, MANUAL_MOVE_HOLD_SECONDS)
            mark_command_complete(
                cmd_id,
                "executed",
                f"Manual rotate: {direction} at speed {manual_speed} -> {manual_cmd}",
            )
            return

        if cmd_type == "speed":
            manual_speed = clamp(int(payload.get("speed", manual_speed)), 0, 100)
            mark_command_complete(cmd_id, "executed", f"Manual speed set to {manual_speed}")
            return

        if cmd_type == "stop":
            enter_manual_mode()
            clear_manual_override(stop=True)
            mark_command_complete(cmd_id, "executed", "Manual stop sent to Arduino")
            return

        # Backward compatibility with older command names.
        legacy_map = {
            "move_forward": "forward",
            "move_backward": "backward",
            "turn_left": "left",
            "turn_right": "right",
        }
        if cmd_type in legacy_map:
            direction = legacy_map[cmd_type]
            manual_cmd = manual_move_command(direction, manual_speed)
            set_manual_override(manual_cmd, MANUAL_MOVE_HOLD_SECONDS)
            mark_command_complete(cmd_id, "executed", f"Legacy manual command: {cmd_type} -> {manual_cmd}")
            return

        if cmd_type in ("ping", "custom"):
            mark_command_complete(cmd_id, "executed", "Rover dashboard command polling is alive")
            return

        if cmd_type == "camera":
            mark_command_complete(cmd_id, "executed", "Camera command received; no pan/tilt serial mapping configured")
            return

        raise ValueError(f"Unsupported command type: {cmd_type}")

    except Exception as e:
        print(f"[DASHBOARD] Command {cmd_id} failed: {e}")
        mark_command_complete(cmd_id, "failed", str(e))


def dashboard_poll_loop():
    if not dashboard_configured():
        return

    last_heartbeat_time = 0.0

    while not shutdown_event.is_set():
        now = time.time()

        commands = fetch_pending_commands()
        for cmd in commands:
            handle_dashboard_command(cmd)

        if now - last_heartbeat_time >= HEARTBEAT_INTERVAL:
            send_heartbeat()
            last_heartbeat_time = now

        shutdown_event.wait(COMMAND_POLL_INTERVAL)


def dashboard_telemetry_loop():
    if not dashboard_configured():
        return

    while not shutdown_event.is_set():
        collect_and_send_tracking_data()
        shutdown_event.wait(TELEMETRY_INTERVAL)


def open_serial():
    global ser
    candidate_ports = [
        SERIAL_PORT,
        "/dev/ttyACM0",
        "/dev/ttyACM1",
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
    ]
    for port in candidate_ports:
        try:
            ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
            time.sleep(2.5)
            print(f"[SERIAL] Connected to {port}")
            return
        except Exception:
            pass
    ser = None
    print("[SERIAL] No Arduino serial port found. Running without serial.")


def send_cmd(cmd: Optional[str]):

    global last_sent, ser, last_send_time, last_action, current_serial_movement_char

    mapping = {
        "FORWARD": "F",
        "BACKWARD": "B",
        "STRAFE_LEFT": "J",
        "STRAFE_RIGHT": "M",
        "STOP": "S",
        "ROTATE_LEFT": "L",
        "ROTATE_RIGHT": "R",
        "ROTATE_LEFT_LOW": "Q",
        "ROTATE_LEFT_MED": "W",
        "ROTATE_LEFT_HIGH": "E",
        "ROTATE_RIGHT_LOW": "I",
        "ROTATE_RIGHT_MED": "O",
        "ROTATE_RIGHT_HIGH": "P",
    }

    if cmd not in mapping:
        return

    serial_char = mapping[cmd].upper()
    last_action = cmd


    with serial_command_lock:
        current_serial_movement_char = serial_char

    if ser is None:
        return


    now = time.time()
    should_send_now = (cmd != last_sent) or ((now - last_send_time) >= SEND_REPEAT_INTERVAL)
    if not should_send_now:
        return

    last_sent = cmd
    last_send_time = now

    try:
        with serial_lock:
            ser.write(serial_char.encode("ascii"))
            ser.flush()
    except Exception as e:
        print(f"[SERIAL] Movement write error: {e}")


def serial_keepalive_send_loop():

    last_reported_char = None

    while not shutdown_event.is_set():
        try:
            with serial_command_lock:
                serial_char = current_serial_movement_char

            if ser is not None and serial_char in MOVEMENT_SERIAL_CHARS:
                with serial_lock:
                    ser.write(serial_char.encode("ascii"))
                    ser.flush()

                if serial_char != last_reported_char:
                    print(f"[SERIAL] Keeping movement command alive: {serial_char}")
                    last_reported_char = serial_char

        except Exception as e:
            print(f"[SERIAL] Keepalive write error: {e}")
            shutdown_event.wait(0.2)
            continue

        shutdown_event.wait(SERIAL_KEEPALIVE_INTERVAL)


def start_serial_keepalive_sender():
    global serial_sender_thread

    if serial_sender_thread is not None and serial_sender_thread.is_alive():
        return

    serial_sender_thread = threading.Thread(target=serial_keepalive_send_loop, daemon=True)
    serial_sender_thread.start()
    print(f"[SERIAL] Keepalive sender started ({SERIAL_KEEPALIVE_INTERVAL:.3f}s interval).")


def send_indicator(indicator: str):

    global last_indicator_sent, ser

    mapping = {
        "FREE": "A",
        "LOCKED": "K",
        "LOST": "X",
        "UNLOCK": "U",
    }

    if indicator not in mapping:
        return

    if indicator == last_indicator_sent:
        return

    last_indicator_sent = indicator

    if ser is None:
        return

    try:
        with serial_lock:
            ser.write(mapping[indicator].encode("ascii"))
            ser.flush()
    except Exception as e:
        print(f"[SERIAL] Indicator write error: {e}")


def update_indicator_state(now):

    global unlock_indicator_until

    if now < unlock_indicator_until:
        send_indicator("UNLOCK")
        return

    if state == FREE:
        send_indicator("FREE")
    elif state == LOCKED:
        send_indicator("LOCKED")
    elif state == LOST:
        send_indicator("LOST")
    elif state == DISABLED:
        send_indicator("UNLOCK")


def clamp(value, low, high):
    return max(low, min(high, value))


def get_height_based_rotate_pulse(height_px):

    if height_px is None:
        return ROTATE_PULSE_TIME_MID

    height_px = clamp(height_px, ROTATE_HEIGHT_FAR, ROTATE_HEIGHT_CLOSE)

    ratio = (height_px - ROTATE_HEIGHT_FAR) / float(ROTATE_HEIGHT_CLOSE - ROTATE_HEIGHT_FAR)
    ratio = clamp(ratio, 0.0, 1.0)

    pulse = ROTATE_PULSE_TIME_FAR + ratio * (ROTATE_PULSE_TIME_CLOSE - ROTATE_PULSE_TIME_FAR)

    return clamp(pulse, ROTATE_PULSE_TIME_FAR, ROTATE_PULSE_TIME_CLOSE)


def is_peace(hand_lms):
    lm = hand_lms.landmark
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y
    spacing = abs(lm[8].x - lm[12].x)
    return index_up and middle_up and ring_down and pinky_down and spacing > 0.02


def is_open_palm(hand_lms):
    lm = hand_lms.landmark
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y
    return sum([index_up, middle_up, ring_up, pinky_up]) >= 3


def get_hand_center(hand_lms, w, h):
    xs = [int(p.x * w) for p in hand_lms.landmark]
    ys = [int(p.y * h) for p in hand_lms.landmark]
    return sum(xs) // len(xs), sum(ys) // len(ys)


def xyxy_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def point_in_box(px, py, box):
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def get_upper_region(box, frac):
    x1, y1, x2, y2 = box
    upper_y2 = y1 + int((y2 - y1) * frac)
    return (x1, y1, x2, upper_y2)


def process_hands_image_fast(image, detector):
    if image is None or image.size == 0:
        return None

    if HAND_PROCESS_SCALE == 1.0:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        small_image = cv2.resize(
            image,
            (0, 0),
            fx=HAND_PROCESS_SCALE,
            fy=HAND_PROCESS_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )
        rgb_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

    rgb_image.flags.writeable = False
    return detector.process(rgb_image)




def map_crop_hand_results_to_full_frame(hand_results, crop_box, frame_w, frame_h):

    if hand_results is None or not hand_results.multi_hand_landmarks:
        return None

    x1, y1, x2, y2 = crop_box
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    mapped_hands = []
    for crop_hand in hand_results.multi_hand_landmarks:
        full_hand = landmark_pb2.NormalizedLandmarkList()
        for lm in crop_hand.landmark:
            new_lm = full_hand.landmark.add()
            new_lm.x = (x1 + (lm.x * crop_w)) / float(frame_w)
            new_lm.y = (y1 + (lm.y * crop_h)) / float(frame_h)
            new_lm.z = lm.z
        mapped_hands.append(full_hand)

    return SimpleNamespace(
        multi_hand_landmarks=mapped_hands,
        multi_handedness=getattr(hand_results, "multi_handedness", None),
    )




def process_hands_fast(frame, current_state, target_box=None):

    frame_h, frame_w = frame.shape[:2]

    if current_state == LOCKED and target_box is not None:

        crop_box = expand_box_to_frame(target_box, frame_w, frame_h, HAND_CROP_PADDING_FRAC)
        x1, y1, x2, y2 = crop_box

        if x2 <= x1 + 8 or y2 <= y1 + 8:
            return None

        target_crop = frame[y1:y2, x1:x2]
        crop_results = process_hands_image_fast(target_crop, hands_locked)
        return map_crop_hand_results_to_full_frame(crop_results, crop_box, frame_w, frame_h)

    detector = hands_free if current_state == FREE else hands_locked
    return process_hands_image_fast(frame, detector)


def get_intersection_ratio(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    target_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    return inter_area / float(target_area) if target_area > 0 else 0


def body_visible_enough(person_box, frame_w, frame_h):
    x1, y1, x2, y2 = person_box
    pw = max(1, x2 - x1)
    ph = max(1, y2 - y1)

    visible_x1 = max(0, x1)
    visible_y1 = max(0, y1)
    visible_x2 = min(frame_w - 1, x2)
    visible_y2 = min(frame_h - 1, y2)

    visible_w = max(0, visible_x2 - visible_x1)
    visible_h = max(0, visible_y2 - visible_y1)

    visible_area_frac = (visible_w * visible_h) / float(pw * ph)
    visible_height_frac = visible_h / float(ph)

    return (
        ph >= MIN_PERSON_HEIGHT_PX
        and visible_height_frac >= MIN_VISIBLE_HEIGHT_FRAC
        and visible_area_frac >= MIN_VISIBLE_AREA_FRAC
    )


def build_body_ratio_box(person_box):
    x1, y1, x2, y2 = person_box
    ph = max(1, y2 - y1)
    cx = (x1 + x2) // 2
    cy = int(y1 + ph * BOX_CENTER_Y_RATIO)
    side = int(ph * BOX_SIDE_RATIO)
    half = side // 2
    return (cx - half, cy - half, cx + half, cy + half)


def score_person_candidate(person_box, frame_w, frame_h):
    x1, y1, x2, y2 = person_box
    pw = max(1, x2 - x1)
    ph = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0

    center_dist = abs(cx - (frame_w / 2.0)) / (frame_w / 2.0)
    center_score = 1.0 - min(1.0, center_dist)

    size_score = min(1.0, ph / float(frame_h * 0.85))
    area_score = min(1.0, (pw * ph) / float(frame_w * frame_h * 0.35))

    return (CENTER_PREFERENCE_WEIGHT * center_score) + (SIZE_PREFERENCE_WEIGHT * max(size_score, area_score))


def clamp_box_to_frame(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    x1 = int(clamp(x1, 0, frame_w - 1))
    y1 = int(clamp(y1, 0, frame_h - 1))
    x2 = int(clamp(x2, 0, frame_w - 1))
    y2 = int(clamp(y2, 0, frame_h - 1))
    return x1, y1, x2, y2


def expand_box_to_frame(box, frame_w, frame_h, padding_frac):
 
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)
    return clamp_box_to_frame((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame_w, frame_h)


def get_overlap_metrics(boxA, boxB):

    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = max(1, areaA + areaB - inter_area)

    return inter_area / float(areaA), inter_area / float(areaB), inter_area / float(union)


def target_has_occluding_person(target_tid, target_box, person_boxes, frame_w, frame_h):

    for tid, pbox in person_boxes.items():
        if tid == target_tid:
            continue
        if not body_visible_enough(pbox, frame_w, frame_h):
            continue

        target_overlap, other_overlap, iou = get_overlap_metrics(target_box, pbox)
        if (
            target_overlap >= OCCLUSION_TARGET_OVERLAP_FRAC
            or other_overlap >= OCCLUSION_OTHER_OVERLAP_FRAC
            or iou >= OCCLUSION_IOU_FRAC
        ):
            return True

    return False


def extract_target_hist(frame, person_box):

    h, w = frame.shape[:2]
    region = get_upper_region(person_box, TARGET_HIST_UPPER_FRAC)
    x1, y1, x2, y2 = clamp_box_to_frame(region, w, h)

    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except Exception:
        return None


def compare_target_hist(reference_hist, candidate_hist):
    if reference_hist is None or candidate_hist is None:
        return 0.0
    try:
        return float(cv2.compareHist(reference_hist, candidate_hist, cv2.HISTCMP_CORREL))
    except Exception:
        return 0.0




def locked_target_identity_ok(frame, target_box, frame_w, last_tx, last_height, reference_hist):

    if last_tx is None or last_height is None or reference_hist is None:
        return True

    x1, y1, x2, y2 = target_box
    cx = (x1 + x2) / 2.0
    height = max(1, y2 - y1)

    center_jump = abs(cx - last_tx)
    height_diff = abs(height - last_height)

    max_center_jump = frame_w * IDENTITY_CENTER_JUMP_FRAC
    max_height_diff = max(1.0, last_height * IDENTITY_HEIGHT_DIFF_FRAC)

    if center_jump > max_center_jump:
        print(
            f"[ANTI-HIJACK] Reject target: center jump too large "
            f"({center_jump:.1f}px > {max_center_jump:.1f}px)"
        )
        return False

    if height_diff > max_height_diff:
        print(
            f"[ANTI-HIJACK] Reject target: height changed too much "
            f"({height_diff:.1f}px > {max_height_diff:.1f}px)"
        )
        return False

    current_hist = extract_target_hist(frame, target_box)
    hist_corr = compare_target_hist(reference_hist, current_hist)

    if hist_corr < IDENTITY_MIN_HIST_CORR:
        print(f"[ANTI-HIJACK] Reject target: appearance mismatch (hist={hist_corr:.2f})")
        return False

    return True




def update_target_histogram(frame, person_box, current_hist):

    new_hist = extract_target_hist(frame, person_box)
    if new_hist is None:
        return current_hist

    if current_hist is None:
        return new_hist

    try:
        alpha = clamp(TARGET_HIST_UPDATE_ALPHA, 0.0, 1.0)
        updated = cv2.addWeighted(new_hist, alpha, current_hist, 1.0 - alpha, 0.0)
        cv2.normalize(updated, updated, 0, 1, cv2.NORM_MINMAX)
        return updated
    except Exception:
        return current_hist


def find_locked_target_reacquire_candidate(frame, person_boxes, frame_w, frame_h, last_tx, last_height, reference_hist):

    if last_tx is None or last_height is None or reference_hist is None:
        return None

    max_center_dist = frame_w * REACQUIRE_CENTER_DISTANCE_FRAC
    max_height_diff = max(1.0, last_height * REACQUIRE_HEIGHT_DIFF_FRAC)

    candidates = []

    for tid, box in person_boxes.items():
        if not body_visible_enough(box, frame_w, frame_h):
            continue


        if target_has_occluding_person(tid, box, person_boxes, frame_w, frame_h):
            continue

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        height = max(1, y2 - y1)

        center_dist = abs(cx - last_tx)
        height_diff = abs(height - last_height)

        if center_dist > max_center_dist:
            continue
        if height_diff > max_height_diff:
            continue

        cand_hist = extract_target_hist(frame, box)
        hist_corr = compare_target_hist(reference_hist, cand_hist)
        if hist_corr < REACQUIRE_MIN_HIST_CORR:
            continue

        center_score = 1.0 - min(1.0, center_dist / max_center_dist)
        height_score = 1.0 - min(1.0, height_diff / max_height_diff)
        hist_score = max(0.0, min(1.0, hist_corr))

        score = (0.45 * center_score) + (0.25 * height_score) + (0.30 * hist_score)
        candidates.append((score, tid))

    if not candidates:
        return None

    candidates.sort(reverse=True)


    if len(candidates) >= 2:
        if (candidates[0][0] - candidates[1][0]) < REACQUIRE_SCORE_GAP:
            return None

    return candidates[0][1]

def select_best_unlocked_person(person_boxes, frame_w, frame_h):
    best_tid = None
    best_score = -1.0

    for tid, pbox in person_boxes.items():
        if not body_visible_enough(pbox, frame_w, frame_h):
            continue

        score = score_person_candidate(pbox, frame_w, frame_h)
        if score > best_score:
            best_score = score
            best_tid = tid

    return best_tid


def person_is_close_enough_for_peace_lock(person_box):
    x1, y1, x2, y2 = person_box
    ph = max(1, y2 - y1)
    return ph >= PEACE_MIN_PERSON_HEIGHT_PX


def find_person_for_hand_strict(px, py, person_boxes):

    best_tid = None
    best_dist = None

    for tid, box in person_boxes.items():
        if not point_in_box(px, py, box):
            continue

        cx, cy = xyxy_center(box)
        dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_tid = tid

    return best_tid

def open_camera():
    camera_indexes = [CAM_INDEX] + [idx for idx in CAMERA_FALLBACKS if idx != CAM_INDEX]
    for camera_index in camera_indexes:
        cam = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cam.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cam.isOpened():
            print(f"[CAMERA] Opened camera index {camera_index}")
            with opened_captures_lock:
                opened_captures.append(cam)
            return cam

        cam.release()

    raise RuntimeError(f"Failed to open camera. Tried indexes: {camera_indexes}")


class LatestFrameCamera:


    def __init__(self, capture):
        self.capture = capture
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.fail_count = 0
        self.thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self.thread.start()
        print("[CAMERA] Latest-frame reader started.")

    def stop(self, timeout: float = 2.0):

        shutdown_event.set()


        try:
            if self.capture is not None:
                self.capture.release()
        except Exception as e:
            print(f"[CAMERA] Reader capture release error: {e}")

        if self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                print("[CAMERA] Warning: camera reader thread did not exit before timeout.")

        with self.lock:
            self.frame = None

        self.capture = None

    def _read_loop(self):
        min_interval = 1.0 / CAMERA_READ_THROTTLE_FPS if CAMERA_READ_THROTTLE_FPS > 0 else 0.0

        while not shutdown_event.is_set():
            capture = self.capture
            if capture is None:
                break

            read_started = time.time()

            try:
                ret, frame = capture.read()
            except Exception as e:
                print(f"[CAMERA] Read exception: {e}")
                with self.lock:
                    self.fail_count += 1
                shutdown_event.wait(0.02)
                continue

            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.seq += 1
                    self.fail_count = 0
                update_latest_frame(frame)
                update_latest_hls_frame(frame)
            else:
                with self.lock:
                    self.fail_count += 1
                shutdown_event.wait(0.02)

            elapsed = time.time() - read_started
            delay = max(0.0, min_interval - elapsed)
            if delay > 0:
                shutdown_event.wait(delay)

        with self.lock:
            self.frame = None

    def read(self, last_seq=None, timeout=CAMERA_READ_TIMEOUT):
        deadline = time.time() + max(0.0, timeout)

        while not shutdown_event.is_set():
            with self.lock:
                if self.frame is not None and self.seq != last_seq:
                    return True, self.frame.copy(), self.seq

            if time.time() >= deadline:
                return False, None, last_seq

            shutdown_event.wait(0.005)

        return False, None, last_seq


def get_turn_command_by_section(tx, frame_w, deadzone_width):
    center_x = frame_w / 2.0
    left_deadzone = center_x - (frame_w * deadzone_width)
    right_deadzone = center_x + (frame_w * deadzone_width)

    if left_deadzone <= tx <= right_deadzone:
        return None

    if tx < left_deadzone:
        left_span = max(1.0, left_deadzone - 0.0)
        section = left_span / 3.0

        left_near_start = left_deadzone - section
        left_mid_start = left_deadzone - (2.0 * section)

        if tx >= left_near_start:
            return "ROTATE_LEFT_LOW"
        elif tx >= left_mid_start:
            return "ROTATE_LEFT_MED"
        else:
            return "ROTATE_LEFT_HIGH"

    if tx > right_deadzone:
        right_span = max(1.0, frame_w - right_deadzone)
        section = right_span / 3.0

        right_near_end = right_deadzone + section
        right_mid_end = right_deadzone + (2.0 * section)

        if tx <= right_near_end:
            return "ROTATE_RIGHT_LOW"
        elif tx <= right_mid_end:
            return "ROTATE_RIGHT_MED"
        else:
            return "ROTATE_RIGHT_HIGH"

    return None



def get_speed_status_text(now: float) -> str:
    fresh = (
        arduino_speed_mps is not None
        and (now - arduino_speed_time) <= ARDUINO_TELEMETRY_MAX_AGE
    )

    if not fresh:
        return "None"

    requirement_txt = "OK >=0.50" if arduino_speed_mps >= SPEED_REQUIREMENT_MPS else "LOW <0.50"

    return (
        f"{arduino_speed_mps:.2f} m/s "
        f"({arduino_speed_cmps:.1f} cm/s, {arduino_speed_kmh:.2f} km/h, "
        f"{requirement_txt})"
    )


def update_forward_speed_stats(now: float) -> str:
  
    global speed_forward_sample_count, speed_forward_sum_mps
    global speed_forward_max_mps, speed_forward_last_report_time

    speed_fresh = (
        arduino_speed_mps is not None
        and (now - arduino_speed_time) <= ARDUINO_TELEMETRY_MAX_AGE
    )

    valid_forward_sample = (
        state == LOCKED
        and last_action == "FORWARD"
        and speed_fresh
    )

    if valid_forward_sample:
        speed_forward_sample_count += 1
        speed_forward_sum_mps += float(arduino_speed_mps)
        speed_forward_max_mps = max(speed_forward_max_mps, float(arduino_speed_mps))

    if speed_forward_sample_count <= 0:
        return "ForwardAvg: None"

    avg_mps = speed_forward_sum_mps / speed_forward_sample_count
    avg_ok = "OK" if avg_mps >= SPEED_REQUIREMENT_MPS else "LOW"

    return (
        f"ForwardAvg: {avg_mps:.2f} m/s ({avg_ok}) | "
        f"Max: {speed_forward_max_mps:.2f} m/s | "
        f"N: {speed_forward_sample_count}"
    )


def print_status(now):
    global last_print_time
    if (now - last_print_time) < PRINT_INTERVAL:
        return
    last_print_time = now

    height_txt = "None" if smooth_height is None else str(int(smooth_height))
    target_txt = "None" if target_track_id is None else str(target_track_id)
    pulse_txt = f"{pulse_duration:.3f}s" if pulse_duration > 0 else "0.000s"
    fps_txt = f"{loop_fps_ewma:.1f}" if loop_fps_ewma > 0 else "0.0"
    speed_txt = get_speed_status_text(now)
    forward_stats_txt = update_forward_speed_stats(now)

    with manual_lock:
        mode_txt = "MANUAL" if manual_mode_enabled else "AUTO"
        manual_txt = "ON" if manual_active_cmd is not None and now < manual_override_until else "IDLE"

    print(
        f"[STATE] {state} | "
        f"Mode: {mode_txt} | "
        f"Manual: {manual_txt} | "
        f"Target: {target_txt} | "
        f"Height: {height_txt} | "
        f"Action: {last_action} | "
        f"Speed: {speed_txt} | "
        f"{forward_stats_txt} | "
        f"FPS: {fps_txt} | "
        f"Pulse: {pulse_txt} | "
        f"Stable: {target_stable_count}/{POST_LOCK_STABLE_FRAMES}"
    )



# ---------------- ARDUINO SERIAL TELEMETRY ----------------

def parse_arduino_line(line: str):

    global arduino_battery_voltage, arduino_battery_percentage, arduino_battery_time
    global arduino_dist_f, arduino_dist_l, arduino_dist_r, arduino_sensor_time
    global arduino_speed_mps, arduino_speed_cmps, arduino_speed_kmh
    global arduino_speed_rpm, arduino_speed_pulses, arduino_speed_time

    line = line.strip()
    if not line:
        return

    try:
        parts = line.split(",")
        tag = parts[0].upper()

        if tag == "BAT" and len(parts) >= 3:
            arduino_battery_voltage = float(parts[1])
            arduino_battery_percentage = int(float(parts[2]))
            arduino_battery_time = time.time()
            return

        if tag == "SENS" and len(parts) >= 4:
            arduino_dist_f = float(parts[1])
            arduino_dist_l = float(parts[2])
            arduino_dist_r = float(parts[3])
            arduino_sensor_time = time.time()
            return

        if tag == "SPD" and len(parts) >= 5:
            arduino_speed_mps = float(parts[1])
            arduino_speed_cmps = float(parts[2])
            arduino_speed_kmh = float(parts[3])
            arduino_speed_rpm = float(parts[4])

            if len(parts) >= 9:
                arduino_speed_pulses = (
                    int(float(parts[5])),
                    int(float(parts[6])),
                    int(float(parts[7])),
                    int(float(parts[8])),
                )

            arduino_speed_time = time.time()
            return

    except Exception as e:
        print(f"[SERIAL] Failed to parse Arduino line '{line}': {e}")


def arduino_serial_read_loop():

    while not shutdown_event.is_set():
        if ser is None:
            shutdown_event.wait(0.25)
            continue

        try:
            if ser.in_waiting <= 0:
                shutdown_event.wait(0.01)
                continue

            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode(errors="ignore").strip()
            parse_arduino_line(line)
        except Exception as e:
            print(f"[SERIAL] Read error: {e}")
            shutdown_event.wait(0.1)


def get_lost_strafe_command(now: float) -> Optional[str]:

    global lost_recovery_cmd, lost_recovery_timer_seen
    global lost_recovery_phase, lost_recovery_phase_until

    def reset_lost_recovery_runtime():
        global lost_recovery_cmd, lost_recovery_phase, lost_recovery_phase_until
        lost_recovery_cmd = None
        lost_recovery_phase = "IDLE"
        lost_recovery_phase_until = 0.0

    def real_distance(value: float) -> bool:
        return 2.0 <= value < 998.0

    def side_is_close(value: float) -> bool:
        return real_distance(value) and value <= LOST_SIDE_STOP_CM

    def choose_initial_side_once(left: float, right: float) -> str:
        
        left_real = real_distance(left)
        right_real = real_distance(right)

        if left_real and right_real:
            return "STRAFE_LEFT" if left <= right else "STRAFE_RIGHT"

        if left_real:
            return "STRAFE_LEFT"

        if right_real:
            return "STRAFE_RIGHT"

      
        return "STRAFE_LEFT"

    if lost_timer is None:
        lost_recovery_cmd = None
        lost_recovery_timer_seen = None
        lost_recovery_phase = "IDLE"
        lost_recovery_phase_until = 0.0
        return None


    if lost_recovery_timer_seen != lost_timer:
        lost_recovery_timer_seen = lost_timer
        reset_lost_recovery_runtime()

 
    if (now - lost_timer) < LOST_STRAFE_AFTER_SECONDS:
        return None


    if (now - arduino_sensor_time) > ARDUINO_TELEMETRY_MAX_AGE:
        print("[LOST] Ultrasonic telemetry is old. STOP.")
        return None

    if arduino_dist_l is None or arduino_dist_r is None:
        print("[LOST] Missing side ultrasonic reading. STOP.")
        return None

    left = float(arduino_dist_l)
    right = float(arduino_dist_r)


    if side_is_close(left) or side_is_close(right):
        print(
            f"[LOST] Close side detected <= {LOST_SIDE_STOP_CM:.1f} cm. STOP. "
            f"L={left:.1f} cm, R={right:.1f} cm"
        )
        lost_recovery_cmd = "STOP"
        lost_recovery_phase = "STOP"
        lost_recovery_phase_until = 0.0
        return None


    if lost_recovery_phase == "STOP" or lost_recovery_cmd == "STOP":
        return None

  
    if lost_recovery_phase == "IDLE" or lost_recovery_cmd is None:
        lost_recovery_cmd = choose_initial_side_once(left, right)
        lost_recovery_phase = "STRAFE"
        lost_recovery_phase_until = now + LOST_STRAFE_PULSE_SECONDS
        print(
            f"[LOST] Initial side chosen once: {lost_recovery_cmd} "
            f"for {LOST_STRAFE_PULSE_SECONDS:.2f}s "
            f"(L={left:.1f} cm, R={right:.1f} cm, stop <= {LOST_SIDE_STOP_CM:.1f} cm)"
        )
        return lost_recovery_cmd


    if lost_recovery_phase == "CHECK":
        if now < lost_recovery_phase_until:
            return None

        lost_recovery_phase = "STRAFE"
        lost_recovery_phase_until = now + LOST_STRAFE_PULSE_SECONDS
        print(
            f"[LOST] Check done. Continuing same side: {lost_recovery_cmd} "
            f"for {LOST_STRAFE_PULSE_SECONDS:.2f}s "
            f"(L={left:.1f} cm, R={right:.1f} cm)"
        )
        return lost_recovery_cmd


    if lost_recovery_phase == "STRAFE":
        if now < lost_recovery_phase_until:
            return lost_recovery_cmd

        lost_recovery_phase = "CHECK"
        lost_recovery_phase_until = now + LOST_STRAFE_CHECK_SECONDS
        print(
            f"[LOST] Strafe pulse finished. STOP/check for "
            f"{LOST_STRAFE_CHECK_SECONDS:.2f}s "
            f"(same side={lost_recovery_cmd}, L={left:.1f} cm, R={right:.1f} cm)"
        )
        return None


    return None


# ---------------- DASHBOARD FRAME RELAY ----------------

def update_latest_frame(frame):
    global latest_stream_frame, latest_stream_frame_seq, last_stream_frame_capture_time

    if not (frame_relay_enabled() and dashboard_configured()):
        return

    now = time.time()
    if STREAM_CAPTURE_FPS > 0:
        min_interval = 1.0 / STREAM_CAPTURE_FPS
    else:
        min_interval = 0.0

    if min_interval > 0 and (now - last_stream_frame_capture_time) < min_interval:
        return

    last_stream_frame_capture_time = now
    with latest_frame_lock:
        latest_stream_frame = frame.copy()
        latest_stream_frame_seq += 1


def encode_stream_frame(frame) -> Optional[bytes]:
    output = frame

    if STREAM_FRAME_WIDTH > 0 and output.shape[1] > STREAM_FRAME_WIDTH:
        ratio = STREAM_FRAME_WIDTH / float(output.shape[1])
        target_h = STREAM_FRAME_HEIGHT if STREAM_FRAME_HEIGHT > 0 else int(output.shape[0] * ratio)
        output = cv2.resize(output, (STREAM_FRAME_WIDTH, target_h), interpolation=cv2.INTER_AREA)
    elif STREAM_FRAME_HEIGHT > 0 and output.shape[0] > STREAM_FRAME_HEIGHT:
        ratio = STREAM_FRAME_HEIGHT / float(output.shape[0])
        target_w = int(output.shape[1] * ratio)
        output = cv2.resize(output, (target_w, STREAM_FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

    quality = int(clamp(STREAM_JPEG_QUALITY, 20, 95))
    success, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return None

    return encoded.tobytes()


def stream_publish_loop():

    global latest_stream_jpeg, latest_stream_jpeg_seq

    interval = 1.0 / STREAM_PUBLISH_FPS if STREAM_PUBLISH_FPS > 0 else 0.0
    last_logged_failure = 0.0
    last_logged_success = 0.0
    consecutive_failures = 0
    last_published_seq = -1

    while not shutdown_event.is_set():
        if not (frame_relay_enabled() and dashboard_configured()):
            shutdown_event.wait(2.0)
            continue

        with latest_frame_lock:
            frame = latest_stream_frame
            frame_seq = latest_stream_frame_seq
            cached_jpeg = latest_stream_jpeg if latest_stream_jpeg_seq == latest_stream_frame_seq else None

        if frame is None:
            shutdown_event.wait(0.005)
            continue

        if frame_seq == last_published_seq and not STREAM_RESEND_STALE_FRAMES:
            shutdown_event.wait(0.005)
            continue

        frame_bytes = cached_jpeg if cached_jpeg is not None else encode_stream_frame(frame)
        if frame_bytes is None:
            shutdown_event.wait(0.005)
            continue

        if cached_jpeg is None:
            with latest_frame_lock:
                latest_stream_jpeg = frame_bytes
                latest_stream_jpeg_seq = frame_seq

        last_published_seq = frame_seq
        upload_started = time.time()
        try:
            response = stream_session.post(
                dashboard_api_url("/rover/frame"),
                data=frame_bytes,
                headers={
                    "Authorization": f"Bearer {API_TOKEN}",
                    "X-Rover-Id": ROVER_ID,
                    "Content-Type": "image/jpeg",
                    "Accept": "application/json",
                    "Connection": "keep-alive",
                },
                timeout=(STREAM_CONNECT_TIMEOUT, STREAM_READ_TIMEOUT),
            )
            elapsed = time.time() - upload_started
            if response.status_code in (200, 201, 204):
                consecutive_failures = 0
                if time.time() - last_logged_success > 30:
                    print(f"[STREAM] Frame uploaded ({len(frame_bytes)} B in {elapsed*1000:.0f} ms)")
                    last_logged_success = time.time()
            else:
                consecutive_failures += 1
                if time.time() - last_logged_failure > 5:
                    print(f"[STREAM] Frame upload failed: {response.status_code} {response.text[:160]}")
                    last_logged_failure = time.time()
        except requests.exceptions.ReadTimeout:
            consecutive_failures += 1
            if time.time() - last_logged_failure > 5:
                print(
                    f"[STREAM] Read timeout after {STREAM_READ_TIMEOUT}s — "
                    f"upstream too slow. Lower STREAM_PUBLISH_FPS or move off ngrok-free."
                )
                last_logged_failure = time.time()
        except Exception as e:
            consecutive_failures += 1
            if time.time() - last_logged_failure > 5:
                print(f"[STREAM] Frame upload error: {e}")
                last_logged_failure = time.time()

        elapsed = time.time() - upload_started
        delay = max(0.0, interval - elapsed) if interval > 0 else 0.0
        if consecutive_failures >= 5:
            delay = max(delay, STREAM_FAILURE_BACKOFF)
        if delay > 0:
            shutdown_event.wait(delay)


def start_stream_publisher():
    if not frame_relay_enabled():
        print("[STREAM] Dashboard frame relay publisher disabled.")
        return

    threading.Thread(target=stream_publish_loop, daemon=True).start()
    if STREAM_PUBLISH_FPS > 0:
        print(f"[STREAM] Frame publisher started (target ~{STREAM_PUBLISH_FPS:.0f} fps to dashboard).")
    else:
        print("[STREAM] Frame publisher started (uncapped best-effort to dashboard).")


# ---------------- HLS VIDEO STREAM ----------------

def update_latest_hls_frame(frame):
    global latest_hls_frame, latest_hls_frame_seq, last_hls_frame_time

    if not hls_stream_enabled():
        return

    now = time.time()
    if HLS_FPS > 0:
        min_interval = 1.0 / HLS_FPS
        if (now - last_hls_frame_time) < min_interval:
            return

    last_hls_frame_time = now
    with latest_hls_frame_lock:
        latest_hls_frame = frame.copy()
        latest_hls_frame_seq += 1


def get_latest_hls_frame(timeout: float = 5.0):
    deadline = time.time() + timeout

    while not shutdown_event.is_set():
        with latest_hls_frame_lock:
            if latest_hls_frame is not None:
                return latest_hls_frame.copy()

        if time.time() >= deadline:
            return None

        shutdown_event.wait(0.02)


def build_hls_ffmpeg_command(width: int, height: int) -> list[str]:
    fps = max(1, int(HLS_FPS))
    segment_seconds = max(0.25, HLS_SEGMENT_SECONDS)
    gop = max(1, int(fps * segment_seconds))
    playlist_path = str(HLS_OUTPUT_DIR / HLS_PLAYLIST_NAME)
    segment_path = str(HLS_OUTPUT_DIR / "segment_%06d.ts")

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        HLS_ENCODER,
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        HLS_BITRATE,
        "-maxrate",
        HLS_BITRATE,
        "-bufsize",
        HLS_BITRATE,
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-f",
        "hls",
        "-hls_time",
        str(segment_seconds),
        "-hls_list_size",
        str(max(2, HLS_LIST_SIZE)),
        "-hls_flags",
        "delete_segments+omit_endlist+independent_segments",
        "-hls_segment_filename",
        segment_path,
        playlist_path,
    ]


def hls_encode_loop():
    if not hls_stream_enabled():
        return

    if shutil.which("ffmpeg") is None:
        print("[HLS] ffmpeg not found; HLS streaming disabled.")
        return

    first_frame = get_latest_hls_frame()
    if first_frame is None:
        print("[HLS] No camera frame available for HLS encoder.")
        return

    HLS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_file in HLS_OUTPUT_DIR.glob("*"):
        if stale_file.is_file():
            try:
                stale_file.unlink()
            except Exception:
                pass

    height, width = first_frame.shape[:2]
    cmd = build_hls_ffmpeg_command(width, height)

    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except Exception as e:
        print(f"[HLS] Failed to start ffmpeg: {e}")
        return

    print(f"[HLS] Encoder started ({width}x{height} @ ~{max(1, int(HLS_FPS))} fps, {HLS_BITRATE}).")
    frame_interval = 1.0 / max(1.0, HLS_FPS)
    next_frame_at = time.time()

    while not shutdown_event.is_set() and process.poll() is None:
        frame = get_latest_hls_frame(timeout=1.0)
        if frame is None:
            continue

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        try:
            process.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"[HLS] ffmpeg stdin error: {e}")
            break

        next_frame_at += frame_interval
        delay = max(0.0, next_frame_at - time.time())
        if delay > 0:
            shutdown_event.wait(delay)
        elif delay < -1.0:
            next_frame_at = time.time()

    try:
        if process.stdin:
            process.stdin.close()
    except Exception:
        pass

    try:
        process.terminate()
    except Exception:
        pass

    print("[HLS] Encoder stopped.")


def upload_hls_file(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except Exception:
        return False

    if path.suffix == ".m3u8":
        data = tune_hls_playlist_for_low_latency(data)

    try:
        response = stream_session.post(
            dashboard_api_url(f"/rover/hls/{path.name}"),
            data=data,
            headers={
                "Authorization": f"Bearer {API_TOKEN}",
                "X-Rover-Id": ROVER_ID,
                "Content-Type": "application/vnd.apple.mpegurl" if path.suffix == ".m3u8" else "video/mp2t",
                "Accept": "application/json",
                "Connection": "keep-alive",
            },
            timeout=(STREAM_CONNECT_TIMEOUT, STREAM_READ_TIMEOUT),
        )
        return response.status_code in (200, 201, 204)
    except Exception as e:
        print(f"[HLS] Upload error for {path.name}: {e}")
    return False


def tune_hls_playlist_for_low_latency(data: bytes) -> bytes:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data

    if "#EXT-X-START:" in text:
        return data

    lines = text.splitlines()
    start_hint = f"#EXT-X-START:TIME-OFFSET=-{max(0.25, HLS_SEGMENT_SECONDS):.2f},PRECISE=YES"

    if lines and lines[0].strip() == "#EXTM3U":
        lines.insert(1, start_hint)
    else:
        lines.insert(0, start_hint)

    return ("\n".join(lines) + "\n").encode("utf-8")


def hls_upload_loop():
    if not (hls_stream_enabled() and dashboard_configured()):
        return

    last_logged = 0.0

    while not shutdown_event.is_set():
        files = sorted(
            [path for path in HLS_OUTPUT_DIR.glob("*") if path.suffix in (".ts", ".m3u8", ".m4s", ".mp4")],
            key=lambda path: (path.suffix == ".m3u8", path.name),
        )

        uploaded = 0
        for path in files:
            try:
                stat = path.stat()
            except Exception:
                continue

            marker = (stat.st_mtime, stat.st_size)
            if hls_upload_state.get(path.name) == marker:
                continue

            if upload_hls_file(path):
                hls_upload_state[path.name] = marker
                uploaded += 1

        if uploaded and time.time() - last_logged > 10:
            print(f"[HLS] Uploaded {uploaded} changed HLS file(s) to dashboard.")
            last_logged = time.time()

        shutdown_event.wait(HLS_UPLOAD_POLL_INTERVAL)


def start_hls_publisher():
    if not hls_stream_enabled():
        return

    if not dashboard_configured():
        print("[HLS] Dashboard not configured; HLS publisher disabled.")
        return

    threading.Thread(target=hls_encode_loop, daemon=True).start()
    threading.Thread(target=hls_upload_loop, daemon=True).start()
    print("[HLS] HLS encoder/uploader threads started.")




def release_all_cameras():

    global cap, latest_stream_frame, latest_stream_jpeg, latest_hls_frame

    try:
        with latest_frame_lock:
            latest_stream_frame = None
            latest_stream_jpeg = None
    except Exception:
        pass

    try:
        with latest_hls_frame_lock:
            latest_hls_frame = None
    except Exception:
        pass

    released_ids = set()

    def release_capture(capture):
        if capture is None:
            return
        ident = id(capture)
        if ident in released_ids:
            return
        released_ids.add(ident)
        try:
            capture.release()
            print("[CAMERA] VideoCapture released.")
        except Exception as e:
            print(f"[CAMERA] Release error: {e}")

    release_capture(cap)
    cap = None

    with opened_captures_lock:
        captures = list(opened_captures)
        opened_captures.clear()

    for capture in captures:
        release_capture(capture)


    try:
        gc.collect()
    except Exception:
        pass

    time.sleep(1.0)


def show_camera_users():

    try:
        result = subprocess.run(
            ["bash", "-lc", "fuser -v /dev/video* 2>&1 || true"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        output = (result.stdout or "").strip()
        if output:
            print("[CAMERA] Current /dev/video* users:")
            print(output)
    except Exception as e:
        print(f"[CAMERA] Could not check camera users: {e}")


def acquire_single_instance_lock():

    global rover_lock_handle

    rover_lock_handle = open(ROVER_LOCK_FILE, "w")
    try:
        fcntl.flock(rover_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[LOCK] Another rover Python process is already running.")
        print("[LOCK] Stop the old one first, otherwise the camera will stay busy.")
        show_camera_users()
        sys.exit(1)

    rover_lock_handle.seek(0)
    rover_lock_handle.truncate()
    rover_lock_handle.write(str(os.getpid()))
    rover_lock_handle.flush()
    print(f"[LOCK] Single-instance lock acquired: {ROVER_LOCK_FILE}")


def release_single_instance_lock():
    global rover_lock_handle
    if rover_lock_handle is None:
        return
    try:
        fcntl.flock(rover_lock_handle, fcntl.LOCK_UN)
        rover_lock_handle.close()
        print("[LOCK] Single-instance lock released.")
    except Exception:
        pass
    rover_lock_handle = None


def send_serial_stop_before_close():
    global current_serial_movement_char

    with serial_command_lock:
        current_serial_movement_char = "S"

    if ser is None or not getattr(ser, "is_open", False):
        return

    try:
        with serial_lock:
            for _ in range(5):
                ser.write(b"S")
                ser.flush()
                time.sleep(0.05)
    except Exception as e:
        print(f"[SERIAL] Stop-before-close error: {e}")




def emergency_resource_cleanup():
    try:
        send_serial_stop_before_close()
    except Exception:
        pass

    try:
        release_all_cameras()
    except Exception:
        pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    try:
        if ser is not None and getattr(ser, "is_open", False):
            ser.close()
    except Exception:
        pass

    try:
        release_single_instance_lock()
    except Exception:
        pass


atexit.register(emergency_resource_cleanup)


def _handle_shutdown_signal(signum, frame):
    print(f"\n[EXIT] Received signal {signum}; cleaning up resources.")
    shutdown_event.set()
    raise KeyboardInterrupt


try:
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
except Exception:
    pass

# ---------------- STARTUP ----------------
acquire_single_instance_lock()
load_dashboard_config()
open_serial()
start_serial_keepalive_sender()
serial_reader_thread = threading.Thread(target=arduino_serial_read_loop, daemon=True)
serial_reader_thread.start()

reset_to_unlocked_state(show_free_indicator=True)

cap = open_camera()
camera_reader = None
if CAMERA_THREADED:
    camera_reader = LatestFrameCamera(cap)
    camera_reader.start()
send_indicator("FREE")
start_stream_publisher()
start_hls_publisher()

local_ip = get_local_ip()
if local_ip:
    ROVER_IP = local_ip

direct_stream_url = derive_public_stream_url() if STREAM_MODE == "direct" else None
hls_stream_url = (
    HLS_PUBLIC_URL or dashboard_public_url(f"/rover/hls/{HLS_PLAYLIST_NAME}")
) if hls_stream_enabled() else None
dashboard_stream_url = (
    direct_stream_url
    or hls_stream_url
    or ("relay://dashboard" if frame_relay_enabled() else "")
)

if dashboard_configured() and dashboard_stream_url:
    update_rover_network_info(
        dashboard_stream_url,
        local_ip or ROVER_IP or "",
        int(STREAM_PORT or "8081"),
    )

dashboard_thread = threading.Thread(target=dashboard_poll_loop, daemon=True)
dashboard_thread.start()
telemetry_thread = threading.Thread(target=dashboard_telemetry_loop, daemon=True)
telemetry_thread.start()
if dashboard_configured():
    print("[DASHBOARD] Command polling and telemetry threads started.")


# ---------------- MAIN LOOP ----------------
last_camera_seq = None
try:
    while True:
        if camera_reader is not None:
            ret, frame, last_camera_seq = camera_reader.read(last_camera_seq)
        else:
            ret, frame = cap.read()

        if not ret or frame is None:
            camera_fail_count += 1
            if camera_fail_count >= MAX_CAMERA_FAILS:
                print("[CAMERA] Too many failures, stopping.")
                break
            continue

        camera_fail_count = 0
        now = time.time()
        h, w = frame.shape[:2]
        frame_count += 1
        if last_loop_time > 0:
            loop_dt = max(0.001, now - last_loop_time)
            instant_fps = 1.0 / loop_dt
            loop_fps_ewma = instant_fps if loop_fps_ewma <= 0 else (
                (0.90 * loop_fps_ewma) + (0.10 * instant_fps)
            )
        last_loop_time = now
        if camera_reader is None:
            update_latest_frame(frame)
            update_latest_hls_frame(frame)

        if apply_manual_override(now):
            update_indicator_state(now)
            print_status(now)
            continue

        # ---------- PERSON TRACKING ----------
        if state == FREE:
            run_track_now = ((frame_count % TRACK_EVERY_N_FRAMES_FREE) == 0)
        else:
            run_track_now = ((frame_count % TRACK_EVERY_N_FRAMES_LOCKED) == 0)

        if run_track_now:
            person_boxes = {}
            result = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=PERSON_CONF,
                imgsz=YOLO_IMGSZ,
                max_det=YOLO_MAX_DET,
                tracker=TRACKER_CONFIG,
                verbose=False,
            )[0]

            boxes = result.boxes
            if boxes is not None and boxes.xyxy is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                ids = boxes.id.cpu().numpy().astype(int)
                for tid, box in zip(ids, xyxy):
                    person_boxes[int(tid)] = (box[0], box[1], box[2], box[3])

            cached_person_boxes = person_boxes
        else:
            person_boxes = cached_person_boxes.copy()

        # In FREE/unlocked state, do not automatically lock/reacquire a target.
        # Locking must happen only through the peace-sign gesture.
        if state == FREE and auto_reacquire_until:
            auto_reacquire_until = 0.0

        # ---------- GESTURES ----------
        if state == FREE and len(person_boxes) > 0:
            run_hands_now = ((frame_count % HANDS_EVERY_N_FRAMES_FREE) == 0)
        elif state == LOCKED and target_track_id in person_boxes:
            run_hands_now = ((frame_count % HANDS_EVERY_N_FRAMES_LOCKED) == 0)
        else:
            run_hands_now = False

        if run_hands_now:
            if state == LOCKED and target_track_id in person_boxes:
                cached_hand_results = process_hands_fast(
                    frame,
                    state,
                    target_box=person_boxes[target_track_id],
                )
            else:
                cached_hand_results = process_hands_fast(frame, state)

        hand_results = cached_hand_results if state in (FREE, LOCKED) else None

        frame_peace_candidate_track = None
        frame_has_target_open_palm = False
        valid_peace_candidates = []

        if hand_results is not None and hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                hx, hy = get_hand_center(hand_lms, w, h)

                if state == FREE and is_peace(hand_lms):
                    candidate_tid = find_person_for_hand_strict(hx, hy, person_boxes)

                    if candidate_tid is not None and candidate_tid in person_boxes:
                        if (
                            body_visible_enough(person_boxes[candidate_tid], w, h)
                            and person_is_close_enough_for_peace_lock(person_boxes[candidate_tid])
                        ):
                            valid_peace_candidates.append(candidate_tid)

                if state == LOCKED and is_open_palm(hand_lms) and target_track_id is not None:
                    target_box = person_boxes.get(target_track_id)
                    if target_box is not None and point_in_box(hx, hy, target_box):
                        frame_has_target_open_palm = True

        if state == FREE and valid_peace_candidates:
            unique_candidates = sorted(set(valid_peace_candidates))
            if len(unique_candidates) == 1:
                frame_peace_candidate_track = unique_candidates[0]
            else:
                # Safer behavior: if more than one person is asking to lock, do not choose randomly.
                if now - last_ambiguous_peace_print > 1.0:
                    print(f"[GESTURE] Multiple peace-sign candidates {unique_candidates}; not locking.")
                    last_ambiguous_peace_print = now
                frame_peace_candidate_track = None

        # ---------- STATE MACHINE ----------
        if state == FREE:
            smooth_height = None

            if frame_peace_candidate_track is not None:
                if peace_candidate_track != frame_peace_candidate_track:
                    peace_candidate_track = frame_peace_candidate_track
                    peace_hold_start = now
                elif peace_hold_start is not None and (now - peace_hold_start) >= LOCK_HOLD_SECONDS:
                    target_track_id = peace_candidate_track
                    state = LOCKED
                    send_indicator("LOCKED")

                    lost_timer = None
                    smooth_tx = None
                    open_palm_hold_start = None
                    smooth_height = None
                    target_stable_count = 0
                    target_invalid_since = None
                    target_hist = None
                    pulse_active = False
                    pulse_cmd = None
                    pulse_duration = 0.0
                    settle_until = 0.0
                    send_cmd("STOP")
            else:
                peace_candidate_track = None
                peace_hold_start = None

        if state == LOCKED and target_track_id is not None:
            if frame_has_target_open_palm:
                open_palm_hold_start = now if open_palm_hold_start is None else open_palm_hold_start
                if (now - open_palm_hold_start) >= UNLOCK_HOLD_SECONDS:
                    state = DISABLED
                    unlock_indicator_until = now + 1.5
                    send_indicator("UNLOCK")

                    target_track_id = None
                    open_palm_hold_start = None
                    target_stable_count = 0
                    target_invalid_since = None
                    target_hist = None
                    pulse_active = False
                    pulse_cmd = None
                    pulse_duration = 0.0
                    settle_until = 0.0
                    send_cmd("STOP")
            else:
                open_palm_hold_start = None

        # ---------- FOLLOW LOGIC ----------
        if state == LOCKED:
            target_valid = (
                target_track_id in person_boxes
                and body_visible_enough(person_boxes[target_track_id], w, h)
            )

            # During rotation, ByteTrack can briefly assign a new ID to the same
            # visible person. Before declaring LOST, try to recover the target
            # using the previous center position and height.
            if not target_valid:
                reacquired_tid = find_locked_target_reacquire_candidate(
                    frame,
                    person_boxes,
                    w,
                    h,
                    smooth_tx,
                    smooth_height,
                    target_hist,
                )

                if reacquired_tid is not None:
                    target_track_id = reacquired_tid
                    target_valid = True
                    target_invalid_since = None

            if target_valid:
                target_person_box = person_boxes[target_track_id]

                identity_ok = locked_target_identity_ok(
                    frame,
                    target_person_box,
                    w,
                    smooth_tx,
                    smooth_height,
                    target_hist,
                )

                if not identity_ok:
                    if target_invalid_since is None:
                        target_invalid_since = now

                    send_cmd("STOP")
                    pulse_active = False
                    pulse_cmd = None
                    pulse_duration = 0.0
                    settle_until = 0.0

                    if (now - target_invalid_since) >= OCCLUSION_LOST_TIME:
                        if lost_timer is None:
                            lost_timer = now
                        lost_recovery_cmd = None
                        lost_recovery_timer_seen = None
                        lost_recovery_cmd = None
                        lost_recovery_timer_seen = None
                        state = LOST
                        send_indicator("LOST")
                        target_stable_count = 0

                    update_indicator_state(now)
                    print_status(now)
                    continue

                x1, y1, x2, y2 = target_person_box
                person_height = max(1, y2 - y1)

                target_occluded = target_has_occluding_person(
                    target_track_id,
                    target_person_box,
                    person_boxes,
                    w,
                    h,
                )

                # Anti-hijack behavior: if another person overlaps/covers the
                # locked target, do not update the target ID or appearance.
                # Stop first, then enter LOST if the occlusion continues.
                if target_occluded:
                    if target_invalid_since is None:
                        target_invalid_since = now

                    send_cmd("STOP")
                    pulse_active = False
                    pulse_cmd = None
                    pulse_duration = 0.0
                    settle_until = 0.0

                    if (now - target_invalid_since) >= OCCLUSION_LOST_TIME:
                        if lost_timer is None:
                            lost_timer = now
                        lost_recovery_cmd = None
                        lost_recovery_timer_seen = None
                        state = LOST
                        send_indicator("LOST")
                        target_stable_count = 0
                else:
                    target_invalid_since = None
                    target_stable_count += 1
                    lost_timer = None
                    lost_recovery_cmd = None
                    lost_recovery_timer_seen = None

                    smooth_height = person_height if smooth_height is None else (
                        (HEIGHT_SMOOTH_ALPHA * person_height) + ((1 - HEIGHT_SMOOTH_ALPHA) * smooth_height)
                    )

                    target_hist = update_target_histogram(frame, target_person_box, target_hist)

                    lock_box = build_body_ratio_box(target_person_box)
                    control_tx, _ = xyxy_center(lock_box)

                    if target_stable_count < POST_LOCK_STABLE_FRAMES:
                        send_cmd("STOP")
                        pulse_active = False
                        pulse_cmd = None
                        pulse_duration = 0.0
                        settle_until = 0.0
                    else:
                        smooth_tx = control_tx if smooth_tx is None else (
                            (SMOOTHING_FACTOR * control_tx) + ((1 - SMOOTHING_FACTOR) * smooth_tx)
                        )

                        turn_cmd = get_turn_command_by_section(smooth_tx, w, DEADZONE_WIDTH)
                        now_t = time.time()

                        if turn_cmd is None:
                            pulse_active = False
                            pulse_cmd = None
                            pulse_duration = 0.0
                            settle_until = 0.0

                            if smooth_height is not None and smooth_height >= STOP_HEIGHT_THRESHOLD:
                                send_cmd("STOP")
                            elif smooth_height is not None and smooth_height < FORWARD_HEIGHT_THRESHOLD:
                                send_cmd("FORWARD")
                            else:
                                send_cmd("STOP")
                        else:
                            if now_t < settle_until:
                                if smooth_height is not None and smooth_height < FAR_CURVE_HEIGHT_THRESHOLD:
                                    send_cmd("FORWARD")
                                else:
                                    send_cmd("STOP")
                            elif pulse_active:
                                if (now_t - pulse_start_time) < pulse_duration:
                                    send_cmd(pulse_cmd)
                                else:
                                    send_cmd("STOP")
                                    pulse_active = False
                                    pulse_cmd = None
                                    pulse_duration = 0.0
                                    settle_until = now_t + ROTATE_SETTLE_TIME
                            else:
                                pulse_active = True
                                pulse_cmd = turn_cmd
                                pulse_start_time = now_t

                                pulse_duration = get_height_based_rotate_pulse(smooth_height)

                                send_cmd(turn_cmd)

            else:
                if target_invalid_since is None:
                    target_invalid_since = now

                was_rotating_or_settling = pulse_active or (now < settle_until)
                grace_time = ROTATION_TARGET_GRACE_TIME if was_rotating_or_settling else TARGET_INVALID_GRACE_TIME

                send_cmd("STOP")
                pulse_active = False
                pulse_cmd = None
                pulse_duration = 0.0
                settle_until = 0.0

                if (now - target_invalid_since) >= grace_time:
                    if lost_timer is None:
                        lost_timer = now
                    lost_recovery_cmd = None
                    lost_recovery_timer_seen = None
                    state = LOST
                    send_indicator("LOST")
                    target_stable_count = 0

        elif state == LOST:
            lost_strafe_cmd = get_lost_strafe_command(now)
            send_cmd(lost_strafe_cmd or "STOP")
            pulse_active = False
            pulse_cmd = None
            pulse_duration = 0.0
            settle_until = 0.0

            if target_track_id in person_boxes and body_visible_enough(person_boxes[target_track_id], w, h):
                state = LOCKED
                send_indicator("LOCKED")

                lost_timer = None
                lost_recovery_cmd = None
                lost_recovery_timer_seen = None
                smooth_tx = None
                open_palm_hold_start = None
                target_stable_count = 0
                target_invalid_since = None
            elif lost_timer is not None and now - lost_timer > LOST_TIMEOUT:
                state = FREE
                send_indicator("FREE")

                target_track_id = None
                lost_recovery_cmd = None
                lost_recovery_timer_seen = None
                open_palm_hold_start = None
                target_stable_count = 0
                target_invalid_since = None
                target_hist = None
                send_cmd("STOP")

        elif state == DISABLED:
            send_cmd("STOP")
            pulse_active = False
            pulse_cmd = None
            pulse_duration = 0.0
            settle_until = 0.0


            if now >= unlock_indicator_until:
                state = FREE
                send_indicator("FREE")
                peace_candidate_track = None
                peace_hold_start = None
                open_palm_hold_start = None
                target_stable_count = 0
                target_invalid_since = None
                target_hist = None

        update_indicator_state(now)
        print_status(now)

except KeyboardInterrupt:
    print("\n[EXIT] Stopped by user.")

finally:

    try:
        clear_manual_override(stop=True)
        send_indicator("FREE")
        send_serial_stop_before_close()
    except Exception:
        pass


    shutdown_event.set()
    try:
        if camera_reader is not None:
            camera_reader.stop(timeout=2.0)
    except Exception as e:
        print(f"[CAMERA] Camera reader stop error: {e}")

    try:
        release_all_cameras()
    except Exception as e:
        print(f"[CAMERA] Final release error: {e}")

    try:
        if serial_sender_thread is not None and serial_sender_thread.is_alive():
            serial_sender_thread.join(timeout=0.5)
    except Exception:
        pass

    try:
        if serial_reader_thread is not None and serial_reader_thread.is_alive():
            serial_reader_thread.join(timeout=0.5)
    except Exception:
        pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    try:
        if ser is not None and getattr(ser, "is_open", False):
            ser.close()
            print("[SERIAL] Serial closed.")
    except Exception:
        pass

    try:
        hands_free.close()
    except Exception:
        pass

    try:
        hands_locked.close()
    except Exception:
        pass

    try:
        release_single_instance_lock()
    except Exception:
        pass

    show_camera_users()

    print("[CLEANUP] Done.")
