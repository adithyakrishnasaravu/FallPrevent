"""
ElderWatch Agent Loop
=====================
Event-driven fall triage pipeline:
  Loop 1: Perception (MediaPipe PoseLandmarker) — always on
  Loop 2: Fall Detection (LSTM) — always on
  Loop 3: Agentic Routing (Gemma-style JSON function calling) — event-driven

This module provides the ElderWatchAgent class that runs in real-time
on a video stream (webcam, RTSP, or file).
"""

import json
import logging
import os
import threading
import time
import urllib.request
import urllib.parse
import base64
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from train_lstm import FallDetectorLSTM

logger = logging.getLogger("elderwatch")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
DEFAULT_MODEL_PATH = "models/pose_landmarker.task"


def load_env_file(env_path: str = ".env") -> None:
    """Load KEY=VALUE pairs from a .env file into process env."""
    if not os.path.exists(env_path):
        return

    with open(env_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and os.getenv(key) is None:
                os.environ[key] = value


def normalize_us_phone(raw: str) -> str:
    """Normalize US numbers to E.164 where possible."""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if raw.strip().startswith("+") and digits:
        return "+" + digits
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return raw.strip()


def ensure_pose_model(model_path: str = DEFAULT_MODEL_PATH):
    """Download pose landmarker model if not present."""
    if os.path.exists(model_path):
        return model_path
    print(f"Downloading pose landmarker model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Done.")
    return model_path


class AlertLevel(Enum):
    NONE = "none"
    MONITOR = "monitor"
    ALERT = "alert"
    ESCALATE = "escalate"
    EMERGENCY = "emergency"


class AgentPhase(Enum):
    IDLE = "idle"
    FALL_TRIGGERED = "fall_triggered"
    PROVISIONAL_ACTION = "provisional_action"
    RECHECK = "recheck"
    FINALIZED = "finalized"


@dataclass
class PoseFrame:
    """Single frame of pose data."""

    timestamp: float
    keypoints: np.ndarray
    frame: Optional[np.ndarray] = None


@dataclass
class FeatureFrame:
    """Per-frame lightweight vision features used for event aggregation."""

    timestamp: float
    motion: float
    visible_ratio: float
    body_aspect_ratio: float
    pose_state: str
    recovery_attempt: bool


@dataclass
class FallEvent:
    """Detected fall event and downstream routing results."""

    event_id: int
    timestamp: float
    confidence: float
    pose_window: np.ndarray
    pose_summary: str
    scene_description: str = ""
    alert_level: AlertLevel = AlertLevel.ALERT
    patient_status: str = "unknown"
    severity: str = "warning"
    resolved: bool = False


@dataclass
class AgentState:
    """Current state of the monitoring agent."""

    is_running: bool = False
    phase: AgentPhase = AgentPhase.IDLE
    current_alert: AlertLevel = AlertLevel.NONE
    last_fall_event: Optional[FallEvent] = None
    last_movement_time: float = 0.0
    frames_processed: int = 0
    falls_detected: int = 0
    status_message: str = "Initializing..."


class ElderWatchAgent:
    """Main agent that orchestrates perception, detection, and routing."""

    def __init__(self, config: dict):
        self.config = config
        self.state = AgentState()

        self.pose_buffer = deque(maxlen=config["lstm"]["sequence_length"])
        fps_guess = int(config.get("agent", {}).get("expected_fps", 30))
        history_seconds = int(config.get("agent", {}).get("history_seconds", 2))
        self.feature_buffer = deque(maxlen=max(1, fps_guess * history_seconds))

        self.on_alert: Optional[Callable[[FallEvent], None]] = None
        self.on_escalation: Optional[Callable[[FallEvent], None]] = None
        self.on_scene_description: Optional[Callable[[FallEvent], None]] = None
        self.on_state_change: Optional[Callable[[AgentState], None]] = None
        self.on_frame: Optional[Callable[[np.ndarray, PoseFrame], None]] = None

        self._landmarker = None
        self._lstm_model = None
        self._device = None
        self._lock = threading.Lock()
        self._is_video_source = False
        self._event_counter = 0

        self._log_path = Path(
            self.config.get("agent", {}).get(
                "patient_log_path", "data/patient_events.jsonl"
            )
        )

    def _setup(self, is_video: bool = False):
        """Initialize ML models."""
        load_env_file()
        logger.info("Loading models...")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_video_source = is_video

        pose_model_path = self.config.get("model_paths", {}).get(
            "pose_landmarker", DEFAULT_MODEL_PATH
        )
        ensure_pose_model(pose_model_path)

        base_options = mp_python.BaseOptions(model_asset_path=pose_model_path)

        if is_video:
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self._latest_pose_result = None

            def pose_callback(result, output_image, timestamp_ms):
                self._latest_pose_result = result

            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=pose_callback,
            )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        lstm_cfg = self.config["lstm"]
        self._lstm_model = FallDetectorLSTM(
            input_size=lstm_cfg["input_size"],
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=0,
        ).to(self._device)

        model_path = self.config["model_paths"]["lstm"]
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=True)
        self._lstm_model.load_state_dict(checkpoint["model_state_dict"])
        self._lstm_model.eval()

        norm_path = f"{self.config['data']['processed_dir']}/norm_stats.npz"
        stats = np.load(norm_path)
        self._norm_mean = torch.FloatTensor(stats["mean"]).to(self._device)
        self._norm_std = torch.FloatTensor(stats["std"]).to(self._device)

        logger.info(f"Models loaded on {self._device}")

    def _extract_pose_video(self, frame: np.ndarray, timestamp_ms: int) -> PoseFrame:
        """Extract pose from a frame in VIDEO mode."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            keypoints = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                dtype=np.float32,
            )
        else:
            keypoints = np.zeros((33, 4), dtype=np.float32)

        return PoseFrame(timestamp=time.time(), keypoints=keypoints, frame=frame)

    def _extract_pose_live(self, frame: np.ndarray, timestamp_ms: int) -> PoseFrame:
        """Extract pose from a frame in LIVE_STREAM mode."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._landmarker.detect_async(mp_image, timestamp_ms)

        if (
            self._latest_pose_result is not None
            and self._latest_pose_result.pose_landmarks
            and len(self._latest_pose_result.pose_landmarks) > 0
        ):
            landmarks = self._latest_pose_result.pose_landmarks[0]
            keypoints = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                dtype=np.float32,
            )
        else:
            keypoints = np.zeros((33, 4), dtype=np.float32)

        return PoseFrame(timestamp=time.time(), keypoints=keypoints, frame=frame)

    def _extract_pose(self, frame: np.ndarray, timestamp_ms: int) -> PoseFrame:
        if self._is_video_source:
            return self._extract_pose_video(frame, timestamp_ms)
        return self._extract_pose_live(frame, timestamp_ms)

    @torch.no_grad()
    def _detect_fall(self) -> Optional[float]:
        """Run LSTM on current pose buffer."""
        seq_len = self.config["lstm"]["sequence_length"]
        if len(self.pose_buffer) < seq_len:
            return None

        window = np.array([pf.keypoints.flatten() for pf in self.pose_buffer], dtype=np.float32)
        x = torch.FloatTensor(window).unsqueeze(0).to(self._device)
        x = (x - self._norm_mean) / self._norm_std
        logit = self._lstm_model(x)
        return torch.sigmoid(logit).item()

    def _compute_pose_summary(self) -> str:
        """Summarize the recent pose window."""
        if len(self.pose_buffer) < 2:
            return "Insufficient data"

        keypoints = np.array([pf.keypoints for pf in self.pose_buffer])
        hip_y = keypoints[:, 23:25, 1].mean(axis=1)
        shoulder_y = keypoints[:, 11:13, 1].mean(axis=1)
        hip_drop = hip_y[-1] - hip_y[0]
        shoulder_drop = shoulder_y[-1] - shoulder_y[0]

        velocities = np.diff(keypoints[:, :, :2], axis=0)
        avg_velocity = np.linalg.norm(velocities, axis=-1).mean()
        final_velocity = np.linalg.norm(velocities[-1], axis=-1).mean()

        final_kp = keypoints[-1]
        visible = final_kp[:, 3] > 0.5
        xs = final_kp[:, 0][visible]
        ys = final_kp[:, 1][visible]

        if len(xs) > 0 and len(ys) > 0:
            width = xs.max() - xs.min()
            height = ys.max() - ys.min()
            aspect = width / max(height, 0.01)
        else:
            aspect = 0.0

        orientation = (
            "horizontal" if aspect > 1.35 else "upright" if aspect < 0.85 else "transitioning"
        )

        return (
            f"Hip drop={hip_drop:+.3f}, shoulder drop={shoulder_drop:+.3f}, "
            f"avg motion={avg_velocity:.4f}, final motion={final_velocity:.4f}, "
            f"body_aspect={aspect:.2f} ({orientation})."
        )

    def _extract_feature_frame(self, pose_frame: PoseFrame) -> FeatureFrame:
        """Compute low-cost visual features for event-driven triage."""
        kp = pose_frame.keypoints
        visible = kp[:, 3] > 0.5
        visible_ratio = float(np.mean(visible))

        if visible.any():
            xs = kp[:, 0][visible]
            ys = kp[:, 1][visible]
            width = float(xs.max() - xs.min())
            height = float(ys.max() - ys.min())
            aspect = width / max(height, 1e-2)
        else:
            aspect = 0.0

        motion = 0.0
        if len(self.pose_buffer) >= 2:
            recent = list(self.pose_buffer)[-2:]
            motion = float(np.linalg.norm(recent[-1].keypoints - recent[-2].keypoints))

        pose_state = "unknown"
        if aspect > 1.35:
            pose_state = "on_floor"
        elif aspect < 0.85:
            pose_state = "upright"
        else:
            pose_state = "transitioning"

        recovery_attempt = motion > float(
            self.config.get("agent", {}).get("recovery_motion_threshold", 0.03)
        )

        return FeatureFrame(
            timestamp=pose_frame.timestamp,
            motion=motion,
            visible_ratio=visible_ratio,
            body_aspect_ratio=aspect,
            pose_state=pose_state,
            recovery_attempt=recovery_attempt,
        )

    def _aggregate_context(self, fall_score: float, window_ms: int, stage: str) -> dict[str, Any]:
        """Create strict structured context for Gemma routing."""
        if self.feature_buffer:
            recent = list(self.feature_buffer)
            motion_values = np.array([f.motion for f in recent], dtype=np.float32)
            avg_motion = float(motion_values.mean())
            max_motion = float(motion_values.max())
            recovery_attempt = any(f.recovery_attempt for f in recent)
            on_floor_count = sum(1 for f in recent if f.pose_state == "on_floor")
            upright_count = sum(1 for f in recent if f.pose_state == "upright")
            pose_state = "on_floor" if on_floor_count >= upright_count else "upright"
            visible_ratio = float(np.mean([f.visible_ratio for f in recent]))
        else:
            avg_motion = 0.0
            max_motion = 0.0
            recovery_attempt = False
            pose_state = "unknown"
            visible_ratio = 0.0

        low_th = float(self.config.get("agent", {}).get("motion_low_threshold", 0.02))
        high_th = float(self.config.get("agent", {}).get("motion_high_threshold", 0.05))

        if avg_motion < low_th:
            motion_level = "none"
        elif avg_motion < high_th:
            motion_level = "low"
        else:
            motion_level = "high"

        return {
            "event": "fall_detected",
            "stage": stage,
            "fall_score": round(float(fall_score), 4),
            "window_ms": int(window_ms),
            "pose_state": pose_state,
            "motion_level": motion_level,
            "avg_motion": round(avg_motion, 5),
            "max_motion": round(max_motion, 5),
            "recovery_attempt": bool(recovery_attempt),
            "person_visible": bool(visible_ratio > 0.2),
            "visible_ratio": round(visible_ratio, 3),
            "room_id": str(self.config.get("agent", {}).get("room_id", "unknown")),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def _route_with_gemma(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Gemma decision-router contract.

        In production, replace this body with real Gemma 270M inference that returns
        only JSON matching the same schema.
        """
        fall_score = float(context["fall_score"])
        pose_state = context["pose_state"]
        motion_level = context["motion_level"]
        recovery_attempt = bool(context["recovery_attempt"])
        room_id = context["room_id"]
        bands = self.config.get("agent", {})
        not_fall_max = float(bands.get("not_fall_max_score", 0.6))
        possible_fall_max = float(bands.get("possible_fall_max_score", 0.8))
        log_non_fall = bool(bands.get("log_non_fall_events", True))

        # First pass from Gemma object: classify by configurable fall_score bands.
        if fall_score < not_fall_max:
            patient_status = "stable"
            severity = "none"
            urgency = "low"
            event_type = "not_fall"
            alert_message = (
                f"Room {room_id}: no fall action needed (score={fall_score:.2f})."
            )
        elif fall_score < possible_fall_max:
            patient_status = "possibly_fallen"
            severity = "warning"
            urgency = "medium"
            event_type = "possible_fall"
            alert_message = (
                f"Room {room_id}: this person may have fallen. "
                "Please check in on them."
            )
        else:
            patient_status = (
                "unresponsive"
                if pose_state == "on_floor" and motion_level == "none" and not recovery_attempt
                else "high_risk_fall"
            )
            severity = "critical"
            urgency = "high"
            event_type = "likely_fall"
            alert_message = (
                f"Room {room_id}: this person has highly likely fallen down. "
                "Please check up on them immediately."
            )

        actions: list[dict[str, Any]] = []
        if severity in {"critical", "high", "warning"}:
            actions.append(
                {
                    "tool": "alert_caregiver",
                    "args": {
                        "urgency": urgency,
                        "message": alert_message,
                    },
                }
            )

        if severity != "none" or log_non_fall:
            actions.append(
                {
                    "tool": "update_patient_log",
                    "args": {
                        "event_type": event_type,
                        "source": "gemma_270m_router",
                    },
                }
            )

        # Enforce log action when alerting.
        if any(a.get("tool") == "alert_caregiver" for a in actions) and not any(
            a.get("tool") == "update_patient_log" for a in actions
        ):
            actions.append(
                {
                    "tool": "update_patient_log",
                    "args": {
                        "event_type": "fall_event",
                        "source": "gemma_270m_router",
                    },
                }
            )

        return {
            "patient_status": patient_status,
            "severity": severity,
            "actions": actions,
        }

    def _validate_router_output(self, payload: Any) -> tuple[bool, str]:
        """Strict JSON schema validation for router output."""
        if not isinstance(payload, dict):
            return False, "payload must be an object"

        for field in ("patient_status", "severity", "actions"):
            if field not in payload:
                return False, f"missing field '{field}'"

        if not isinstance(payload["patient_status"], str):
            return False, "patient_status must be a string"
        if not isinstance(payload["severity"], str):
            return False, "severity must be a string"
        if not isinstance(payload["actions"], list):
            return False, "actions must be a list"

        for action in payload["actions"]:
            if not isinstance(action, dict):
                return False, "each action must be an object"
            if "tool" not in action or "args" not in action:
                return False, "each action requires tool and args"
            if action["tool"] not in {"alert_caregiver", "update_patient_log", "monitor"}:
                return False, f"unsupported tool '{action['tool']}'"
            if not isinstance(action["args"], dict):
                return False, "action args must be an object"

        return True, "ok"

    def _severity_to_alert(self, severity: str, stage: str) -> AlertLevel:
        if severity == "none":
            return AlertLevel.NONE
        if severity == "critical":
            return AlertLevel.ESCALATE if stage == "recheck" else AlertLevel.ALERT
        if severity == "high":
            return AlertLevel.ALERT
        if severity == "warning":
            return AlertLevel.MONITOR
        return AlertLevel.MONITOR

    def _tool_alert_caregiver(self, event: FallEvent, args: dict[str, Any], stage: str):
        urgency = str(args.get("urgency", "high"))
        message = str(args.get("message", "Fall detected"))
        alert_mode = str(self.config.get("agent", {}).get("alert_mode", "local")).lower()

        if alert_mode == "local":
            local_alert = AlertLevel.MONITOR if urgency in {"low", "medium"} else AlertLevel.ALERT
            logger.warning(
                "LOCAL_ALERT | event_id=%s | stage=%s | urgency=%s | %s",
                event.event_id,
                stage,
                urgency,
                message,
            )
            with self._lock:
                self.state.current_alert = local_alert
                self.state.status_message = f"LOCAL ALERT ({urgency.upper()}): {message[:72]}"
            return

        caregiver_phone = normalize_us_phone(
            str(
            self.config.get("agent", {}).get("caregiver_phone", "")
            ).strip()
        )

        if not caregiver_phone:
            logger.warning(
                "ALERT_CAREGIVER | event_id=%s | stage=%s | urgency=%s | %s | no caregiver_phone configured",
                event.event_id,
                stage,
                urgency,
                message,
            )
            return

        payload = {
            "event_id": event.event_id,
            "stage": stage,
            "urgency": urgency,
            "phone_number": caregiver_phone,
            "message": message,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        webhook_url = str(
            self.config.get("agent", {}).get("alert_webhook_url", "")
        ).strip()
        if webhook_url:
            try:
                req = urllib.request.Request(
                    webhook_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5):
                    pass
                logger.warning(
                    "ALERT_CAREGIVER_SENT | event_id=%s | stage=%s | via=webhook | phone=%s",
                    event.event_id,
                    stage,
                    caregiver_phone,
                )
                return
            except Exception as exc:
                logger.error("Webhook alert failed: %s", exc)

        # Twilio fallback: only used when credentials are present.
        sid = (
            os.getenv("TWILIO_ACCOUNT_SID", "").strip()
            or str(self.config.get("agent", {}).get("twilio_account_sid", "")).strip()
        )
        token = (
            os.getenv("TWILIO_AUTH_TOKEN", "").strip()
            or str(self.config.get("agent", {}).get("twilio_auth_token", "")).strip()
        )
        from_number = (
            os.getenv("TWILIO_FROM_NUMBER", "").strip()
            or str(self.config.get("agent", {}).get("twilio_from_number", "")).strip()
        )
        from_number = normalize_us_phone(from_number)

        if sid and token and from_number:
            twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
            form = urllib.parse.urlencode(
                {"To": caregiver_phone, "From": from_number, "Body": message}
            ).encode("utf-8")
            auth_raw = f"{sid}:{token}".encode("utf-8")
            auth = base64.b64encode(auth_raw).decode("utf-8")

            try:
                req = urllib.request.Request(
                    twilio_url,
                    data=form,
                    headers={
                        "Authorization": f"Basic {auth}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5):
                    pass
                logger.warning(
                    "ALERT_CAREGIVER_SENT | event_id=%s | stage=%s | via=twilio | phone=%s",
                    event.event_id,
                    stage,
                    caregiver_phone,
                )
                return
            except Exception as exc:
                logger.error("Twilio alert failed: %s", exc)

        logger.warning(
            "ALERT_CAREGIVER | event_id=%s | stage=%s | urgency=%s | phone=%s | delivery not configured",
            event.event_id,
            stage,
            urgency,
            caregiver_phone,
        )

    def _tool_update_patient_log(
        self,
        event: FallEvent,
        args: dict[str, Any],
        stage: str,
        context: dict[str, Any],
        router_output: dict[str, Any],
    ):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "event_id": event.event_id,
            "stage": stage,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "room_id": context.get("room_id", "unknown"),
            "fall_score": context.get("fall_score"),
            "pose_state": context.get("pose_state"),
            "motion_level": context.get("motion_level"),
            "patient_status": router_output.get("patient_status"),
            "severity": router_output.get("severity"),
            "event_type": args.get("event_type", "fall_event"),
            "source": args.get("source", "gemma_270m_router"),
        }

        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.info("PATIENT_LOG_UPDATED | event_id=%s | stage=%s", event.event_id, stage)

    def _execute_actions(
        self,
        event: FallEvent,
        stage: str,
        context: dict[str, Any],
        router_output: dict[str, Any],
    ):
        for action in router_output["actions"]:
            tool = action["tool"]
            args = action["args"]

            if tool == "alert_caregiver":
                self._tool_alert_caregiver(event, args, stage)
            elif tool == "update_patient_log":
                self._tool_update_patient_log(event, args, stage, context, router_output)
            elif tool == "monitor":
                logger.info("MONITOR | event_id=%s | stage=%s", event.event_id, stage)

    def _fallback_router_output(self) -> dict[str, Any]:
        """Safety fallback when model output is invalid."""
        return {
            "patient_status": "unknown",
            "severity": "high",
            "actions": [
                {
                    "tool": "alert_caregiver",
                    "args": {
                        "urgency": "high",
                        "message": "Fall detected; model output invalid. Escalating per safety policy.",
                    },
                },
                {
                    "tool": "update_patient_log",
                    "args": {
                        "event_type": "fall_invalid_model_output",
                        "source": "safety_fallback",
                    },
                },
            ],
        }

    def _apply_routing(self, event: FallEvent, stage: str, context: dict[str, Any]):
        """Route event, validate, execute tool actions, and update state."""
        raw_output = self._route_with_gemma(context)

        is_valid, reason = self._validate_router_output(raw_output)
        if not is_valid:
            logger.error("Invalid router output (%s). Applying fallback policy.", reason)
            raw_output = self._fallback_router_output()

        event.patient_status = raw_output["patient_status"]
        event.severity = raw_output["severity"]
        event.alert_level = self._severity_to_alert(event.severity, stage)

        event.scene_description = (
            f"Router stage={stage}; patient_status={event.patient_status}; "
            f"severity={event.severity}; context={json.dumps(context)}"
        )

        self._execute_actions(
            event=event,
            stage=stage,
            context=context,
            router_output=raw_output,
        )

        if self.on_scene_description:
            self.on_scene_description(event)

    def _handle_fall_event(self, confidence: float):
        """Initial trigger: immediate triage and provisional action."""
        self._event_counter += 1

        event = FallEvent(
            event_id=self._event_counter,
            timestamp=time.time(),
            confidence=float(confidence),
            pose_window=np.array([pf.keypoints for pf in self.pose_buffer]),
            pose_summary=self._compute_pose_summary(),
        )

        with self._lock:
            self.state.last_fall_event = event
            self.state.falls_detected += 1
            self.state.phase = AgentPhase.FALL_TRIGGERED
            self.state.status_message = f"FALL DETECTED (p={confidence:.2f})"

        trigger_window_ms = int(self.config.get("agent", {}).get("trigger_window_ms", 1200))
        context = self._aggregate_context(confidence, trigger_window_ms, stage="initial")

        with self._lock:
            self.state.phase = AgentPhase.PROVISIONAL_ACTION

        self._apply_routing(event, stage="initial", context=context)

        with self._lock:
            self.state.current_alert = event.alert_level

        if self.on_alert:
            self.on_alert(event)

        threading.Thread(target=self._run_recheck, args=(event,), daemon=True).start()

    def _run_recheck(self, event: FallEvent):
        """Micro re-check after short delay to upgrade/downgrade severity."""
        delay_s = float(self.config.get("agent", {}).get("recheck_delay_s", 1.2))
        time.sleep(delay_s)

        if event.resolved:
            return

        with self._lock:
            self.state.phase = AgentPhase.RECHECK

        window_ms = int(delay_s * 1000)
        context = self._aggregate_context(event.confidence, window_ms, stage="recheck")
        self._apply_routing(event, stage="recheck", context=context)

        with self._lock:
            self.state.current_alert = event.alert_level
            if event.alert_level == AlertLevel.MONITOR:
                self.state.status_message = "Monitoring after fall"
            elif event.alert_level in {AlertLevel.ALERT, AlertLevel.ESCALATE}:
                self.state.status_message = f"{event.alert_level.value.upper()} after recheck"
            self.state.phase = AgentPhase.FINALIZED

        if event.alert_level == AlertLevel.ESCALATE and self.on_escalation:
            self.on_escalation(event)

    def start(self, video_source=0):
        """Start the agent on a video source."""
        is_video = isinstance(video_source, str)
        self._setup(is_video=is_video)

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        threshold = self.config["lstm"]["fall_threshold"]

        self.state.is_running = True
        self.state.phase = AgentPhase.IDLE
        self.state.status_message = "Monitoring - no events"
        self.state.last_movement_time = time.time()

        logger.info("Agent started | source=%s | fps=%s", video_source, fps)

        start_time = time.time()

        try:
            while self.state.is_running:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, str):
                        logger.info("Video ended")
                        break
                    continue

                if is_video:
                    timestamp_ms = int(self.state.frames_processed * 1000 / fps)
                else:
                    timestamp_ms = int((time.time() - start_time) * 1000)

                t0 = time.perf_counter()
                pose_frame = self._extract_pose(frame, timestamp_ms)
                t1 = time.perf_counter()

                self.pose_buffer.append(pose_frame)
                self.feature_buffer.append(self._extract_feature_frame(pose_frame))
                self.state.frames_processed += 1

                fall_prob = self._detect_fall()
                t2 = time.perf_counter()

                if self.on_frame:
                    self.on_frame(frame, pose_frame)

                if (
                    fall_prob is not None
                    and fall_prob > threshold
                    and self.state.current_alert == AlertLevel.NONE
                ):
                    self._handle_fall_event(fall_prob)

                if len(self.feature_buffer) >= 1:
                    latest_motion = self.feature_buffer[-1].motion
                    move_threshold = float(
                        self.config.get("agent", {}).get("recovery_motion_threshold", 0.03)
                    )
                    if latest_motion > move_threshold:
                        self.state.last_movement_time = time.time()

                        if (
                            self.state.current_alert != AlertLevel.NONE
                            and self.state.last_fall_event is not None
                        ):
                            self.state.last_fall_event.resolved = True
                            self.state.current_alert = AlertLevel.NONE
                            self.state.phase = AgentPhase.IDLE
                            self.state.status_message = "Monitoring - no events"

                if self.state.frames_processed % 10 == 0:
                    pose_ms = (t1 - t0) * 1000
                    lstm_ms = (t2 - t1) * 1000
                    prob_str = f"{fall_prob:.3f}" if fall_prob is not None else "N/A"
                    logger.info(
                        "Frame %s | Pose: %.1fms | LSTM: %.1fms | P(fall): %s",
                        self.state.frames_processed,
                        pose_ms,
                        lstm_ms,
                        prob_str,
                    )

        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            self._landmarker.close()
            cap.release()
            self.state.is_running = False
            logger.info(
                "Agent stopped | Frames: %s | Falls: %s",
                self.state.frames_processed,
                self.state.falls_detected,
            )

    def stop(self):
        """Stop the agent loop."""
        self.state.is_running = False


def run_demo(video_source, config_path: str = "configs/config.yaml"):
    """Run agent with visual display for demo."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    agent = ElderWatchAgent(config)

    def on_alert(event: FallEvent):
        print(f"\n{'=' * 60}")
        print(f"FALL ALERT | Event #{event.event_id} | Confidence: {event.confidence:.2f}")
        print(f"Router: patient_status={event.patient_status} | severity={event.severity}")
        print(f"Summary: {event.pose_summary}")
        print(f"{'=' * 60}\n")

    def on_escalation(event: FallEvent):
        print(f"\n{'!' * 60}")
        print(f"ESCALATION | Event #{event.event_id} | No recovery observed")
        print(f"{'!' * 60}\n")

    def on_scene(event: FallEvent):
        print(f"\nScene Assessment: {event.scene_description}\n")

    def on_frame(frame, pose_frame: PoseFrame):
        annotated = frame.copy()
        status = agent.state.status_message
        alert = agent.state.current_alert
        phase = agent.state.phase.value
        color = (0, 255, 0) if alert == AlertLevel.NONE else (0, 0, 255)

        cv2.putText(
            annotated,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.putText(
            annotated,
            f"Phase: {phase}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            annotated,
            f"Frame: {agent.state.frames_processed}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        h, w = frame.shape[:2]
        kp = pose_frame.keypoints
        for i in range(33):
            if kp[i, 3] > 0.5:
                x, y = int(kp[i, 0] * w), int(kp[i, 1] * h)
                cv2.circle(annotated, (x, y), 3, (0, 255, 255), -1)

        cv2.imshow("ElderWatch", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            agent.stop()

    agent.on_alert = on_alert
    agent.on_escalation = on_escalation
    agent.on_scene_description = on_scene
    agent.on_frame = on_frame

    agent.start(video_source)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Run ElderWatch agent")
    parser.add_argument(
        "--source", default="0", help="Video source: 0 for webcam, path for file"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run_demo(source, args.config)
