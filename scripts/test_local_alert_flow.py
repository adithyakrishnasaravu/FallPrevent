"""Quick local test for Gemma-style in-app alert + patient log write."""

import json
import time
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    room_id = "302"
    log_path = root / "data/patient_events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    alert_message = (
        f"Gemma 270 Alert: Fall detected in Room {room_id}. "
        "No movement observed. In-app notification triggered."
    )

    event = {
        "event_id": int(time.time()),
        "stage": "initial",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "room_id": room_id,
        "fall_score": 0.93,
        "pose_state": "on_floor",
        "motion_level": "none",
        "patient_status": "unresponsive",
        "severity": "critical",
        "event_type": "fall_unresponsive_video_only",
        "source": "gemma_270m_router",
        "notification_message": alert_message,
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\\n")

    print("IN-APP ALERT")
    print(alert_message)
    print(f"PATIENT LOG WRITTEN: {log_path}")


if __name__ == "__main__":
    main()
