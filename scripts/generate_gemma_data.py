"""
Gemma 270M Fine-tuning Data Generator
======================================
Generates structured function-calling training examples from URFall metadata.
Each fall event produces 3-5 examples covering different scenarios.

Output: JSONL file suitable for Gemma fine-tuning with PEFT/LoRA.
"""

import json
import random
import argparse
import numpy as np
from pathlib import Path


# ─── Tool Definitions (match the agent's API surface) ───────────────

TOOLS = {
    "call_emergency": {
        "description": "Call emergency services",
        "params": ["confidence", "location"],
    },
    "alert_caregiver": {
        "description": "Send alert to caregiver",
        "params": ["urgency", "confidence", "pose_summary"],
    },
    "monitor_closely": {
        "description": "Increase monitoring frequency",
        "params": ["reason"],
    },
    "log_incident": {
        "description": "Log incident for records",
        "params": ["severity", "description"],
    },
    "escalate": {
        "description": "Escalate — person hasn't moved",
        "params": ["reason"],
    },
}


# ─── Scenario Templates ────────────────────────────────────────────

def generate_high_confidence_fall(seq_id: str, confidence: float):
    """High confidence fall → immediate caregiver alert."""
    pose_summary = random.choice([
        f"Rapid vertical drop detected in hip/shoulder keypoints. "
        f"Body transitioned from upright to horizontal in {random.randint(8,15)} frames. "
        f"Subject now stationary on floor. Aspect ratio inverted.",

        f"Sudden velocity spike in torso keypoints followed by complete stillness. "
        f"Hip Y-coordinate dropped from {random.uniform(0.3,0.5):.2f} to {random.uniform(0.7,0.9):.2f}. "
        f"Subject horizontal, no movement detected.",

        f"Fall signature detected: shoulder keypoints descended {random.uniform(0.3,0.5):.2f} units "
        f"in {random.randint(5,12)} frames. Body now prone. Limbs stationary.",
    ])

    return {
        "input": f"<fall_event>\n"
                 f"  sequence: {seq_id}\n"
                 f"  confidence: {confidence:.2f}\n"
                 f"  pose_summary: {pose_summary}\n"
                 f"  seconds_since_fall: 0\n"
                 f"</fall_event>",
        "output": f'alert_caregiver(urgency="high", confidence={confidence:.2f}, '
                  f'pose_summary="{pose_summary[:80]}...")',
        "tool": "alert_caregiver",
    }


def generate_low_confidence_detection(seq_id: str, confidence: float):
    """Low confidence → monitor closely instead of alerting."""
    reasons = [
        "Gradual descent detected — could be sitting down slowly",
        "Partial pose loss during transition — person may be bending",
        "Moderate velocity change but controlled — possible intentional movement",
        "Hip keypoints lowered but shoulders stable — likely crouching",
    ]
    pose_summary = random.choice([
        f"Moderate vertical change in hip keypoints. Movement appears controlled. "
        f"Velocity below fall threshold but above normal.",
        f"Body lowering detected but at controlled rate. "
        f"Shoulder-hip angle changing gradually. Not consistent with sudden fall.",
    ])

    return {
        "input": f"<fall_event>\n"
                 f"  sequence: {seq_id}\n"
                 f"  confidence: {confidence:.2f}\n"
                 f"  pose_summary: {pose_summary}\n"
                 f"  seconds_since_fall: 0\n"
                 f"</fall_event>",
        "output": f'monitor_closely(reason="{random.choice(reasons)}")',
        "tool": "monitor_closely",
    }


def generate_escalation(seq_id: str, confidence: float):
    """Person hasn't moved after fall → escalate."""
    seconds = random.choice([60, 90, 120])
    reasons = [
        f"No movement detected for {seconds} seconds after fall event",
        f"Subject remains stationary {seconds}s post-fall. No limb movement detected.",
        f"Stillness threshold exceeded. {seconds}s since last detected movement.",
    ]

    return {
        "input": f"<fall_event>\n"
                 f"  sequence: {seq_id}\n"
                 f"  confidence: {confidence:.2f}\n"
                 f"  pose_summary: Subject stationary on floor. No keypoint movement.\n"
                 f"  seconds_since_fall: {seconds}\n"
                 f"</fall_event>",
        "output": f'escalate(reason="{random.choice(reasons)}")',
        "tool": "escalate",
    }


def generate_emergency_call(seq_id: str, confidence: float):
    """Very high confidence + no movement for extended time → emergency."""
    seconds = random.choice([180, 240, 300])

    return {
        "input": f"<fall_event>\n"
                 f"  sequence: {seq_id}\n"
                 f"  confidence: {confidence:.2f}\n"
                 f"  pose_summary: Subject prone, no movement for {seconds}s. "
                 f"Caregiver alert sent {seconds - 60}s ago with no response.\n"
                 f"  seconds_since_fall: {seconds}\n"
                 f"  caregiver_responded: false\n"
                 f"</fall_event>",
        "output": f'call_emergency(confidence={confidence:.2f}, location="primary_room")',
        "tool": "call_emergency",
    }


def generate_incident_log(seq_id: str, vlm_description: str):
    """Log incident with VLM description."""
    severities = {
        "high": ["person is prone", "unusual angle", "no movement"],
        "medium": ["person on floor", "attempting to move", "conscious"],
        "low": ["person sitting on floor", "appears alert", "moving"],
    }

    severity = random.choice(["high", "medium", "low"])

    return {
        "input": f"<vlm_assessment>\n"
                 f"  sequence: {seq_id}\n"
                 f"  description: {vlm_description}\n"
                 f"</vlm_assessment>",
        "output": f'log_incident(severity="{severity}", '
                  f'description="{vlm_description[:100]}...")',
        "tool": "log_incident",
    }


# ─── VLM Scene Descriptions (for log_incident training) ────────────

SCENE_DESCRIPTIONS = [
    "Person is prone on floor, facing ceiling. Left arm extended at unusual angle. "
    "Coffee table within 30cm of head. No movement detected since fall.",

    "Person lying on right side near bathroom doorway. Eyes appear open. "
    "Right hand gripping door frame. Slight leg movement detected.",

    "Person sitting on floor against wall. Appears conscious and alert. "
    "Walker tipped over 1m away. Person reaching toward walker.",

    "Person face down on carpet. No visible movement. Glasses displaced "
    "approximately 50cm from head. Chair overturned nearby.",

    "Person on floor in kitchen. Left leg bent at unusual angle. "
    "Spilled liquid visible on floor near fall location. No movement for 30s.",

    "Person lying on back in hallway. Eyes closed. Arms at sides. "
    "No environmental hazards visible. Breathing movement detected in chest.",

    "Person partially on couch, partially on floor. Appears to have "
    "slid off seating surface. Attempting to push up with right arm.",

    "Person on floor near bed. Tangled in bedsheet. Moving arms but "
    "unable to stand. Nightstand displaced. Lamp on floor.",
]


def generate_all_examples(num_fall_sequences: int = 30,
                           num_adl_sequences: int = 40,
                           seed: int = 42) -> list[dict]:
    """Generate full training set."""
    random.seed(seed)
    np.random.seed(seed)
    examples = []

    # From fall sequences
    for i in range(1, num_fall_sequences + 1):
        seq_id = f"fall-{i:02d}"
        confidence = np.clip(np.random.normal(0.88, 0.08), 0.6, 0.99)

        # High confidence fall → alert
        examples.append(generate_high_confidence_fall(seq_id, confidence))

        # Escalation scenario
        examples.append(generate_escalation(seq_id, confidence))

        # Emergency (subset)
        if random.random() < 0.4:
            examples.append(generate_emergency_call(seq_id, confidence))

        # VLM log
        desc = random.choice(SCENE_DESCRIPTIONS)
        examples.append(generate_incident_log(seq_id, desc))

        # Sometimes a borderline detection
        if random.random() < 0.3:
            low_conf = np.clip(np.random.normal(0.55, 0.1), 0.4, 0.69)
            examples.append(
                generate_low_confidence_detection(seq_id, low_conf)
            )

    # From ADL sequences — these should NOT trigger alerts
    for i in range(1, num_adl_sequences + 1):
        seq_id = f"adl-{i:02d}"
        # Low confidence "detections" from normal activities
        if random.random() < 0.5:
            low_conf = np.clip(np.random.normal(0.45, 0.1), 0.3, 0.65)
            examples.append(
                generate_low_confidence_detection(seq_id, low_conf)
            )

    random.shuffle(examples)
    return examples


def format_for_finetuning(examples: list[dict]) -> list[dict]:
    """
    Format examples for Gemma fine-tuning.
    Uses a structured prompt template.
    """
    system_prompt = (
        "You are a fall detection agent controller. Given a fall event or "
        "assessment, call the appropriate function. Available functions: "
        "call_emergency, alert_caregiver, monitor_closely, log_incident, escalate. "
        "Respond with exactly one function call."
    )

    formatted = []
    for ex in examples:
        formatted.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["input"]},
                {"role": "assistant", "content": ex["output"]},
            ],
            "tool_used": ex["tool"],
        })
    return formatted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Gemma 270M fine-tuning data"
    )
    parser.add_argument("--output", default="data/gemma_270m_finetune.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = generate_all_examples(seed=args.seed)
    formatted = format_for_finetuning(examples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")

    # Stats
    tool_counts = {}
    for item in formatted:
        tool = item["tool_used"]
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print(f"Generated {len(formatted)} training examples → {output_path}")
    print(f"\nTool distribution:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")
