const btn = document.getElementById("simulateBtn");
const notification = document.getElementById("notification");
const logPreview = document.getElementById("logPreview");
const fallScoreInput = document.getElementById("fallScore");
const scoreValue = document.getElementById("scoreValue");
const notFallMaxInput = document.getElementById("notFallMax");
const possibleFallMaxInput = document.getElementById("possibleFallMax");

function classifyScore(score, notFallMax, possibleFallMax) {
  if (score < notFallMax) {
    return {
      patient_status: "stable",
      severity: "none",
      event_type: "not_fall",
      message: "No fall detected. No action needed."
    };
  }
  if (score < possibleFallMax) {
    return {
      patient_status: "possibly_fallen",
      severity: "warning",
      event_type: "possible_fall",
      message: "This person may have fallen down, you might want to check in on them."
    };
  }
  return {
    patient_status: "unresponsive",
    severity: "critical",
    event_type: "likely_fall",
    message: "This person has highly likely fallen down, please check up on them immediately."
  };
}

function buildIncident(score, notFallMax, possibleFallMax) {
  const triage = classifyScore(score, notFallMax, possibleFallMax);
  const ts = new Date().toISOString();
  const event = {
    event_id: Math.floor(Math.random() * 100000),
    stage: "initial",
    timestamp: ts,
    room_id: "302",
    fall_score: Number(score.toFixed(2)),
    pose_state: "on_floor",
    motion_level: "none",
    patient_status: triage.patient_status,
    severity: triage.severity,
    event_type: triage.event_type,
    source: "gemma_270m_router"
  };

  const message = `Gemma 270 Notification: ${triage.message}`;
  return { event, message };
}

fallScoreInput?.addEventListener("input", () => {
  scoreValue.textContent = Number(fallScoreInput.value).toFixed(2);
});

btn?.addEventListener("click", () => {
  const score = Number(fallScoreInput?.value || 0);
  const notFallMax = Number(notFallMaxInput?.value || 0.6);
  const possibleFallMax = Number(possibleFallMaxInput?.value || 0.8);

  if (notFallMax >= possibleFallMax) {
    notification.textContent = "Invalid thresholds: 'Not fall max' must be smaller than 'Possible fall max'.";
    notification.classList.remove("hidden");
    logPreview.textContent = "Threshold configuration is invalid.";
    return;
  }

  const { event, message } = buildIncident(score, notFallMax, possibleFallMax);
  notification.textContent = message;
  notification.classList.remove("hidden");
  logPreview.textContent = JSON.stringify(event, null, 2);
});
