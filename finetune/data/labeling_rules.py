"""Deterministic labelling rules: a 30-minute sensor window -> a JSON work order.

This module is the "domain knowledge" of the demo. It is deliberately a rule engine
and not a model: the labels must be reproducible, auditable and reviewable in a diff.
No teacher model, no API call, no randomness.

Thresholds were calibrated on the 23.7 h telemetry snapshot (1116 windows, W=30,
stride=5) so that every class is actually represented:

    normal_operation 41 % | bearing_wear 26 % | cooling_degradation 19 % | pressure_loss 14 %
    info 41 %             | warning 43 %      | critical 16 %

That calibration is not cosmetic. With naive thresholds, ~95 % of the windows come out
"normal" and the model learns to always answer the same thing - the fine-tuning would
look successful and demonstrate nothing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

import numpy as np

# --- vocabulary (mirrored in schema.json) ------------------------------------
SEVERITIES = ("info", "warning", "critical")
ROOT_CAUSES = ("normal_operation", "cooling_degradation", "bearing_wear", "pressure_loss")
ACTION_CODES = ("NONE", "MONITOR", "CHK_COOLING", "CHK_BEARING", "CHK_SEALS", "STOP_MAINT")

SENSORS = ("temperature_c", "humidity_pct", "pressure_bar", "vibration_mm_s")

# --- thresholds ---------------------------------------------------------------
# Calibrated on the real distribution, not guessed. See the module docstring.
THRESHOLDS: Dict[str, float] = {
    "temp_slope_per15": 3.0,   # degC per 15 min
    "temp_last": 40.0,         # degC
    "vib_mean": 3.5,           # mm/s
    "vib_max": 6.28,           # mm/s
    "press_slope_per15": -0.62,  # bar per 15 min (negative = losing pressure)
    "press_min": 4.08,         # bar
}

ETA_BY_SEVERITY = {"info": 120, "warning": 60, "critical": 15}

ACTION_BY_CAUSE = {
    "normal_operation": "NONE",
    "cooling_degradation": "CHK_COOLING",
    "bearing_wear": "CHK_BEARING",
    "pressure_loss": "CHK_SEALS",
}


def _slope_per_15min(values: Sequence[float]) -> float:
    """Least-squares slope, rescaled to a 15-minute horizon.

    The window is sampled once per minute, so the raw slope is per minute.
    """
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0] * 15.0)


def window_features(window: Any) -> Dict[str, float]:
    """Summarise one 30-minute window. `window` is a pandas DataFrame."""
    temp = window["temperature_c"].to_numpy(dtype=float)
    hum = window["humidity_pct"].to_numpy(dtype=float)
    press = window["pressure_bar"].to_numpy(dtype=float)
    vib = window["vibration_mm_s"].to_numpy(dtype=float)

    return {
        "temp_last": float(temp[-1]),
        "temp_max": float(temp.max()),
        "temp_slope_per15": _slope_per_15min(temp),
        "hum_last": float(hum[-1]),
        "press_last": float(press[-1]),
        "press_min": float(press.min()),
        "press_slope_per15": _slope_per_15min(press),
        "vib_mean": float(vib.mean()),
        "vib_max": float(vib.max()),
        "vib_std": float(vib.std()),
    }


def label_window(device_id: str, features: Dict[str, float]) -> Dict[str, object]:
    """Apply the rules and return the work order as a plain dict."""
    t = THRESHOLDS

    cooling = features["temp_slope_per15"] >= t["temp_slope_per15"] or features["temp_last"] >= t["temp_last"]
    bearing = features["vib_mean"] >= t["vib_mean"] or features["vib_max"] >= t["vib_max"]
    pressure = features["press_slope_per15"] <= t["press_slope_per15"] or features["press_min"] <= t["press_min"]

    triggered = sum([cooling, bearing, pressure])

    if triggered == 0:
        root_cause = "normal_operation"
        severity = "info"
    else:
        # Fixed precedence: a pressure loss is the most immediately dangerous,
        # then mechanical wear, then thermal drift. Stated explicitly so that a
        # reviewer can challenge the ordering rather than reverse-engineer it.
        if pressure:
            root_cause = "pressure_loss"
        elif bearing:
            root_cause = "bearing_wear"
        else:
            root_cause = "cooling_degradation"
        severity = "critical" if triggered >= 2 else "warning"

    if severity == "critical":
        action_code = "STOP_MAINT"
    elif root_cause == "normal_operation":
        action_code = "NONE"
    else:
        action_code = ACTION_BY_CAUSE[root_cause]

    return {
        "device_id": device_id,
        "severity": severity,
        "root_cause": root_cause,
        "action_code": action_code,
        "eta_minutes": ETA_BY_SEVERITY[severity],
    }


def label_to_text(label: Dict[str, object]) -> str:
    """Serialise the work order exactly as the model must reproduce it.

    Key order and separators are fixed: the model is learning a surface form, so
    that surface form has to be stable across every single training example.
    """
    ordered = {
        "device_id": label["device_id"],
        "severity": label["severity"],
        "root_cause": label["root_cause"],
        "action_code": label["action_code"],
        "eta_minutes": label["eta_minutes"],
    }
    return json.dumps(ordered, separators=(", ", ": "))
