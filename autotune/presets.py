"""Persistence for autotune input/output presets.

Presets map a human-readable name to the input/output ULog signals used for
system identification. They live in an external, user-editable ``presets.yaml``
file next to this module so they can be edited by hand or from the GUI without
touching the source code. If the file is missing it is seeded with the built-in
defaults below; if it is unreadable/corrupt we fall back to the defaults rather
than crashing.
"""

import os

import yaml

# Built-in presets used to seed presets.yaml on first run (and as a fallback if
# the file cannot be read). Each preset maps a name to a dict with "input" and
# "output" topics, plus optional "input_legacy"/"output_legacy" fallbacks.
DEFAULT_PRESETS = {
    "Rollrate": {
        "input": "vehicle_torque_setpoint/xyz[0].0",
        "output": "vehicle_angular_velocity/xyz[0].0",
        "input_legacy": "actuator_controls_0/control[0].0",
    },
    "Pitchrate": {
        "input": "vehicle_torque_setpoint/xyz[1].0",
        "output": "vehicle_angular_velocity/xyz[1].0",
        "input_legacy": "actuator_controls_0/control[1].0",
    },
    "Yawrate": {
        "input": "vehicle_torque_setpoint/xyz[2].0",
        "output": "vehicle_angular_velocity/xyz[2].0",
        "input_legacy": "actuator_controls_0/control[2].0",
    },
    "Rollrate(FW)": {
        "input": "vehicle_torque_setpoint/xyz[0].1",
        "output": "vehicle_angular_velocity/xyz[0].0",
        "input_legacy": "actuator_controls_1/control[0].0",
    },
    "Pitchrate(FW)": {
        "input": "vehicle_torque_setpoint/xyz[1].1",
        "output": "vehicle_angular_velocity/xyz[1].0",
        "input_legacy": "actuator_controls_1/control[1].0",
    },
    "Yawrate(FW)": {
        "input": "vehicle_torque_setpoint/xyz[2].1",
        "output": "vehicle_angular_velocity/xyz[2].0",
        "input_legacy": "actuator_controls_1/control[2].0",
    },
    "Rollrate(closed-loop)": {
        "input": "vehicle_rates_setpoint/roll.0",
        "output": "vehicle_angular_velocity/xyz[0].0",
    },
    "Pitchrate(closed-loop)": {
        "input": "vehicle_rates_setpoint/pitch.0",
        "output": "vehicle_angular_velocity/xyz[1].0",
    },
    "Yawrate(closed-loop)": {
        "input": "vehicle_rates_setpoint/yaw.0",
        "output": "vehicle_angular_velocity/xyz[2].0",
    },
    "GimbalRollRate": {
        "input": "motor_state/roll.effort_cmd.0",
        "output": "motor_angular_rates/roll.angular_rate.0",
    },
    "GimbalPitchRate": {
        "input": "motor_state/pitch.effort_cmd.0",
        "output": "motor_angular_rates/pitch.angular_rate.0",
    },
    "GimbalYawRate": {
        "input": "motor_state/yaw.effort_cmd.0",
        "output": "motor_angular_rates/yaw.angular_rate.0",
    },
    "GimbalRollAtt": {
        "input": "motor_control/motor_commands.roll.angular_rate_setpoint.0",
        "output": "attitude_info/roll.0",
    },
    "GimbalPitchAtt": {
        "input": "motor_control/motor_commands.pitch.angular_rate_setpoint.0",
        "output": "attitude_info/pitch.0",
    },
    "GimbalYawAtt": {
        "input": "motor_control/motor_commands.yaw.angular_rate_setpoint.0",
        "output": "attitude_info/yaw.0",
    },
}


def presets_file_path():
    """Return the path to presets.yaml next to this module."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.yaml")


def load_presets():
    """Load presets from presets.yaml.

    Seeds the file with DEFAULT_PRESETS if it does not exist. Falls back to a
    copy of DEFAULT_PRESETS if the file is missing or cannot be parsed.
    """
    path = presets_file_path()

    if not os.path.exists(path):
        save_presets(DEFAULT_PRESETS)
        return {k: dict(v) for k, v in DEFAULT_PRESETS.items()}

    try:
        with open(path, "r") as f:
            presets = yaml.safe_load(f)
        if not isinstance(presets, dict):
            raise ValueError("presets.yaml does not contain a mapping")
        return presets
    except (OSError, yaml.YAMLError, ValueError) as e:
        print(f"Warning: could not read {path} ({e}); using built-in defaults.")
        return {k: dict(v) for k, v in DEFAULT_PRESETS.items()}


def save_presets(presets):
    """Write presets to presets.yaml, preserving insertion order."""
    path = presets_file_path()
    try:
        with open(path, "w") as f:
            yaml.safe_dump(presets, f, sort_keys=False, default_flow_style=False)
    except OSError as e:
        print(f"Warning: could not write {path} ({e}).")
