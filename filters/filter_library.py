#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: filter_library.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
    Pure (numpy + scipy only) library of digital filters and a chain that
    links them in series. No GUI dependency so it can be reused directly in
    the control loop of other tools (e.g. autotune).

    A filter type is described in a data-driven registry (``FILTER_TYPES``):
    a display name, a list of parameters and a function turning
    ``(params, fs) -> (b, a)``. This keeps the UI generic: it can build the
    parameter fields and summaries from the registry alone.
"""

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy import signal

Coefficients = Tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Parameter / type description
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    """Description of a single tunable filter parameter."""

    key: str
    label: str
    default: float
    unit: str = ""
    minimum: float = 0.0
    maximum: float = 1e9
    decimals: int = 3


@dataclass(frozen=True)
class FilterType:
    """A kind of filter: how to build its coefficients and describe it."""

    type_id: str
    name: str
    params: Tuple[ParamSpec, ...]
    func: Callable[[Dict[str, float], float], Coefficients]

    def coefficients(self, params: Dict[str, float], fs: float) -> Coefficients:
        b, a = self.func(params, fs)
        return np.asarray(b, dtype=float), np.asarray(a, dtype=float)

    def params_text(self, params: Dict[str, float]) -> str:
        parts = []
        for spec in self.params:
            value = params.get(spec.key, spec.default)
            unit = f" {spec.unit}" if spec.unit else ""
            parts.append(f"{spec.label}: {_fmt(value)}{unit}")
        return ", ".join(parts)

    def summary(self, params: Dict[str, float]) -> str:
        text = self.params_text(params)
        return f"{self.name} — {text}" if text else self.name


def _fmt(value: float) -> str:
    """Compact number formatting (drops trailing zeros)."""
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Coefficient functions (ported from digital_filter_compare.py)
# Each takes a params dict and the sampling frequency fs.
# ---------------------------------------------------------------------------
def _lpf1_butter(p, fs):
    fc = p["fc"]
    gamma = np.tan(np.pi * fc / fs)
    d = gamma + 1.0
    b = [gamma / d, gamma / d]
    a = [1.0, (gamma - 1.0) / d]
    return b, a


def _lpf2_butter(p, fs):
    fc = p["fc"]
    gamma = np.tan(np.pi * fc / fs)
    gamma2 = gamma**2
    d = gamma2 + np.sqrt(2.0) * gamma + 1.0
    b = np.array([gamma2, 2.0 * gamma2, gamma2]) / d
    a = np.array([d, 2.0 * (gamma2 - 1.0), gamma2 - np.sqrt(2.0) * gamma + 1.0]) / d
    return b, a


def _lpf2_px4(p, fs):
    fc = p["fc"]
    fr = fs / fc
    ohm = np.tan(np.pi / fr)
    c = 1.0 + 2.0 * np.cos(np.pi / 4.0) * ohm + ohm**2
    b0 = ohm**2 / c
    b = [b0, 2.0 * b0, b0]
    a = [
        1.0,
        2.0 * (ohm**2 - 1.0) / c,
        (1.0 - 2.0 * np.cos(np.pi / 4.0) * ohm + ohm**2) / c,
    ]
    return b, a


def _lpf1_alpha(p, fs):
    fc = p["fc"]
    dt = 1.0 / fs
    tau = 1.0 / (2.0 * np.pi * fc)
    alpha = dt / (tau + dt)
    b = [alpha]
    a = [1.0, alpha - 1.0]
    return b, a


def _lpf2_damped(p, fs):
    fc = p["fc"]
    zeta = p["zeta"]
    t = 1.0 / fs
    wn = 2.0 * np.pi * fc
    k = wn / np.tan(wn * t / 2.0)
    k2 = k**2
    a1a = 2.0 * zeta * wn
    a2a = wn**2
    d = k2 + a1a * k + a2a
    b = np.array([a2a, 2.0 * a2a, a2a]) / d
    a = np.array([d, 2.0 * a2a - 2.0 * k2, k2 - a1a * k + a2a]) / d
    return b, a


def _lpf2_crit_damped(p, fs):
    fc = p["fc"]
    wn = 2.0 * np.pi * fc
    k = 2.0 * fs
    k2 = k**2
    a1a = 2.0 * wn
    a2a = wn**2
    d = k2 + a1a * k + a2a
    b = np.array([a2a, 2.0 * a2a, a2a]) / d
    a = np.array([d, 2.0 * a2a - 2.0 * k2, k2 - a1a * k + a2a]) / d
    return b, a


def _hpf1_alpha(p, fs):
    fc = p["fc"]
    alpha = fs / (2.0 * np.pi * fc + fs)
    b = [1.0, -1.0]
    a = [1.0 / alpha, -1.0]
    return b, a


def _hpf1_butter(p, fs):
    fc = p["fc"]
    gamma = np.tan(np.pi * fc / fs)
    b = [1.0, -1.0]
    a = [gamma + 1.0, gamma - 1.0]
    return b, a


def _hpf2_butter(p, fs):
    fc = p["fc"]
    gamma = np.tan(np.pi * fc / fs)
    gamma2 = gamma**2
    d = gamma2 + np.sqrt(2.0) * gamma + 1.0
    b = np.array([1.0, -2.0, 1.0]) / d
    a = np.array([d, 2.0 * (gamma2 - 1.0), gamma2 - np.sqrt(2.0) * gamma + 1.0]) / d
    return b, a


def _notch2(p, fs):
    fc = p["fc"]
    bw = p["bw"]
    alpha = np.tan(np.pi * bw / fs)
    beta = -np.cos(2.0 * np.pi * fc / fs)
    d = alpha + 1.0
    b = np.array([1.0, 2.0 * beta, 1.0]) / d
    a = np.array([d, 2.0 * beta, 1.0 - alpha]) / d
    return b, a


def _bandstop2_butter(p, fs):
    fc = p["fc"]
    bw = p["bw"]
    gamma = np.tan(np.pi * fc / fs)
    gamma2 = gamma**2
    d = (1.0 + gamma2) * fc + gamma * bw
    b0 = fc * (gamma2 + 1.0)
    b1 = 2.0 * fc * (gamma2 - 1.0)
    b = np.array([b0, b1, b0]) / d
    a = np.array([d, b1, (1.0 + gamma2) * fc - gamma * bw]) / d
    return b, a


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_FC = ParamSpec("fc", "f_c", 20.0, "Hz", minimum=0.01, maximum=1e6)
_FC_NOTCH = ParamSpec("fc", "f_c", 80.0, "Hz", minimum=0.01, maximum=1e6)
_BW = ParamSpec("bw", "BW", 30.0, "Hz", minimum=0.01, maximum=1e6)
_ZETA = ParamSpec("zeta", "Damping", 1.0, "", minimum=0.01, maximum=10.0)

FILTER_TYPES: Dict[str, FilterType] = {
    ft.type_id: ft
    for ft in (
        FilterType("lpf1_butter", "Butterworth LPF 1st order", (_FC,), _lpf1_butter),
        FilterType("lpf2_butter", "Butterworth LPF 2nd order", (_FC,), _lpf2_butter),
        FilterType("lpf2_px4", "PX4 LPF2p (2nd order)", (_FC,), _lpf2_px4),
        FilterType("lpf1_alpha", "LPF 1st order (alpha)", (_FC,), _lpf1_alpha),
        FilterType("lpf2_damped", "LPF 2nd order (damped)", (_FC, _ZETA), _lpf2_damped),
        FilterType(
            "lpf2_crit", "LPF 2nd order (critically damped)", (_FC,), _lpf2_crit_damped
        ),
        FilterType("hpf1_alpha", "HPF 1st order (alpha)", (_FC,), _hpf1_alpha),
        FilterType("hpf1_butter", "Butterworth HPF 1st order", (_FC,), _hpf1_butter),
        FilterType("hpf2_butter", "Butterworth HPF 2nd order", (_FC,), _hpf2_butter),
        FilterType("notch2", "Notch 2nd order", (_FC_NOTCH, _BW), _notch2),
        FilterType(
            "bandstop2_butter",
            "Butterworth band-stop 2nd order",
            (_FC_NOTCH, _BW),
            _bandstop2_butter,
        ),
    )
}

# Order used by the UI combo box.
FILTER_TYPE_IDS: List[str] = list(FILTER_TYPES.keys())


# ---------------------------------------------------------------------------
# Filter instance and chain
# ---------------------------------------------------------------------------
@dataclass
class Filter:
    """A concrete filter: a type id plus its parameter values."""

    type_id: str
    params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.type_id not in FILTER_TYPES:
            raise KeyError(f"Unknown filter type '{self.type_id}'")
        # Fill in defaults for any missing parameter.
        merged = {spec.key: spec.default for spec in self.type.params}
        merged.update(self.params)
        self.params = merged

    @property
    def type(self) -> FilterType:
        return FILTER_TYPES[self.type_id]

    @property
    def name(self) -> str:
        return self.type.name

    def coefficients(self, fs: float) -> Coefficients:
        return self.type.coefficients(self.params, fs)

    def params_text(self) -> str:
        return self.type.params_text(self.params)

    def summary(self) -> str:
        return self.type.summary(self.params)

    def copy(self) -> "Filter":
        return Filter(self.type_id, dict(self.params))


class FilterChain:
    """An ordered list of filters linked in series."""

    def __init__(self, filters: List[Filter] = None):
        self.filters: List[Filter] = list(filters) if filters else []

    # list-like helpers -----------------------------------------------------
    def __len__(self):
        return len(self.filters)

    def __iter__(self):
        return iter(self.filters)

    def __getitem__(self, index):
        return self.filters[index]

    def add(self, flt: Filter):
        self.filters.append(flt)

    def replace(self, index: int, flt: Filter):
        self.filters[index] = flt

    def remove(self, index: int):
        del self.filters[index]

    # math -------------------------------------------------------------------
    def coefficients(self, fs: float) -> Coefficients:
        """Series combination: convolve all numerators and denominators."""
        b_total = np.array([1.0])
        a_total = np.array([1.0])
        for flt in self.filters:
            b, a = flt.coefficients(fs)
            b_total = np.convolve(b_total, b)
            a_total = np.convolve(a_total, a)
        return b_total, a_total


# ---------------------------------------------------------------------------
# Response helpers (shared by the UI and any analysis code)
# ---------------------------------------------------------------------------
def frequency_response(b, a, fs, n=2048):
    """Return (freq_hz, magnitude_db, phase_deg)."""
    w, h = signal.freqz(b, a, worN=n, fs=fs)
    mag_db = 20.0 * np.log10(np.abs(h) + 1e-12)
    phase_deg = np.rad2deg(np.unwrap(np.angle(h)))
    return w, mag_db, phase_deg


def group_delay_ms(b, a, fs, n=2048):
    """Return (freq_hz, group_delay_ms)."""
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        # High-pass / band-stop chains are singular at DC (0 Hz), which we
        # discard when plotting on a log axis anyway. scipy phrases this either
        # as "singularity may be present" or "group delay is singular", plus a
        # numpy divide warning (silenced via errstate).
        warnings.filterwarnings("ignore", message=".*singular.*")
        w, gd = signal.group_delay((b, a), w=n, fs=fs)
    return w, gd / fs * 1e3


def step_response(b, a, fs):
    """Return (time_s, response)."""
    t, y = signal.dstep((b, a, 1.0 / fs))
    return t, np.squeeze(y)
