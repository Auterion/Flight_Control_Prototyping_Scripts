"""Unit tests for the pure filter core (no GUI).

Run with:
    pytest test_filter_library.py
"""

import numpy as np
import pytest
from filter_library import (
    FILTER_TYPE_IDS,
    FILTER_TYPES,
    Filter,
    FilterChain,
    frequency_response,
    group_delay_ms,
    step_response,
)

FS = 1000.0


def _mag_db_at(b, a, freq, fs=FS):
    w, mag, _ = frequency_response(b, a, fs)
    return float(np.interp(freq, w, mag))


# --- registry / coefficients ------------------------------------------------
def test_every_type_builds_finite_coefficients():
    for tid in FILTER_TYPE_IDS:
        b, a = Filter(tid).coefficients(FS)
        assert b.size >= 1 and a.size >= 1
        assert np.all(np.isfinite(b)) and np.all(np.isfinite(a))


def test_lowpass_types_pass_dc():
    for tid in ("lpf1_butter", "lpf2_butter", "lpf2_px4", "lpf1_alpha", "lpf2_crit"):
        b, a = Filter(tid, {"fc": 20.0}).coefficients(FS)
        assert _mag_db_at(b, a, 0.0) == pytest.approx(0.0, abs=0.1)


def test_highpass_types_block_dc():
    for tid in ("hpf1_alpha", "hpf1_butter", "hpf2_butter"):
        b, a = Filter(tid, {"fc": 10.0}).coefficients(FS)
        assert _mag_db_at(b, a, 0.0) < -40.0


def test_notch_attenuates_center_frequency():
    b, a = Filter("notch2", {"fc": 80.0, "bw": 30.0}).coefficients(FS)
    assert _mag_db_at(b, a, 80.0) < -20.0
    assert _mag_db_at(b, a, 0.0) == pytest.approx(0.0, abs=0.5)


# --- chain ------------------------------------------------------------------
def test_empty_chain_is_passthrough():
    b, a = FilterChain().coefficients(FS)
    assert list(b) == [1.0]
    assert list(a) == [1.0]


def test_chain_is_series_convolution():
    f1 = Filter("lpf2_butter", {"fc": 20.0})
    f2 = Filter("notch2", {"fc": 80.0, "bw": 30.0})
    b1, a1 = f1.coefficients(FS)
    b2, a2 = f2.coefficients(FS)

    b, a = FilterChain([f1, f2]).coefficients(FS)

    assert b == pytest.approx(np.convolve(b1, b2))
    assert a == pytest.approx(np.convolve(a1, a2))


def test_chain_list_operations():
    chain = FilterChain([Filter("lpf1_butter", {"fc": 10.0})])
    chain.add(Filter("lpf1_butter", {"fc": 20.0}))
    assert len(chain) == 2
    chain.replace(0, Filter("notch2", {"fc": 50.0, "bw": 5.0}))
    assert chain[0].type_id == "notch2"
    chain.remove(1)
    assert len(chain) == 1


# --- Filter instance --------------------------------------------------------
def test_defaults_filled_for_missing_params():
    f = Filter("lpf2_damped")  # no params passed
    assert set(f.params) == {"fc", "zeta"}
    assert f.params["zeta"] == FILTER_TYPES["lpf2_damped"].params[1].default


def test_unknown_type_raises():
    with pytest.raises(KeyError):
        Filter("does_not_exist")


def test_copy_is_independent():
    f = Filter("lpf2_damped", {"fc": 15.0, "zeta": 0.7})
    g = f.copy()
    g.params["fc"] = 99.0
    assert f.params["fc"] == 15.0


# --- summary / labels -------------------------------------------------------
def test_params_text_uses_short_labels():
    assert Filter("lpf1_butter", {"fc": 20.0}).params_text() == "f_c: 20 Hz"
    text = Filter("notch2", {"fc": 80.0, "bw": 30.0}).params_text()
    assert text == "f_c: 80 Hz, BW: 30 Hz"


def test_summary_joins_name_and_params():
    f = Filter("lpf2_damped", {"fc": 20.0, "zeta": 0.7})
    assert f.summary() == f"{f.name} — f_c: 20 Hz, Damping: 0.7"


# --- response helpers -------------------------------------------------------
def test_response_helpers_shapes():
    b, a = Filter("lpf2_butter", {"fc": 20.0}).coefficients(FS)
    w, mag, phase = frequency_response(b, a, FS)
    assert w.shape == mag.shape == phase.shape
    wg, gd = group_delay_ms(b, a, FS)
    assert wg.shape == gd.shape
    t, y = step_response(b, a, FS)
    assert np.squeeze(y).shape[0] == t.shape[0]


def test_group_delay_no_warning_for_highpass(recwarn):
    b, a = Filter("hpf2_butter", {"fc": 5.0}).coefficients(FS)
    group_delay_ms(b, a, FS)
    assert not [w for w in recwarn.list if "singularity" in str(w.message)]
