# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the `autotune` module within the Flight Control Prototyping Scripts — a PyQt5 GUI application for PX4 fixed-wing/multirotor rate controller tuning via system identification from flight logs.

## Setup and Running

**Install with Poetry (recommended):**
```bash
cd /home/mathieu/src/Flight_Control_Prototyping_Scripts/autotune
poetry install
poetry run python3 autotune.py
```

**Or with venv:**
```bash
python3 -m venv virtualenv-test
source virtualenv-test/bin/activate
pip3 install numpy scipy pyulog control pyqt5 pyyaml
python3 autotune.py
```

**Run simulation-based test (no GUI data needed):**
```bash
poetry run python3 simulated_autotune.py
```

## Code Quality

The root repo uses pre-commit hooks with **black** (formatting) and **isort** (imports, Black profile). Run manually:
```bash
cd /home/mathieu/src/Flight_Control_Prototyping_Scripts
poetry run black autotune/
poetry run isort autotune/
```

## Architecture

### Data flow
```
ULog Flight Log → data_extractor.py → data_selection_window.py →
system_identification.py (ARX/RLS) → pid_design.py (GMVC) →
autotune.py (Bode/step response validation) → PX4 gains
```

### Module responsibilities

- **`autotune.py`** — Main PyQt5 GUI window (~900 lines). Orchestrates the full workflow: log loading, axis/vehicle-type selection, model parameter configuration, gain computation, and result visualization via matplotlib figures embedded in Qt.

- **`data_extractor.py`** — Parses PX4 ULog binary logs using `pyulog`. Extracts and interpolates signals (rates, setpoints, actuator outputs, airspeed). Returns numpy arrays aligned to a common time base.

- **`data_selection_window.py`** — Interactive matplotlib window for selecting the maneuver time window and inspecting signal quality/coherence before running identification. Loads input/output presets via `presets.py` and lets the user add/edit/delete them through `preset_dialogs.py`.

- **`presets.py`** — Loads/saves input/output presets from the user-editable `presets.yaml` (next to the code). Seeds the file with `DEFAULT_PRESETS` on first run; falls back to defaults if the file is missing or unparseable.

- **`preset_dialogs.py`** — `PresetEditDialog`: a Qt dialog to add, edit, or delete a preset, with an old→new signal diff that switches to "create" mode when the preset is renamed.

- **`system_identification.py`** — Preprocesses signals (bias removal, filtering) and runs weighted RLS to fit an ARX model. Returns numerator/denominator polynomial coefficients.

- **`arx_rls.py`** — Core recursive least-squares implementation. Assumes ARX model: `A(q⁻¹)y(k) = q⁻ᵈ B(q⁻¹)u(k) + A(q⁻¹)e(k)`. Uses matrix-inversion-free update for efficiency.

- **`pid_design.py`** — Computes PX4-compatible P/I/D gains from identified model polynomials using **General Minimum Variance Control (GMVC)**. Inputs: ARX coefficients, sample time, rise time, damping ratio.

- **`pid_analyse_window.py`** — Secondary PyQt5 window with pole-zero plots, Bode diagrams, stability margins, and disturbance response for the tuned controller.

- **`closed_loop_sim.py`** — Simulates closed-loop behavior with the identified model and designed gains to validate stability before applying to the aircraft.

- **`simulated_autotune.py`** — End-to-end test on a synthetic 2nd-order system (no flight log needed).

### Key dependencies

| Package | Purpose |
|---------|---------|
| `control` | Transfer functions, Bode plots, pole-zero, simulation |
| `pyulog` | PX4 ULog binary flight log parsing |
| `pyqt5` | GUI framework |
| `numpy` / `scipy` | Numerical computing, signal processing |
| `pyyaml` | Read/write the user-editable `presets.yaml` |

### Sample logs for testing
Located in `logs/`: `quadrotor_sitl_jmavsim.ulg`, `quadrotor_x500.ulg`, `vtol_standard.ulg`.
