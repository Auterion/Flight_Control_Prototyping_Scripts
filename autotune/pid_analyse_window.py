import numpy as np
import plot_style as ps
import pyqtgraph as pg
from numpy.lib.stride_tricks import sliding_window_view
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from scipy.ndimage import gaussian_filter1d


class PIDAnalyseWindow(QDialog):
    """Dialog window to show estimated step response."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Estimated Step Response")

        plots = pg.GraphicsLayoutWidget()
        plots.setMinimumSize(800, 500)
        plot = ps.addPlot(
            plots,
            0,
            0,
            title="Estimated Step Response",
            xlabel="Time [s]",
            ylabel="Step Response",
        )
        plot.showGrid(x=True, y=True, alpha=0.3)
        legend = ps.addLegend(plot)
        self.window_responses_curve = plot.plot(
            pen=ps.pen(ps.MUTED, alpha=0.2), connect="finite"
        )
        self.std_lower_curve = pg.PlotDataItem()
        self.std_upper_curve = pg.PlotDataItem()
        plot.addItem(
            pg.FillBetweenItem(
                self.std_lower_curve,
                self.std_upper_curve,
                brush=ps.color(ps.BLUE, 0.2),
            )
        )
        # The band has no legend sample: a filled square stands for it
        legend.addItem(
            pg.ScatterPlotItem(
                symbol="s", size=10, pen=None, brush=ps.color(ps.BLUE, 0.2)
            ),
            "±1 std",
        )
        self.mean_response_curve = plot.plot(
            pen=ps.pen(ps.BLUE, 2), name="Mean response"
        )
        self.step_input_curve = plot.plot(
            pen=ps.pen(ps.INK, 1, "--"), name="Step Input"
        )
        self.metrics_text = ps.CornerText(plot, "bottom-right", size="9pt")
        self.hover_readout = ps.HoverReadout(plot, "t = {:.3f} s", corner="top-left")

        layout = QVBoxLayout()
        layout.addWidget(plots)
        self.setLayout(layout)

    def generate_step_response(
        self, u: np.ndarray, y: np.ndarray, t: np.ndarray
    ) -> dict:
        """Compute and plot step response."""
        time, response, window_responses = compute_step_responses(u, y, t)
        if response is None:
            self.clearPlot()
            return {}

        metrics = compute_step_response_metrics(time, response)
        self.plotStepResponses(time, response, window_responses, metrics)
        return metrics

    def curves(self):
        return [
            self.window_responses_curve,
            self.std_lower_curve,
            self.std_upper_curve,
            self.mean_response_curve,
            self.step_input_curve,
        ]

    def clearPlot(self):
        for curve in self.curves():
            curve.setData([], [])
        self.metrics_text.setLines([])

    def plotStepResponses(self, time, response, window_responses, metrics):
        n_windows = len(window_responses)
        breaks = np.full((n_windows, 1), np.nan)
        self.window_responses_curve.setData(
            np.hstack([np.tile(time, (n_windows, 1)), breaks]).ravel(),
            np.hstack([window_responses, breaks]).ravel(),
        )

        std_resp = window_responses.std(axis=0)
        self.mean_response_curve.setData(time, response)
        self.std_lower_curve.setData(time, response - std_resp)
        self.std_upper_curve.setData(time, response + std_resp)
        self.step_input_curve.setData([time[0], 0, time[-1]], [0, 1, 1])

        self.metrics_text.setLines(
            [(f"{k}: {v:.2f}", ps.MUTED) for k, v in metrics.items() if not np.isnan(v)]
        )


def compute_step_responses(
    u: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    cutfreq: float = 25.0,
    window_duration: float = 4.0,
    min_relative_energy: float = 0.1,
):
    """Step response from u to y, estimated over sliding windows.

    Returns its time, the response estimated from all the windows together
    and the response of each window, None when the data are too short or
    have no input.

    The windows taper the response by their overlap with themselves shifted
    by its time: 0.5 s of response read 20% low in 1 s windows for a loop
    as slow as the pitch rate of a fixed-wing, 7% low in 4 s windows. Data
    shorter than window_duration are one window.

    Only the windows whose input has more than min_relative_energy times the
    energy of the most excited one are used: the others mostly hold noise.
    """
    dt = np.diff(t).mean()
    if np.isclose(dt, 0):
        return None, None, None
    fs = 1 / dt

    response_window_samples = int(0.5 * fs)
    frame_samples = min(int(window_duration * fs), len(u))
    if frame_samples < 2 * response_window_samples:
        return None, None, None
    shift = max(frame_samples // 16, 1)
    time_response = t[:response_window_samples] - t[0]

    # Sliding windows
    u_windows = sliding_window_view(u, frame_samples)[::shift]
    y_windows = sliding_window_view(y, frame_samples)[::shift]

    # Apply Hanning window
    window_func = np.hanning(frame_samples)
    u_windows = u_windows * window_func
    y_windows = y_windows * window_func

    input_energy = (u_windows**2).sum(axis=1)
    excited = input_energy > min_relative_energy * input_energy.max()
    if not excited.any():
        return None, None, None

    # Deconvolution
    impulse_response, window_impulse_responses = wiener_deconvolution(
        u_windows[excited], y_windows[excited], cutfreq, fs
    )
    step_response = impulse_response[:response_window_samples].cumsum()
    window_step_responses = window_impulse_responses[
        :, :response_window_samples
    ].cumsum(axis=1)

    # Remove outlier windows
    peaks = window_step_responses.max(axis=1)
    median_peak = np.median(peaks)
    mad = np.median(np.abs(peaks - median_peak))
    mask = np.abs(peaks - median_peak) <= 3 * mad

    return time_response, step_response, window_step_responses[mask]


def compute_step_response_metrics(time: np.ndarray, response: np.ndarray) -> dict:
    """Rise time, settling time, overshoot and steady-state error of a step
    response."""
    final_val = response[-1]

    # Rise time (10% -> 90%)
    try:
        t10 = time[np.where(response >= 0.1 * final_val)[0][0]]
        t90 = time[np.where(response >= 0.9 * final_val)[0][0]]
        rise_time = t90 - t10
    except IndexError:
        rise_time = np.nan

    # Settling time (stays within ±10% of final value till the end)
    outside_bounds = np.abs(response - final_val) > 0.1 * np.abs(final_val)
    settling_time = time[
        np.where(outside_bounds)[0][-1] + 1 if outside_bounds.any() else 0
    ]

    peak_val = np.max(response[: int(len(response) * 0.8)])  # ignore drift at end

    return {
        "rise_time (s)": rise_time,
        "settling_time (s)": settling_time,
        "overshoot (%)": (peak_val - final_val) / final_val * 100,
        # Final value vs ideal 1.0
        "steady_state_error": final_val - 1,
    }


def wiener_deconvolution(
    input_: np.ndarray,
    output: np.ndarray,
    cutoff_freq: float,
    fs: float,
    epsilon: float = 1e-5,
):
    """Wiener deconvolution on windows of input/output signals.

    Returns the impulse response of all the windows together and that of
    each window. Together, the spectra of the windows are summed before the
    division: each frequency comes from the windows that excite it, where
    the mean of the windows would be shrunk by those that do not.

    epsilon is the regularization relative to the highest input power, so
    that the response does not depend on the units of the signals. The
    response of all the windows barely depends on it: on the roll rate of
    logs/quadrotor_sitl_jmavsim.ulg, 1e-6 to 1e-4 give the same within 1%.
    Higher shrinks the response to an input whose spectrum has gaps, lower
    lets the noise through in the response of each window.
    """
    n_samples = input_.shape[1]
    n_fft = 2 ** int(np.ceil(np.log2(n_samples)))

    # Pad for FFT
    input_padded = np.pad(input_, ((0, 0), (0, n_fft - n_samples)), mode="constant")
    output_padded = np.pad(output, ((0, 0), (0, n_fft - n_samples)), mode="constant")

    # FFT
    H = np.fft.fft(input_padded, axis=-1)
    G = np.fft.fft(output_padded, axis=-1)

    snr = create_frequency_mask(n_fft, cutoff_freq, fs)
    cross_spectra = np.conj(H) * G
    input_powers = (np.conj(H) * H).real

    def impulse_response(cross_spectrum, input_power):
        # Wiener filter
        regularization = epsilon * input_power.max(axis=-1, keepdims=True) / snr
        deconv_freq = cross_spectrum / (input_power + regularization)
        return np.real(np.fft.ifft(deconv_freq, axis=-1))[..., :n_samples]

    return (
        impulse_response(cross_spectra.sum(axis=0), input_powers.sum(axis=0)),
        impulse_response(cross_spectra, input_powers),
    )


def create_frequency_mask(
    n_samples: int, cutoff_freq: float, fs: float, sigma_factor: float = 6.0
) -> np.ndarray:
    """Create a smooth low-pass Gaussian frequency mask."""
    freqs = np.fft.fftfreq(n_samples, 1 / fs)
    mask = np.exp(-0.5 * (freqs / cutoff_freq) ** 2)
    mask = gaussian_filter1d(mask, sigma=n_samples / sigma_factor)
    return np.clip(mask, 1e-3, 1.0)
