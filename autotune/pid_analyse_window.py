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
        time, responses = compute_step_responses(u, y, t)
        if responses is None:
            self.clearPlot()
            return {}

        metrics = compute_step_response_metrics(time, responses)
        self.plotStepResponses(time, responses, metrics)
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

    def plotStepResponses(self, time, responses, metrics):
        n_windows = len(responses)
        breaks = np.full((n_windows, 1), np.nan)
        self.window_responses_curve.setData(
            np.hstack([np.tile(time, (n_windows, 1)), breaks]).ravel(),
            np.hstack([responses, breaks]).ravel(),
        )

        mean_resp = responses.mean(axis=0)
        std_resp = responses.std(axis=0)
        self.mean_response_curve.setData(time, mean_resp)
        self.std_lower_curve.setData(time, mean_resp - std_resp)
        self.std_upper_curve.setData(time, mean_resp + std_resp)
        self.step_input_curve.setData([time[0], 0, time[-1]], [0, 1, 1])

        self.metrics_text.setLines(
            [
                (f"{k}: {v[0]:.2f} ± {v[1]:.2f}", ps.MUTED)
                for k, v in metrics.items()
                if not np.isnan(v[0])
            ]
        )


def compute_step_responses(
    u: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    cutfreq: float = 25.0,
    window_duration: float = 1.0,
):
    """Compute step responses from input/output signals."""
    dt = np.diff(t).mean()
    if np.isclose(dt, 0):
        return None, None
    fs = 1 / dt

    frame_samples = int(window_duration * fs)
    shift = frame_samples // 16
    response_window_samples = int(0.5 * fs)
    time_response = t[:response_window_samples] - t[0]

    # Sliding windows
    u_windows = sliding_window_view(u, frame_samples)[::shift]
    y_windows = sliding_window_view(y, frame_samples)[::shift]

    # Apply Hanning window
    window_func = np.hanning(frame_samples)
    u_windows = u_windows * window_func
    y_windows = y_windows * window_func

    # Deconvolution
    deconvolved = wiener_deconvolution(u_windows, y_windows, cutfreq, fs)
    step_responses = deconvolved[:, :response_window_samples].cumsum(axis=1)

    # Remove outlier windows
    peaks = step_responses.max(axis=1)
    median_peak = np.median(peaks)
    mad = np.median(np.abs(peaks - median_peak))
    mask = np.abs(peaks - median_peak) < 3 * mad
    step_responses = step_responses[mask]

    return time_response, step_responses


def compute_step_response_metrics(time: np.ndarray, responses: np.ndarray) -> dict:
    """Compute rise time, settling time, overshoot, steady-state error."""
    rise_times, settling_times, overshoots, steady_state_errors = [], [], [], []

    for resp in responses:
        final_val = resp[-1]

        # Rise time (10% -> 90%)
        try:
            t10 = time[np.where(resp >= 0.1 * final_val)[0][0]]
            t90 = time[np.where(resp >= 0.9 * final_val)[0][0]]
            rise_times.append(t90 - t10)
        except IndexError:
            rise_times.append(np.nan)

        # Settling time (within ±10% of final value)
        tolerance = 0.1 * final_val
        within_bounds = np.where(np.abs(resp - final_val) <= tolerance)[0]
        if len(within_bounds) > 0:
            for idx in within_bounds:
                # check that response stays within bounds till the end
                if np.all(np.abs(resp[idx:] - final_val) <= tolerance):
                    settling_times.append(time[idx])
                    break
            else:
                settling_times.append(np.nan)
        else:
            settling_times.append(np.nan)

        # Overshoot
        peak_val = np.max(resp[: int(len(resp) * 0.8)])  # ignore drift at end
        overshoot = (peak_val - final_val) / final_val * 100
        overshoots.append(overshoot)

        # Steady-state error (final value vs ideal 1.0)
        steady_state_errors.append(final_val - 1)

    metrics = {
        "rise_time (s)": (np.nanmean(rise_times), np.nanstd(rise_times)),
        "settling_time (s)": (np.nanmean(settling_times), np.nanstd(settling_times)),
        "overshoot (%)": (np.nanmean(overshoots), np.nanstd(overshoots)),
        "steady_state_error": (
            np.nanmean(steady_state_errors),
            np.nanstd(steady_state_errors),
        ),
    }

    return metrics


def wiener_deconvolution(
    input_: np.ndarray,
    output: np.ndarray,
    cutoff_freq: float,
    fs: float,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Wiener deconvolution on input/output signals."""
    n_samples = input_.shape[1]
    n_fft = 2 ** int(np.ceil(np.log2(n_samples)))

    # Pad for FFT
    input_padded = np.pad(input_, ((0, 0), (0, n_fft - n_samples)), mode="constant")
    output_padded = np.pad(output, ((0, 0), (0, n_fft - n_samples)), mode="constant")

    # FFT
    H = np.fft.fft(input_padded, axis=-1)
    G = np.fft.fft(output_padded, axis=-1)

    # Wiener filter
    snr = create_frequency_mask(n_fft, cutoff_freq, fs)
    H_conj = np.conj(H)
    deconv_freq = (H_conj * G) / (H * H_conj + epsilon / snr[None, :])

    # IFFT to get impulse response
    deconvolved = np.real(np.fft.ifft(deconv_freq, axis=-1))
    return deconvolved[:, :n_samples]


def create_frequency_mask(
    n_samples: int, cutoff_freq: float, fs: float, sigma_factor: float = 6.0
) -> np.ndarray:
    """Create a smooth low-pass Gaussian frequency mask."""
    freqs = np.fft.fftfreq(n_samples, 1 / fs)
    mask = np.exp(-0.5 * (freqs / cutoff_freq) ** 2)
    mask = gaussian_filter1d(mask, sigma=n_samples / sigma_factor)
    return np.clip(mask, 1e-3, 1.0)
