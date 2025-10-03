import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QDialog, QVBoxLayout


class PIDAnalyseWindow(QDialog):
    """Dialog window to show estimated step response"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Estimated Step Response")

        # Figure and canvas
        self.figure, self.ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def generate_step_response(self, u: np.ndarray, y: np.ndarray, t: np.ndarray) -> dict:
        """Compute and plot step response, updating the canvas."""
        self.ax.clear()

        time, responses = compute_step_responses(u, y, t)
        if responses is None:
            return {}

        metrics = compute_step_response_metrics(time, responses)
        plot_step_responses(time, responses, metrics, ax=self.ax)

        self.canvas.draw()
        return metrics


def compute_step_responses(u: np.ndarray, y: np.ndarray, t: np.ndarray,
                           cutfreq: float = 25.0, window_duration: float = 1.0):
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
                if np.all(np.abs(resp[idx:] - final_val) <= tolerance):
                    settling_times.append(time[idx])
                    break
            else:
                settling_times.append(np.nan)
        else:
            settling_times.append(np.nan)

        # Overshoot (relative to final value)
        peak_val = np.max(resp[: int(len(resp) * 0.8)])  # ignore drift at end
        overshoot = (peak_val - final_val) / final_val * 100
        overshoot = max(0.0, overshoot)  # no negative overshoot here
        overshoots.append(overshoot)

        # Steady-state error (final value vs ideal 1.0)
        steady_state_errors.append(final_val - 1)

    metrics = {
        "rise_time": (np.nanmean(rise_times), np.nanstd(rise_times)),
        "settling_time": (np.nanmean(settling_times), np.nanstd(settling_times)),
        "overshoot (%)": (np.nanmean(overshoots), np.nanstd(overshoots)),
        "steady_state_error": (np.nanmean(steady_state_errors), np.nanstd(steady_state_errors)),
    }

    return metrics


def plot_step_responses(time: np.ndarray, responses: np.ndarray, metrics: dict = None, ax=None):
    """Plot step responses and optionally show metrics."""
    mean_resp = responses.mean(axis=0)
    std_resp = responses.std(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Plot all responses lightly
    ax.plot(time, responses.T, alpha=0.2, color="gray")
    ax.plot(time, mean_resp, color="blue", linewidth=2, label="Mean response")
    ax.fill_between(time, mean_resp - std_resp, mean_resp + std_resp,
                    color="blue", alpha=0.2, label="±1 std")

    # Reference step input
    ax.plot([time[0], 0, time[-1]], [0, 1, 1], "k--", label="Step Input")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Step Response")
    ax.set_title("Estimated Step Response")
    ax.legend()
    ax.grid(True)

    if metrics:
        metrics_text = "\n".join(
            f"{k}: {v[0]:.2f} ± {v[1]:.2f}" for k, v in metrics.items() if not np.isnan(v[0])
        )
        ax.text(0.98, 0.02, metrics_text, ha="right", va="bottom",
                transform=ax.transAxes, fontsize=9, color="gray")

    return ax


def wiener_deconvolution(input_: np.ndarray, output: np.ndarray,
                         cutoff_freq: float, fs: float, epsilon: float = 1e-3) -> np.ndarray:
    """Perform Wiener deconvolution on input/output signals."""
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


def create_frequency_mask(n_samples: int, cutoff_freq: float, fs: float,
                          sigma_factor: float = 6.0) -> np.ndarray:
    """Create a smooth low-pass Gaussian frequency mask."""
    freqs = np.fft.fftfreq(n_samples, 1 / fs)
    mask = np.exp(-0.5 * (freqs / cutoff_freq) ** 2)
    mask = gaussian_filter1d(mask, sigma=n_samples / sigma_factor)
    return np.clip(mask, 1e-3, 1.0)
