import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d


def plot_closed_loop_step_response(
    u: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    cutfreq: float = 25.0,
    window_duration: float = 1.0,
) -> dict:
    """
    Estimate and plot the closed-loop step response using Wiener deconvolution,
    including uncertainty bounds and key metrics.

    Parameters
    ----------
    u : ndarray
        Input setpoint signal.
    y : ndarray
        Measured output signal.
    t : ndarray
        Time vector corresponding to u and y.
    cutfreq : float
        Cutoff frequency for Wiener deconvolution (Hz).
    window_duration : float
        Duration of each analysis window in seconds.

    Returns
    -------
    dict
        Metrics: rise_time, settling_time, overshoot, steady_state_error (mean ± std)
    """
    fs = estimate_sampling_frequency(t)
    frame_samples = int(window_duration * fs)
    shift = frame_samples // 16
    response_window_samples = int(0.5 * fs)
    time_response = t[:response_window_samples] - t[0]

    # Extract overlapping windows
    u_windows = sliding_window_view(u, frame_samples)[::shift]
    y_windows = sliding_window_view(y, frame_samples)[::shift]

    # Apply Hanning window
    window_func = np.hanning(frame_samples)
    u_windows = u_windows * window_func
    y_windows = y_windows * window_func

    # Wiener deconvolution
    deconvolved = wiener_deconvolution(u_windows, y_windows, cutfreq, fs)
    step_responses = deconvolved[:, :response_window_samples].cumsum(axis=1)

    # Plot with uncertainty and compute metrics
    metrics = plot_step_responses_with_metrics(time_response, step_responses)
    return metrics


def estimate_sampling_frequency(t: np.ndarray) -> float:
    dt = np.diff(t).mean()
    if dt == 0:
        raise ValueError("Time vector has zero differences.")
    return 1 / dt


def wiener_deconvolution(
    input_: np.ndarray, output: np.ndarray, cutfreq: float, fs: float
) -> np.ndarray:
    pad_len = 1024 - (input_.shape[1] % 1024)
    input_padded = np.pad(input_, ((0, 0), (0, pad_len)), mode="constant")
    output_padded = np.pad(output, ((0, 0), (0, pad_len)), mode="constant")

    H = np.fft.fft(input_padded, axis=-1)
    G = np.fft.fft(output_padded, axis=-1)

    sn = create_frequency_mask(H.shape[1], cutfreq, fs)
    H_conj = np.conj(H)
    denom = (H * H_conj) + (1.0 / sn)
    deconvolved = np.real(np.fft.ifft(G * H_conj / denom, axis=-1))

    return deconvolved


def create_frequency_mask(n_samples: int, cutfreq: float, fs: float) -> np.ndarray:
    freqs = np.abs(np.fft.fftfreq(n_samples, 1 / fs))
    mask = np.clip(freqs, cutfreq - 1e-9, cutfreq)
    mask = normalize(mask)
    len_lpf = np.sum(1 - mask)
    mask = normalize(gaussian_filter1d(mask, len_lpf / 6.0))
    return 10.0 * (-mask + 1.0 + 1e-9)


def normalize(arr: np.ndarray) -> np.ndarray:
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 1e-10:
        arr /= max_val
    return arr


def plot_step_responses_with_metrics(time: np.ndarray, responses: np.ndarray) -> dict:
    mean_resp = responses.mean(axis=0)
    std_resp = responses.std(axis=0)

    plt.figure(figsize=(8, 5))

    # Plot individual responses lightly
    for resp in responses:
        plt.plot(time, resp, alpha=0.2, color="gray")

    # Mean response
    plt.plot(time, mean_resp, color="blue", linewidth=2, label="Mean response")

    # Uncertainty bounds (±1 std)
    plt.fill_between(
        time,
        mean_resp - std_resp,
        mean_resp + std_resp,
        color="blue",
        alpha=0.2,
        label="±1 std",
    )

    # Reference step input
    step_time = np.concatenate([[-0.01, 0], time])
    step_values = np.concatenate([[0, 1], np.ones_like(time)])

    plt.step(step_time, step_values, where='post', color='red', label="Step Input")

    plt.xlabel("Time [s]")
    plt.ylabel("Step Response")
    plt.title("Estimated Step Response with Uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    # Compute metrics
    metrics = compute_step_response_metrics(time, responses)
    return metrics


def compute_step_response_metrics(time: np.ndarray, responses: np.ndarray) -> dict:

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
        above_lower = np.where(resp < final_val + tolerance)[0]
        below_upper = np.where(resp > final_val - tolerance)[0]
        within_bounds = np.intersect1d(above_lower, below_upper)

        # Find first index after which response stays within bounds for the rest of the signal
        for idx in within_bounds:
            if np.all(resp[idx:] <= final_val + tolerance) and np.all(resp[idx:] >= final_val - tolerance):
                settling_times.append(time[idx])
                break
        else:
            settling_times.append(np.nan)

        # Overshoot (%)
        overshoot = (np.max(resp) - final_val) / final_val * 100
        overshoots.append(overshoot)

        # Steady-state error
        steady_state_errors.append(final_val - 1)

    metrics = {
        "rise_time": (np.nanmean(rise_times), np.nanstd(rise_times)),
        "settling_time": (np.nanmean(settling_times), np.nanstd(settling_times)),
        "overshoot (%)": (np.nanmean(overshoots), np.nanstd(overshoots)),
        "steady_state_error": (
            np.nanmean(steady_state_errors),
            np.nanstd(steady_state_errors),
        ),
    }

    print("Step Response Metrics (mean ± std):")
    for k, v in metrics.items():
        print(f"{k}: {v[0]:.3f} ± {v[1]:.3f}")

    return metrics
