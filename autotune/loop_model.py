"""Closed loop made of the identified plant and a PX4 PID controller.

The controller has two degrees of freedom:

    u = C_r(z) r - C_y(z) y + sign * disturbance

C_y, which closes the loop, is always sign * Kp * (1 + I + D). Where P and D
act (on the error or on the feedback only) and the feedforward only change
the reference path C_r.
"""

import control as ctrl
import numpy as np
from system_identification import arx_transfer_function

kDerivativeCutoffFreq = 10.0  # Hz


def idealGains(gains, form="parallel"):
    """Gains P, I, D and FF to the ideal form Kp * (1 + I + D) used here.

    form is "parallel" (P + I + D) or "ideal". Returns None for parallel gains
    without P, which have no ideal form.
    """
    p = gains.get("P", 0.0)
    i = gains.get("I", 0.0)
    d = gains.get("D", 0.0)
    if form == "parallel":
        if p == 0.0:
            return None
        i /= p
        d /= p
    return {"P": p, "I": i, "D": d, "FF": gains.get("FF", 0.0)}


def parallelGains(ideal_gains):
    p = ideal_gains["P"]
    return {
        "P": p,
        "I": p * ideal_gains["I"],
        "D": p * ideal_gains["D"],
        "FF": ideal_gains["FF"],
    }


class LoopModel:
    def __init__(
        self,
        num,
        den,
        dt,
        delays,
        gains,
        negate_output=False,
        p_on_feedback=False,
        d_on_feedback=True,
    ):
        self.dt = dt
        self.plant = self._buildPlant(num, den, dt, delays)
        self.controller = self._buildController(
            dt, gains, negate_output, p_on_feedback, d_on_feedback
        )
        self.closed_loop = ctrl.interconnect(
            [self.controller, self.plant],
            inputs=["r", "disturbance"],
            outputs="y",
        )

        self.reference_controller = ctrl.minreal(
            ctrl.tf(self.controller[0, 0]), verbose=False
        )
        self.feedback_controller = ctrl.minreal(
            -ctrl.tf(self.controller[0, 1]), verbose=False
        )
        self.loop_gain = self.feedback_controller * ctrl.tf(self.plant)

    @staticmethod
    def _buildPlant(num, den, dt, delays):
        # The identified delay is part of the plant and the measurement is
        # available one sample later, both sit inside the loop
        plant = (
            arx_transfer_function(num, den, dt)
            * ctrl.tf([1], np.append([1], np.zeros(delays)), dt)
            * ctrl.tf([1], [1, 0], dt)
        )
        return ctrl.tf(plant.num, plant.den, dt, inputs="u", outputs="y")

    @staticmethod
    def _buildController(dt, gains, negate_output, p_on_feedback, d_on_feedback):
        kc = gains["P"]
        ki = gains["I"]
        kd = gains["D"]
        kff = gains["FF"]

        sum_feedback = ctrl.summing_junction(inputs=["r", "-y"], output="e")
        feedforward = ctrl.tf([kff], [1], dt, inputs="r", outputs="ff_out")

        # Integrator discretized using bilinear transform: s = 2(z-1)/(dt(z+1))
        i_control = ctrl.tf(
            [ki * dt, ki * dt], [2, -2], dt, inputs="e", outputs="i_out"
        )

        # Derivative with 1st order LPF (discretized using Euler method: s = (z-1)/dt)
        tau = 1 / (2 * np.pi * kDerivativeCutoffFreq)
        derivative_num = np.array([kd, -kd])
        derivative_den = np.array([tau, -tau + dt])
        if d_on_feedback:
            # Removes the "derivative kick" on setpoint changes
            d_control = ctrl.tf(
                -derivative_num, derivative_den, dt, inputs="y", outputs="d_out"
            )
        else:
            d_control = ctrl.tf(
                derivative_num, derivative_den, dt, inputs="e", outputs="d_out"
            )

        # P on feedback only removes the zero (3-loop autopilot style)
        p_input = "-y" if p_on_feedback else "e"
        id_control = ctrl.summing_junction(
            inputs=[p_input, "i_out", "d_out"], output="id_out"
        )
        p_control = ctrl.tf([kc], [1], dt, inputs="id_out", outputs="pid_out")
        sum_control = ctrl.summing_junction(
            inputs=["pid_out", "ff_out", "disturbance"], output="control_out"
        )
        output_sign = -1.0 if negate_output else 1.0
        out_sign = ctrl.tf(output_sign, 1.0, dt, inputs="control_out", outputs="u")

        return ctrl.interconnect(
            [
                sum_feedback,
                feedforward,
                i_control,
                d_control,
                id_control,
                p_control,
                sum_control,
                out_sign,
            ],
            inputs=["r", "y", "disturbance"],
            outputs="u",
        )

    @property
    def reference_to_output(self):
        return self.closed_loop[0, 0]

    def closedLoopPoles(self, gain_scale=1.0):
        # Roots of 1 + k * L: unlike the poles of the closed-loop realization,
        # they do not include the modes of C_r that the feedback cannot move
        return ctrl.poles(ctrl.feedback(gain_scale * self.loop_gain, 1))

    def rootLocus(self, gain_scales):
        """Closed-loop poles when the loop gain is scaled by each of gain_scales.

        As the PID is in ideal form Kp * (1 + I + D), scaling the loop gain is
        the same as scaling Kp with I and D fixed.
        """
        return ctrl.root_locus_map(self.loop_gain, gains=gain_scales).loci

    def isStable(self):
        return bool(np.all(np.abs(self.closedLoopPoles()) < 1.0))

    def nyquistFrequency(self):
        return np.pi / self.dt

    def marginFrequencies(self):
        return np.geomspace(1e-2, self.nyquistFrequency(), 2000)

    def stabilityMargins(self):
        # Always use the frequency-response method: the polynomial method is
        # often numerically inaccurate for these high-order discrete loops
        return ctrl.stability_margins(
            ctrl.frd(self.loop_gain, self.marginFrequencies(), smooth=True)
        )

    def delayMargin(self):
        """Smallest additional loop delay (s) that destabilizes the loop.

        Returns (delay_margin, crossover_frequency), the delay margin being
        the minimum of phase_margin / crossover_frequency over all the gain
        crossovers, (inf, nan) if the loop gain never crosses 1.
        """
        omega = self.marginFrequencies()
        log_mag = np.log(np.abs(self.loop_gain(np.exp(1j * omega * self.dt))))
        crossings = np.nonzero(np.diff(np.sign(log_mag)))[0]

        delay_margin = np.inf
        crossover_frequency = np.nan
        for i in crossings:
            ratio = -log_mag[i] / (log_mag[i + 1] - log_mag[i])
            w_c = omega[i] * (omega[i + 1] / omega[i]) ** ratio
            loop_c = self.loop_gain(np.exp(1j * w_c * self.dt))
            phase_margin = np.mod(np.angle(loop_c) + np.pi, 2 * np.pi)
            if phase_margin / w_c < delay_margin:
                delay_margin = phase_margin / w_c
                crossover_frequency = w_c

        return delay_margin, crossover_frequency

    def diskMargins(self):
        """Balanced disk margin, robust to simultaneous gain and phase changes.

        Returns (alpha, gain_margin_db, phase_margin_deg, frequency): the
        loop stays stable for any simultaneous gain change within
        +/-gain_margin_db and phase change within +/-phase_margin_deg.
        """
        omega = self.marginFrequencies()
        alpha, gain_margin_db, phase_margin_deg = ctrl.disk_margins(
            self.loop_gain, omega, returnall=True
        )
        i_min = np.argmin(alpha)
        return (
            alpha[i_min],
            gain_margin_db[i_min],
            phase_margin_deg[i_min],
            omega[i_min],
        )

    def simulate(self, t, r, disturbance):
        _, y = ctrl.forced_response(
            self.closed_loop, t, np.vstack((r, disturbance)), squeeze=True
        )
        return y

    def sensitivities(self, omega):
        """Complex frequency responses of the gang of six, at omega (rad/s)."""
        z = np.exp(1j * np.asarray(omega) * self.dt)
        plant = self.plant(z)
        feedback_controller = self.feedback_controller(z)
        reference_controller = self.reference_controller(z)
        loop_gain = feedback_controller * plant
        sensitivity = 1 / (1 + loop_gain)
        return {
            "S": sensitivity,
            "T": loop_gain * sensitivity,
            "PS": plant * sensitivity,
            "CS": feedback_controller * sensitivity,
            "CSF": reference_controller * sensitivity,
            "TF": reference_controller * plant * sensitivity,
        }
