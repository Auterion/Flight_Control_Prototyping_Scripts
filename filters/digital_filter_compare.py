#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: digital_filter_compare.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
    Plot the frequency and step responses of several digital filters
"""

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

frequencies = []
amplitudes = []
times = []
step_responses = []
group_delays = []
names = []

def addFilter(b, a, sampling_freq, name=""):
    w, h = signal.freqz(b, a, fs=sampling_freq)
    t, y = signal.dstep((b, a, 1.0/fs))
    w_gd, gd = signal.group_delay((b, a))

    frequencies.append(w)
    amplitudes.append(h)
    times.append(t)
    step_responses.append(y)
    group_delays.append(gd)
    names.append(name)

def plotFilters():
    plotBode()
    plotStep()

def plotBode():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    n_filters = len(frequencies)
    for n in range(n_filters):
        ax1.semilogx(frequencies[n], 20 * np.log10(abs(amplitudes[n])))
        angles = np.rad2deg(np.unwrap(np.angle(amplitudes[n])))
        ax2.semilogx(frequencies[n], angles)
        group_delay_ms = group_delays[n] / fs * 1e3
        ax3.semilogx(frequencies[n], group_delay_ms)

    ax1.set_title('Digital filter frequency response')
    ax1.set_ylabel('Amplitude (dB)')
    ax2.set_ylabel('Angle (degrees)')
    ax3.set_ylabel('Group delay (ms)')
    ax3.set_xlabel('Frequency (Hz)')
    ax1.legend(names)
    ax1.grid()
    ax2.grid()
    ax2.axis('tight')
    plt.show()

def plotStep():
    n_filters = len(times)
    for n in range(n_filters):
        plt.plot(times[n], np.squeeze(step_responses[n]))

    plt.title('Digital filter frequency response')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (-)')
    plt.legend(names)
    plt.grid()
    plt.show()

def create1stOrderButterworthLpf(fc, fs):
    gamma = np.tan(np.pi * fc / fs)
    D = gamma + 1.0
    b = [gamma / D, gamma / D]
    a = [1.0, (gamma - 1.0) / D]
    name = createName("1st Butter lpf", fc)
    return b, a, name

def create2ndOrderButterworthLpf(fc, fs):
    gamma = np.tan(np.pi * fc / fs)
    gamma2 = gamma**2
    D = gamma2 + np.sqrt(2.0) * gamma + 1.0
    b0_prime = gamma2
    b1_prime = 2.0 * b0_prime
    b2_prime = b0_prime
    a0_prime = D
    a1_prime = 2.0 * (gamma2 - 1.0)
    a2_prime = gamma2 - np.sqrt(2.0) * gamma + 1.0

    b = [b0_prime, b1_prime, b2_prime] / D
    a = [a0_prime, a1_prime, a2_prime] / D
    name = createName("2nd Butter lpf", fc)
    return b, a, name

def createLpf2p(fc, fs):
    fr = fs / fc
    ohm = np.tan(np.pi / fr)
    c = 1.0 + 2.0 * np.cos(np.pi / 4.0) * ohm + ohm**2

    b0 = ohm**2 / c
    b1 = 2.0 * b0
    b2 = b0
    a0 = 1.0
    a1 = 2.0 * (ohm**2 - 1.0) / c
    a2 = (1.0 - 2.0 * np.cos(np.pi / 4.0) * ohm + ohm**2) / c

    b = [b0, b1, b2]
    a = [a0, a1, a2]
    name = createName("PX4 2p lpf", fc)
    return b, a, name

def create2ndOrderNotch(fc, bw, fs):
    alpha = np.tan(np.pi * bw / fs)
    beta = -np.cos(2.0 * np.pi * fc / fs)
    D = alpha + 1.0
    b0_prime = 1.0
    b1_prime = 2.0 * beta
    b2_prime = 1.0
    a0_prime = D
    a1_prime = b1_prime
    a2_prime = 1.0 - alpha

    b = [b0_prime / D, b1_prime / D, b2_prime / D]
    a = [a0_prime / D, a1_prime / D, a2_prime / D]
    name = createName("2nd Notch", fc, bw)
    return b, a, name

def create2ndOrderButterworthBandStop(fc, bw, fs):
    gamma = np.tan(np.pi * fc / fs)
    gamma2 = gamma**2
    D = (1.0 + gamma2) * fc + gamma * bw
    b0_prime = fc * (gamma2 + 1.0)
    b1_prime = 2.0 * fc * (gamma2 - 1.0)
    b2_prime = b0_prime
    a0_prime = D
    a1_prime = b1_prime
    a2_prime = (1.0 + gamma2) * fc - gamma * bw

    b = [b0_prime, b1_prime, b2_prime] / D
    a = [a0_prime, a1_prime, a2_prime] / D
    name = createName("2nd Butter BStop", fc, bw)
    return b, a, name

def create2ndOrderLpf(fc, zeta, fs):
    T = 1.0 / fs
    wn = 2.0 * np.pi * fc
    K = wn / np.tan(wn * T / 2.0)
    K2 = K**2
    b2a = wn**2
    a0a = 1.0
    a1a = 2.0 * zeta * wn
    a2a = wn**2
    D = a0a*K2 + a1a*K + a2a
    b0_prime = b2a
    b1_prime = 2.0 * b2a
    b2_prime = b2a
    a0_prime = D
    a1_prime = 2.0 * a2a - 2.0*K2
    a2_prime = a0a*K2 - a1a*K + a2a

    b = [b0_prime / D, b1_prime / D, b2_prime / D]
    a = [a0_prime / D, a1_prime / D, a2_prime / D]
    name = createName("2nd", fc)
    return b, a, name

def create2ndOrderCriticallyDamped(fc, fs):
    wn = 2.0 * np.pi * fc
    K = 2.0 * fs
    K2 = K**2
    b2a = wn**2
    a1a = 2.0 * wn
    a2a = wn**2
    D = K2 + a1a*K + a2a
    b0_prime = b2a
    b1_prime = 2.0 * b2a
    b2_prime = b2a
    a0_prime = D
    a1_prime = 2.0 * a2a - 2.0*K2
    a2_prime = K2 - a1a*K + a2a

    b = [b0_prime / D, b1_prime / D, b2_prime / D]
    a = [a0_prime / D, a1_prime / D, a2_prime / D]
    name = createName("2nd crit damp", fc)
    return b, a, name

def create1stOrderLpf(fc, fs):
    dt = 1.0 / fs
    tau = 1/fc
    alpha = dt / (tau + dt)
    b0 = alpha
    a0 = 1.0
    a1 = alpha - 1.0
    b = [b0]
    a = [a0, a1]
    name = createName("1st order", fc)
    return b, a, name

def createName(prefix, fc, bw=0.0):
    name = prefix + " (fc = {}".format(fc)
    if bw > 0.1:
        name += " bw = {}".format(bw)
    name += ")".format(bw)
    return name

if __name__ == '__main__':
    fs = 1000.0
    cutoff = 80.0
    bandwidth = 30.0

    b, a, name = create1stOrderButterworthLpf(cutoff, fs)
    addFilter(b, a, fs, name)
    b, a, name = create2ndOrderButterworthLpf(cutoff, fs)
    addFilter(b, a, fs, name)
    b, a, name = create2ndOrderNotch(cutoff, bandwidth, fs)
    addFilter(b, a, fs, name)
    b, a, name = create2ndOrderButterworthBandStop(cutoff, bandwidth, fs)
    addFilter(b, a, fs, name)
    b, a, name = createLpf2p(cutoff, fs)
    addFilter(b, a, fs, name)
    b, a, name = create2ndOrderLpf(fc=cutoff, zeta=1.0, fs=fs)
    addFilter(b, a, fs, name)
    b, a, name = create2ndOrderCriticallyDamped(cutoff, fs=fs)
    addFilter(b, a, fs, name)
    b, a, name = create1stOrderLpf(cutoff, fs=fs)
    addFilter(b, a, fs, name)

    plotFilters()
