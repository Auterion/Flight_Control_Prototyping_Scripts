#!/usr/bin/env python3

"""
File: leaky_integrator.py
Author: Matthias Grob
Email: maetugr@gmail.com
Github: https://github.com/maetugr
Description:
    Prototyping script for the simple single pole IIR filter of the form
    y[n] = alpha * y[n-1] + (alpha - 1) * x[n]
    well known as leaky integrator.
    Requires numpy, matplotlib
"""

import numpy as np
import matplotlib.pylab as plt

def runExperiment(total_time, samples, axis):
    # time setup
    dt = total_time / samples
    t = np.arange(0, total_time, dt)

    # filter configuration
    tau = 0.5
    alpha = tau / (tau + dt)

    # signal and noise setup
    np.random.seed(1)
    signal = 1 + (t / total_time)
    noise = np.random.rand(samples) - 0.5
    noisy = signal + noise

    # filter
    filtered = np.zeros(samples)
    filtered[0] = 0.0
    for n in range(1, samples):
        filtered[n] = alpha * filtered[n-1] + (1 - alpha) * noisy[n]
    
    axis.plot(t, noisy, label = "noisy")
    axis.plot(t, signal, label = "signal")
    axis.plot(t, filtered, label = "filtered")


# ploting
fig, axs = plt.subplots(4, 1)

runExperiment(10, 100, axs[0])
runExperiment(10, 1000, axs[1])
runExperiment(10, 10000, axs[2])
runExperiment(10, 100000, axs[3])

plt.show()