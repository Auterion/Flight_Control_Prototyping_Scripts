# Leaky integrator
Prototyping script for the simple single pole IIR filter of the form y[n] = alpha * y[n-1] + (alpha - 1) * x[n] well known as leaky integrator.

## Setup
To run the script you need Python 3 (tested on 3.8.2) and the following python modules:
```
sudo python3 -m pip install numpy matplotlib
```

Command to run the script:
```
python3 leaky_integrator.py
```

## Details

Useful practical reads:
- https://en.wikipedia.org/wiki/Leaky_integrator
- https://dsp.stackexchange.com/questions/43851/is-there-a-common-name-for-the-first-order-iir-averaging-filter
- https://dsp.stackexchange.com/questions/3179/is-a-leaky-integrator-the-same-thing-as-a-low-pass-filter

The visualization shows plots of a signal (orange) the noisy measured signal (blue) and the leaky integrator filtered signal (green) with 1e2, 1e3, 1e4, 1e5 samples over 10s. The goal is to calculate the filter parameter alpha such that the time constant stays the same independent of the filter frequency. alpha = tau / (tau + dt) with tau being the time constant and dt the time between two samples.
