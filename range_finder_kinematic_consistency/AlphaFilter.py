#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Copyright (c) 2022 PX4 Development Team
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
    3. Neither the name PX4 nor the names of its contributors may be
    used to endorse or promote products derived from this software
    without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
    OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
    AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

File: AlphaFilter.py
Author: Mathieu Bresciani <mathieu@auterion.com>
License: BSD 3-Clause
Description:
"""

import sys
import numpy as np

M_PI_F = 3.14159265
FLT_EPSILON = sys.float_info.epsilon

class AlphaFilter(object):

    def __init__(self):
        self._cutoff_freq = 0.0
        self._alpha = 0.0
        self._filter_state = 0.0

	# Set filter parameters for time abstraction
	#
	# Both parameters have to be provided in the same units.
	#
	# @param sample_interval interval between two samples
	# @param time_constant filter time constant determining convergence
    def setParameters(self, sample_interval, time_constant):
        denominator = time_constant + sample_interval

        if denominator > FLT_EPSILON:
            self.setAlpha(sample_interval / denominator)

    def setCutoffFreq(self, sample_freq, cutoff_freq):
        if (sample_freq <= 0.0) or (cutoff_freq <= 0.0) or (cutoff_freq >= sample_freq / 2.0) or not np.isfinite(sample_freq) or not np.isfinite(cutoff_freq):
			# Invalid parameters
            return False

        self.setParameters(1.0 / sample_freq, 1.0 / (2.0 * M_PI_F * cutoff_freq))
        self._cutoff_freq = cutoff_freq
        return True

	# Set filter parameter alpha directly without time abstraction
	#
	# @param alpha [0,1] filter weight for the previous state. High value - long time constant.
    def setAlpha(self, alpha):
        self._alpha = alpha

	# Set filter state to an initial value
	#
	# @param sample new initial value
    def reset(self, sample):
        self._filter_state = sample

	# Add a new raw value to the filter
	#
	# @return retrieve the filtered result
    def update(self, sample):
        self._filter_state = self.updateCalculation(sample)
        return self._filter_state

    def getState(self):
        return self._filter_state

    def getCutoffFreq(self):
        return self._cutoff_freq

    def updateCalculation(self, sample):
        return (1.0 - self._alpha) * self._filter_state + self._alpha * sample

if __name__ == '__main__':
    lpf = AlphaFilter()
    lpf.setAlpha(0.1)

    for i in range(1, 100):
        out = lpf.update(1)
        print(out);

