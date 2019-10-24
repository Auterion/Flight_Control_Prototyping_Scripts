#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: mixer_multirotor.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
"""

import numpy as np
import numpy.matlib

from mixerlib import *

# --------------------------------------------------
# --------------------------------------------------

# Normalized actuator effectiveness matrix
# m = B * u

# Normal quad x
#B = np.matrix([
#                [-1.0, 1.0, 1.0, -1.0],
#                [1.0, -1.0, 1.0, -1.0],
#                [1.0, 1.0, -1.0, -1.0],
#                [1.0, 1.0, 1.0, 1.0]])

# Wide front arms
B = np.matrix([
                [-1.0, 0.5, 1.0, -0.5],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0]])

# Compute the control allocation matrix using the pseudo inverse of the actuator effectiveness matrix
# u = P * m
P = np.linalg.pinv(B)

##################################
# quad x
P = np.matrix([
	[ -0.707107,  0.707107,  1.000000,  1.000000 ],
	[  0.707107, -0.707107,  1.000000,  1.000000 ],
	[  0.707107,  0.707107, -1.000000,  1.000000 ],
	[ -0.707107, -0.707107, -1.000000,  1.000000 ]])
B = np.linalg.pinv(P)
##################################

# Desired accelerations (actuator controls)
p_dot_sp = 2.2 # roll acceleration (p is the roll rate)
q_dot_sp = -0.1 # pitch acceleration
r_dot_sp = 0.03 # yaw acceleration
T_sp = 1.0 # vertical thrust
m_sp = np.matrix([p_dot_sp, q_dot_sp, r_dot_sp, T_sp]).T # Vector of desired "accelerations"

# Airmode type (none/xy/xyz)
airmode = "none"

# Actuators output saturations
u_max = 1.0
u_min = 0.0

if airmode == "none":
    (u, u_new) = normal_mode(m_sp, P, u_min, u_max)
elif airmode == "xy":
    (u, u_new) = airmode_xy(m_sp, P, u_min, u_max)
elif airmode == "xyz":
    (u, u_new) = airmode_xyz(m_sp, P, u_min, u_max)
else:
    u = 0.0
    u_new = 0.0

# Saturate the outputs between 0 and 1
u_new_sat = np.maximum(u_new, np.matlib.zeros(u.size).T)
u_new_sat = np.minimum(u_new_sat, np.matlib.ones(u.size).T)

# Display some results
print("u = {}\n".format(u))
print("u_new = {}\n".format(u_new))
print("u_new_sat = {}\n".format(u_new_sat))
print("Desired accelerations = {}\n".format(m_sp))
# Compute back the allocated accelerations
m_new = B * u_new_sat
print("Allocated accelerations = {}\n".format(m_new))
