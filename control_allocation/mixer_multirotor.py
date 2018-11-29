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


def compute_desaturation_gain(u, u_min, u_max, delta_u):
    # Computes the gain k by which delta_u has to be multiplied
    # in order to unsaturate the output that has the greatest saturation
    d_u_sat_plus = u_max - u
    d_u_sat_minus = u_min - u
    k = np.zeros(u.size*2)
    for i in range(u.size):
        if d_u_sat_minus[i] > 0.0:
            k[2*i] = d_u_sat_minus[i] / delta_u[i]
        if d_u_sat_plus[i] < 0.0:
            k[2*i+1] = d_u_sat_plus[i] / delta_u[i]

    k_min = min(k)
    k_max = max(k)

    # Reduce the saturation as much as possible
    k = k_min + k_max
    return k


def minimize_sat(u, u_min, u_max, delta_u):
    # Minimize the saturation of the actuators by
    # adding or substracting a fraction of delta_u.
    # Delta_u is the vector that added to the output u,
    # modifies the thrust or angular acceleration on a
    # specific axis.
    # For example, if delta_u is given
    # to slide along the vertical thrust axis, the saturation will
    # be minimized by shifting the vertical thrust setpoint,
    # without changing the roll/pitch/yaw accelerations.
    k_1 = compute_desaturation_gain(u, u_min, u_max, delta_u)
    u_1 = u + k_1 * delta_u # Try to unsaturate
    k_2 = compute_desaturation_gain(u_1, u_min, u_max, delta_u)

    # Compute optimal gain that equilibrates the saturations
    k_opt = k_1 + 0.5 * k_2

    u_prime = u + k_opt * delta_u
    return u_prime

def mix_yaw(m_sp, u, P, u_min, u_max):
    m_sp_yaw_only = np.matlib.zeros(m_sp.size).T
    m_sp_yaw_only[2, 0] = m_sp[2, 0]
    u_p = u + P * m_sp_yaw_only

    # Change yaw acceleration to unsaturate the outputs if needed (do not change roll/pitch),
    # and allow some yaw response at maximum thrust
    u_r_dot = P[:,2]
    u_pp = minimize_sat(u_p, u_min, u_max+0.15, u_r_dot)
    u_T = P[:, 3]
    u_ppp = minimize_sat(u_pp, -1000, u_max, u_T)
    return u_ppp

def airmode_xy(m_sp, P, u_min, u_max):
    # Mix without yaw
    m_sp_no_yaw = m_sp.copy()
    m_sp_no_yaw[2, 0] = 0.0
    u = P * m_sp_no_yaw

    # Use thrust to unsaturate the outputs if needed
    u_T = P[:, 3]
    u_prime = minimize_sat(u, u_min, u_max, u_T)

    # Mix yaw axis independently
    u_final = mix_yaw(m_sp, u_prime, P, u_min, u_max)

    return (u, u_final)


def airmode_xyz(m_sp, P, u_min, u_max):
    # Mix with yaw
    u = P * m_sp

    # Use thrust to unsaturate the outputs if needed
    u_T = P[:, 3]
    u_prime = minimize_sat(u, u_min, u_max, u_T)
    return (u, u_prime)


def normal_mode(m_sp, P, u_min, u_max):
    # Mix without yaw
    m_sp_no_yaw = m_sp.copy()
    m_sp_no_yaw[2, 0] = 0.0
    u = P * m_sp_no_yaw

    # Use thrust to unsaturate the outputs if needed
    # by reducing the thrust only
    u_T = P[:, 3]
    u_prime = minimize_sat(u, u_min, u_max, u_T)
    if (u_prime > (u)).any():
        u_prime = u

    # Reduce roll/pitch acceleration if needed to unsaturate
    u_p_dot = P[:, 0]
    u_p2 = minimize_sat(u_prime, u_min, u_max, u_p_dot)
    u_q_dot = P[:, 1]
    u_p3 = minimize_sat(u_p2, u_min, u_max, u_q_dot)

    # Mix yaw axis independently
    u_final = mix_yaw(m_sp, u_p3, P, u_min, u_max)
    return (u, u_final)

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
