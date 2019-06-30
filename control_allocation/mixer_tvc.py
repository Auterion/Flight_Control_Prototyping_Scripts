#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: mixer_tvc.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
    Control allocation of a Thrust Vector Control (TVC) actuator module.
    TVC is obtained using a single thruster and several adjustable aerodynamic
    deflectors (fins) placed in its flow.
    Since the actuator effectiveness matrix is dependant of the current thrust,
    low thrust can result in a complete loss of control. To protect against this
    problem, Optimal Automatic Throttle Adjustment (OATA) can be anabled via the "airmode" parameter.
    When airmode is activated, the control allocator is allowed to increase
    the thruster output in order to increase the effectiveness of the fins
    when they would have to exceed their saturation to achieve the
    required force for the given thrust.

    The first fin is located at 'alpha' radians and the numbering is done counter clockwise
    N is the number of fins and are equally spaced

    Examples:
    N = 4, Alpha = 0          N = 3, Alpha = pi/2
            2                       1
            |                       |
            |                       |
    3 ----     ---- 1
            |                   /       \
            |                  /         \
            4                 2           3
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
        if abs(delta_u[i]) < 0.000001:
            # avoid division by zero
            continue

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
    # increasing the total thrust of the propulsion system blowing on the fins
    k_1 = compute_desaturation_gain(u, u_min, u_max, delta_u)
    u_1 = u + k_1 * delta_u # Try to unsaturate
    k_2 = compute_desaturation_gain(u_1, u_min, u_max, delta_u)

    # Compute optimal gain that equilibrates the saturations
    k_opt = k_1 + 0.5 * k_2

    u_prime = u + k_opt * delta_u
    return u_prime

def thrust_boost(u, u_max, u_T):
    # Boost gain to unsaturate the actuators
    # As the thrust scales linearly the control effectiveness
    # matrix, the thrust is simply scaled by the gain required to unsaturate
    # the most saturated actuator
    # This gain should at be at least 1
    gain = max(max(abs(u) / u_max).item(0), 1.0)
    u_T_boost = min(gain * u_T, 1.0)

    # Adjust the outputs for the new thrust
    u_prime = u / gain
    return (u_prime, u_T_boost)

def mix_yaw(m_sp, u, P, u_T, u_min, u_max):
    m_sp_yaw_only = np.matlib.zeros(m_sp.size).T
    m_sp_yaw_only[2, 0] = m_sp[2, 0]
    u_p = u + P / u_T * m_sp_yaw_only

    # Change yaw acceleration to unsaturate the outputs if needed (do not change roll/pitch),
    u_r_dot = P[:,2]
    u_pp = minimize_sat(u_p, u_min, u_max, u_r_dot)
    return u_pp

def airmode_xy(m_sp, P, u_T, u_min, u_max):
    # Mix without yaw
    m_sp_no_yaw = m_sp.copy()
    m_sp_no_yaw[2, 0] = 0.0
    u = P / u_T * m_sp_no_yaw

    # Use thrust to unsaturate the fins if needed
    (u_p, u_T) = thrust_boost(u, u_max, u_T)

    # Mix yaw axis independently
    u_final = mix_yaw(m_sp, u_p, P, u_T, u_min, u_max)

    return (u, u_final, u_T)

def airmode_xyz(m_sp, P, u_T, u_min, u_max):
    # Mix with yaw
    u = P / u_T * m_sp

    # Use thrust to unsaturate the fins if needed
    (u_p, u_T) = thrust_boost(u, u_max, u_T)

    # Unsaturate with yaw
    u_r_dot = P[:,2]
    u_final = minimize_sat(u_p, u_min, u_max, u_r_dot)
    return (u, u_final, u_T)

def normal_mode(m_sp, P, u_T, u_min, u_max):
    # Mix without yaw
    m_sp_no_yaw = m_sp.copy()
    m_sp_no_yaw[2, 0] = 0.0
    u = P / u_T * m_sp_no_yaw

    # Mix yaw axis independently
    u_final = mix_yaw(m_sp, u, P, u_T, u_min, u_max)
    return (u, u_final, u_T)

# --------------------------------------------------
# --------------------------------------------------

# m = B * u_T * u
# m - the vector of desired angular accelerations
# B - the actuator effectiveness matrix
# u_T - the desired thrust
# u - the vector of normalized fins deflections

# Generate the actuator effectiveness matrix
# Number of fins, usually 3, 4 or 6
N = 4
# TVC '+' configuration
alpha = 0.0
# TVC 'x' configuration, same as the '+' configuration but rotated by pi/N
# this configuration produces more roll/pitch control effectiveness
alpha = np.pi / N
phi= np.zeros(N)
delta_phi = 2.0 * np.pi / N # angle between two fins

for i in range(1, N):
    phi[i] = i * delta_phi + alpha

B = np.matlib.zeros([3, N])

# Fill B matrix using geometry
for i in range(0, 3):
    for j in range(0, N):
        if i == 0:
            # Roll in hover, rudder in FW
            B[i, j] = -np.sin(phi[j])/N
        elif i == 1:
            # Pitch in hover, elevator
            B[i, j] = np.cos(phi[j])/N
        elif i == 2:
            # Yaw in hover, ailerons in FW
            B[i, j] = 1.0/N

# Compute the control allocation matrix using the pseudo inverse of the actuator effectiveness matrix
# u = P / u_T * m
P = np.linalg.pinv(B)

# Desired accelerations (actuator controls)
p_dot_sp = 0.3 # roll acceleration (p is the roll rate)
q_dot_sp = -0.1 # pitch acceleration
r_dot_sp = 0.13 # yaw acceleration
u_T = 0.5 # vertical thrust
m_sp = np.matrix([p_dot_sp, q_dot_sp, r_dot_sp]).T # Vector of desired "accelerations"

# Airmode type (none/xy/xyz)
airmode = "xy"

# Actuators output saturations
u_max = 1.0
u_min = -1.0

if airmode == "none":
    (u, u_new, u_T_new) = normal_mode(m_sp, P, u_T, u_min, u_max)
elif airmode == "xy":
    (u, u_new, u_T_new) = airmode_xy(m_sp, P, u_T, u_min, u_max)
elif airmode == "xyz":
    (u, u_new, u_T_new) = airmode_xyz(m_sp, P, u_T, u_min, u_max)
else:
    u = 0.0
    u_new = 0.0

# Saturate the outputs between -1 and 1
u_new_sat = np.maximum(u_new, -np.matlib.ones(u.size).T)
u_new_sat = np.minimum(u_new_sat, np.matlib.ones(u.size).T)

# Display some results
print("u = {}\n".format(u))
print("u_new = {}\n".format(u_new))
print("u_new_sat = {}\n".format(u_new_sat))
print("u_T_new = {}\n".format(u_T_new))
print("Desired accelerations = {}\n".format(m_sp))

# Compute back the allocated accelerations
m_new = B * u_T_new * u_new_sat
print("Allocated accelerations = {}\n".format(m_new))
