#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: closed_loop_ziegler_nichols.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
    Implementation of the Ziegler-Nichols rules for close-loop PID tuning. The gains are given
    for a parallel implementation of the PID controller.
    Four methods are available: "classical", "overshoot", "no_overshoot" and "pessen".
    For more details, see http://notesnet.blogspot.com/2013/02/zieglernichols-method.html
"""

def compute_PID(K_u, T_u, rule="classical"):
    if rule == "classical":
        K_p = 0.6*K_u
        T_i = 0.5*T_u
        T_d = T_u/8.0

    elif rule == "overshoot":
        K_p = 0.33*K_u
        T_i = 0.5*T_u
        T_d = T_u/3.0

    elif rule == "no_overshoot":
        K_p = 0.2*K_u
        T_i = 0.5*T_u
        T_d = T_u/3.0

    elif rule == "pessen":
        K_p = 0.7*K_u
        T_i = 0.4*T_u
        T_d = 0.15*T_u

    return (K_p, T_i, T_d)

def non_interacting_to_parallel(Kp_n, Ti_n, Td_n):
    Kp_p = Kp_n
    Ki_p = Kp_n / Ti_n
    Kd_p = Kp_n * Td_n

    return (Kp_p, Ki_p, Kd_p)

def compute_ARW_gain(K_p, K_i, K_d):
    K_ARW = 2.0 / K_p

    return K_ARW

K_u = 0.8 # Ultimate closed-loop gain
T_u = 0.5 # Oscillation period in seconds at critical gain

rules = ["classical", "overshoot", "no_overshoot", "pessen"]

for rule in rules:
    (Kp_i, Ti_i, Td_i) = compute_PID(K_u, T_u, rule=rule)
    (Kp_p, Ki_p, Kd_p) = non_interacting_to_parallel(Kp_i, Ti_i, Td_i)
    K_ARW = compute_ARW_gain(Kp_p, Ki_p, Kd_p)
    print("Rule \"{}\"\nParallel:\nKp = {}\tKi = {}\tKd = {}\tK_ARW = {}".format(rule, Kp_p, Ki_p, Kd_p, K_ARW))
    print("Non-Interacting:\nKp = {}\tTi = {}\tTd = {}\n".format(Kp_i, Ti_i, Td_i))
