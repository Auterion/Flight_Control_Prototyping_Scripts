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
        K_i = 2.0*K_p/T_u
        K_d = K_p*T_u/8.0

    elif rule == "overshoot":
        K_p = 0.33*K_u
        K_i = 2.0*K_p/T_u
        K_d = K_p*T_u/3.0

    elif rule == "no_overshoot":
        K_p = 0.2*K_u
        K_i = 2.0*K_p/T_u
        K_d = K_p*T_u/3.0

    elif rule == "pessen":
        K_p = 0.7*K_u
        K_i = 2.5*K_p/T_u
        K_d = 0.15*K_p*T_u

    return (K_p, K_i, K_d)


K_u = 0.8 # Ultimate closed-loop gain
T_u = 0.5 # Oscillation period in seconds at critical gain

rules = ["classical", "overshoot", "no_overshoot", "pessen"]

for rule in rules:

    (K_p, K_i, K_d) = compute_PID(K_u, T_u, rule=rule)
    print("Rule \"{}\"\nKp = {}\nKi = {}\nKd = {}\n".format(rule, K_p, K_i, K_d))
