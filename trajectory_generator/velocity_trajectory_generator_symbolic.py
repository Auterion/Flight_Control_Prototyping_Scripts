#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: velocity_trajectory_generator_symbolic.py
Author: Mathieu Bresciani
Email: brescianimathieu@gmail.com
Github: https://github.com/bresch
Description:
    Derivation of a limited jerk time optimal velocity-driven
    trajectory generator.
"""

from sympy import *

j = Symbol("j_max", real=True)
a_0 = Symbol("a0", real=True)
a_3 = Symbol("a3", real=True)
v_0 = Symbol("v0", real=True)
v_3 = Symbol("v3", real=True)
T_1 = Symbol("T1", real=True)
T_2 = Symbol("T2", real=True)
T_3 = Symbol("T3", real=True)
a_max = Symbol("a_max", real=True)

f1 = a_0 + j*T_1 - j*T_3 - a_3
f1 = f1.subs(a_3, 0)
res_T3 = solve(f1, T_3)
res_T3 = res_T3[0]
f2 = a_0*T_1 + j/2.0*T_1**2 + v_0 + a_0*T_2 + j*T_1*T_2 + a_0*T_3 + j*T_1*T_3 - j/2.0*T_3**2 - v_3
f2 = f2.subs([(v_0, 0)])
print("Step 1 - Setting T_2 = 0, compute T_1 as follows:")
res_T1 = solve(f2.subs([(T_2, 0), (T_3, res_T3)]), T_1)
res_T1 = res_T1[0]
print("T_1 =")
pprint(res_T1)
print(res_T1)
print("If jerk is too large, recompute it for fixed T1. New jerk = {}")
res_j1 = solve(f2.subs([(T_2, 0), (T_3, res_T3)]), j)
pprint(res_j1[0])
print(res_j1[0])
res1_T1 = solve(f1, T_1)
res1_T1 = res1_T1[0]
res_j3 = solve(f2.subs([(T_2, 0), (T_1, res1_T1)]), j)
print("If jerk is too large, and T1 = 0. New jerk = {}")
pprint(res_j3[0])
print(res_j3[0])
print("Step 2 - Check for saturation. If a_0 + j*T_1 > a_max, recompute T_1 using:")
print("T_1 =")
pprint(solve(a_0 + j*T_1 - a_max, T_1)[0])
print(solve(a_0 + j*T_1 - a_max, T_1)[0])
print("Step 3 - Compute T3 using:")
print("T_3 =")
pprint(res_T3)
print(res_T3)
res_T2 = solve(f2, T_2)
res_T2 = res_T2[0]
print("Step 3 - Finally compute the required constant acceleration part using:")
print("T_2 =")
pprint(res_T2)
print(res_T2)

dt = Symbol("dt", real=True)
j_max = Symbol("j_max", real=True)
x_0 = Symbol("x_0", real=True)
j_sp = Symbol("j_sp", real=True)
a_sp = Symbol("a_sp", real=True)
v_sp = Symbol("v_sp", real=True)

print("=============== Generate the trajectory ===============")
f_j_sp = j_max
f_a_sp = integrate(f_j_sp, dt) + a_0
f_v_sp = integrate(f_a_sp, dt) + v_0
f_x_sp = integrate(f_v_sp, dt) + x_0
print("j_sp =")
pprint(f_j_sp)
print("a_sp =")
f_a_sp_subs = f_a_sp.subs(f_j_sp, j_sp)
pprint(f_a_sp_subs)
print("v_sp =")
f_v_sp_subs = f_v_sp.subs(f_a_sp, a_sp)
pprint(f_v_sp_subs)
print("x_sp =")
pprint(f_x_sp)

print("=============== Time synchronization ===============")
T123 = Symbol("T123", real=True) # Total time

f3 = T123 - T_1 - T_2 - T_3 # Time constraint
res_T2 = solve(f3.subs(T_3, res_T3), T_2)
res_T2 = res_T2[0]
res_T1 = solve(f2.subs([(T_2, res_T2), (T_3, res_T3)]), T_1)
res_T1 = res_T1
print("For multiple axes applications, we need to synchronize the trajectories such that they all end at the same time.\n\
Given the total time of the longest trajectory T, we can solve again T_1, T_2 and T_3 for the other axes:")
print("T_1 =")
pprint(res_T1)
print(res_T1)
print("T_3 =")
pprint(res_T3)
print(res_T2)
print("T_2 =")
pprint(solve(f3, T_2)[0])
