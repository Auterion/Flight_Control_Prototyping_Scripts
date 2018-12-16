"""
    Copyright (c) 2018 PX4 Development Team

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

File:           quaternion_attitude_control_test.py
Author:         Matthias Grob <maetugr@gmail.com> https://github.com/MaEtUgR
License:        BSD 3-Clause
Description:
    This file runs a minimal, ideal multicopter test simulation
    to verify attitude control strategies and visualize the result.
"""

import numpy as np
from numpy import linalg as la

from pyquaternion import Quaternion
from math import sin, cos, asin, acos, degrees, radians, sqrt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plotrotc(q = Quaternion(), p = (0,0,0)):
    """
    Plot 3D RGB (x,y,z) axes of the vehicle body frame body frame

    Params:
        q: [optional] body attitude quaternion. Defaults to level.
        p: [optional] 3D body position in space Defaults to origin.
    """
    # convert the quaternion to a rotation metrix because the columns are the base vectors
    R = q.rotation_matrix
    # plot unit vectors from the body position into the 3 base vector directions
    ay.quiver(*p, *R[:,0], color='red')
    ay.quiver(*p, *R[:,1], color='green')
    ay.quiver(*p, *R[:,2], color='blue')

def qcontrol_full(q = Quaternion(), qd = Quaternion()):
    """
    Calculate angular velocity to get from current to desired attitude
    All axes are treated equally hence the name "full".

    Params:
        q: [optional] current body attitude quaternion. Defaults to level.
        qd: [optional] desired body attitude quaternion setpoint. Defaults to level.

    Returns:
        angular velocity to apply to the body to reach the specified setpoint
    """
    # quaternion attitude control law, qe is rotation from q to qd
    qe = q.inverse * qd
    # using sin(alpha/2) scaled rotation axis as attitude error (see quaternion definition by axis angle)
    # also taking care of the antipodal unit quaternion ambiguity
    return 2 * np.sign(qe[0]+1e-10) * np.array([qe[1], qe[2], qe[3]])

def qcontrol_reduced(q = Quaternion(), qd = Quaternion(), yw = 1):
    """
    Calculate angular velocity to get from current to desired attitude
    The body yaw axis has less priority then roll and pitch hence the name "reduced".

    Params:
        q: [optional] current body attitude quaternion. Defaults to level.
        qd: [optional] desired body attitude quaternion setpoint. Defaults to level.
        yw: [optional] desired body attitude quaternion setpoint. Defaults to full yaw priority.

    Returns:
        angular velocity to apply to the body to reach the specified setpoint
    """
    # extract body z-axis and desired body z-axis
    ez = dcm_z(q)
    ezd = dcm_z(qd)
    # reduced rotation that only aligns the body z-axis
    qd_red = vtoq(ez, ezd)
    # transform rotation from current to desired z-axis
    # into a world frame reduced desired attitude
    qd_red *= q

    # mix full and reduced desired attitude using yaw weight
    q_mix = qd_red.inverse * qd
    q_mix *= np.sign(q_mix[0])
    # catch numerical problems with the domain of acosf and asinf
    q_mix.q = np.clip(q_mix.q, -1, 1)
    qd = qd_red * Quaternion(cos(yw * acos(q_mix[0])), 0, 0, sin(yw * asin(q_mix[3])))

    # quaternion attitude control law, qe is rotation from q to qd
    qe = q.inverse * qd
    # using sin(alpha/2) scaled rotation axis as attitude error (see quaternion definition by axis angle)
    # also taking care of the antipodal unit quaternion ambiguity
    return 2 * np.sign(qe[0]+1e-10) * np.array([qe[1], qe[2], qe[3]])

def ftoq(f = np.array([0,0,1]), yaw = 0):
    """
    Calculate a desired attitude from 3D thrust and yaw in world frame
    Assuming a default multicopter configuration.

    Note: 0 yaw in world frame results in body x-axis projection
    onto world x-y-plane being aligned with world x-axis.

    Params:
        f: [optional] desired 3D thrust vector in world frame. Defaults to upwards.
        yaw: [optional] desired body yaw.

    Returns:
        attitude quaternion setpoint which meets the given goals
    """
    # align z-axis with thrust
    body_z = f / la.norm(f)
    # choose body x-axis orthogonal to z and with given yaw
    yaw_direction = np.array([-sin(yaw), cos(yaw), 0])
    body_x = np.cross(yaw_direction, body_z)
    body_x /= la.norm(body_x)
    # choose body y-axis right hand orthogonal to x- and z-axis
    body_y = np.cross(body_z, body_x)
    # assemble attitude matrix and quaternion from base vectors
    Rd = np.column_stack((body_x, body_y, body_z))
    return Quaternion(matrix=Rd)

def dcm_z(q = Quaternion()):
    """
    Calculate the body z-axis base vector from body attitude.

    Params:
        q: [optional] body attitude quaternion. Defaults to level.

    Returns:
        unit base vector of body z-axis in world frame
    """
    # convert quaternion to rotation matrix
    R = q.rotation_matrix
    # take last column vector
    return R[:,2]

def vtoq(src = np.array([0,0,1]), dst = np.array([0,0,1]), eps = 1e-5):
    """
    Calculate quaternion representing the shortest rotation
    from one vector to the other

    Params:
        src: [optional] Source 3D vector from which to start the rotation. Defaults to upwards direction.
        dst: [optional] Destination 3D vector to which to rotate. Defaults to upwards direction.
        eps: [optional] numerical thershold to catch 180 degree rotations. Defaults to 1e-5.

    Returns:
        quaternion rotationg the shortest way from src to dst
    """
    q = Quaternion()
    cr = np.cross(src, dst)
    dt = np.dot(src, dst)

    if(la.norm(cr) < eps and dt < 0):
        # handle corner cases with 180 degree rotations
        # if the two vectors are parallel, cross product is zero
        # if they point opposite, the dot product is negative
        cr = np.abs(src)
        if(cr[0] < cr[2]):
            if(cr[0] < cr[2]):
                cr = np.array([1,0,0])
            else:
                cr = np.array([0,0,1])
        else:
            if(cr[1] < cr[2]):
                cr = np.array([0,1,0])
            else:
                cr = np.array([0,0,1])
        q[0] = 0
    else:
        # normal case, do half-way quaternion solution
        q[0] = dt + sqrt(np.dot(src,src) * np.dot(dst,dst))
    q[1] = cr[0]
    q[2] = cr[1]
    q[3] = cr[2]
    return q.normalised

# setup 3D plot
fig = plt.figure()
ay = fig.add_subplot(111, projection='3d')
ay.set_xlim([-5, 1])
ay.set_ylim([-5, 1])
ay.set_zlim([-5, 1])
ay.set_xlabel('X')
ay.set_ylabel('Y')
ay.set_zlabel('Z')

# initialize state
steps = 0
q = qd = Quaternion() # atttitude state
v = np.array([0.,0.,0.]) # velocity state
p = np.array([0.,0.,0.]) # position state

# specify goal and parameters
pd = np.array([-5,0,0]) # desired position
yd = radians(180) # desired yaw
dt = 0.2 # time steps

# run simulation until the goal is reached
while steps < 1000 and (not np.isclose(p, pd).all() or (q.inverse * qd).degrees > 0.1):
    # plot current vehicle body state abstraction
    plotrotc(q, p)

    # run minimal position & velocity control
    vd = (pd - p)
    fd = (vd - v)
    fd += np.array([0,0,1]) # "gravity"

    # run attitude control
    qd = ftoq(fd, yd)
    thrust = np.dot(fd, dcm_z(q))
    w = 3*qcontrol_reduced(q, qd, 0.4)

    # propagate states with minimal, ideal simulation
    q.integrate(w, dt)
    f = dcm_z(q) * thrust
    v += f - np.array([0,0,1])
    p += dt*v

    # print progress
    steps += 1
    print(steps, '\t', p)

plt.show()