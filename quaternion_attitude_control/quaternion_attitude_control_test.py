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
from math import sin, cos, degrees, radians

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
    w = 3*qcontrol_full(q, qd)

    # propagate states with minimal, ideal simulation
    q.integrate(w, dt)
    f = dcm_z(q) * thrust
    v += f - np.array([0,0,1])
    p += dt*v

    # print progress
    steps += 1
    print(steps, '\t', p)

plt.show()