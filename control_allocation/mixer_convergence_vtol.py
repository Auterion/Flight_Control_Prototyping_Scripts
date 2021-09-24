import numpy as np
from mixerlib import normal_mode, mix_forward_thrust_and_yaw

import matplotlib.pylab as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_arrow(start,end,figure_handle, axis_handle=None):
    style="Simple,head_length=15,head_width=15,tail_width=10"
    vec = Arrow3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], arrowstyle=style)

    if axis_handle == None:
        axis_handle = figure_handle.add_subplot(111, projection='3d')

    axis_handle.add_artist(vec)

    return axis_handle

def mix_hover_prio(controls, P):

    # extract roll, pitch and thrust Z setpoints
    m_sp = np.matrix([controls[0,0] ,controls[0,1],0, controls[0,4]]).T
    u, u_opt = normal_mode(m_sp, P, np.matrix([0,0,0]).T, np.matrix([1,1,1]).T)
    return [u, u_opt]

def calculate_forward_thrust_limits(u_z):
    u_min = np.zeros(2)
    u_max = np.zeros(2)

    for index in range(2):
        # total thrust needs to be smaller than one in magnitude
        u_max[index] = np.sqrt(1.0 - u_z[index]**2)


    return np.matrix(u_min),np.matrix(u_max)


# demanded torque / thrust vector
torque_demanded = np.array([0, 0, 0])
thrust_demanded = np.array([1, -0.5])

torque_thurst_demanded = np.matrix(np.concatenate((torque_demanded, thrust_demanded), axis=None))


# vehicle configuration for convergence VTOL
# Motor 1 -> front right motor
# Motor 2 -> front left motor
# Motor 3 -> back motor (non-tilting)

# Actuator definition
# Actuator 1 -> Force in X direction of motor 1
# Actuator 2 -> Force in X direction of motor 2
# Actuator 3 -> Force in Z direction of motor 1
# Actuator 4 -> Force in Z direction of motor 2
# Actuator 5 -> Force in Z direction of motor 3

motor1_pos = np.array([0.1, 0.3,0])
motor2_pos = np.array([0.1, -0.3,0])
motor3_pos = np.array([-0.3, 0, 0])

# torque vector that one unit of the respective actuator would generate
actuator1_torque = np.cross(motor1_pos, np.array([1, 0, 0]))
actuator2_torque = np.cross(motor2_pos, np.array([1, 0, 0]))
actuator3_torque = np.cross(motor1_pos, np.array([0, 0, -1]))
actuator4_torque = np.cross(motor2_pos, np.array([0, 0, -1]))
actuator5_torque = np.cross(motor3_pos, np.array([0, 0, -1]))


# actuator effectiveness matrix
B = np.matrix([
	[actuator1_torque[0], actuator2_torque[0], actuator3_torque[0], actuator4_torque[0], actuator5_torque[0]],
	[actuator1_torque[1], actuator2_torque[1], actuator3_torque[1], actuator4_torque[1], actuator5_torque[1]],
	[actuator1_torque[2], actuator2_torque[2], actuator3_torque[2], actuator4_torque[2], actuator5_torque[2]],
	[1, 1, 0, 0, 0],
	[0, 0, -1, -1, -1]])

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

P = np.linalg.pinv(B)

# do some weird scaling
max_val_rpy = np.max(P[:,0:3])
max_thr = np.max(P[:,3:5])
P[:,0:3] = P[:,0:3] / max_val_rpy
P[:,3:5] = P[:,3:5] / max_thr

# calculate mixer matrix only for actuating, roll, pitch and thrust in Z direction
P_mc = P
P_mc = np.delete(P_mc, [0,1],0)
P_mc = np.delete(P_mc, [3],1)

u_hover_orig, u_hover_opt = mix_hover_prio(torque_thurst_demanded, P_mc)

# calculate forward thrust (x direction limits)
thrust_x_min,thrust_x_max = calculate_forward_thrust_limits(u_hover_opt)

# calculate mixer mixer which only takes into account forward thrust and yaw
P_fx_yaw = P
P_fx_yaw = np.delete(P_fx_yaw, [2,3,4],0)
P_fx_yaw = np.delete(P_fx_yaw, [0,1,4],1)

# compute forward thrust values, given the limits imposed by the other actuators already being mixed
input_vector = np.matrix([torque_thurst_demanded[0,2], torque_thurst_demanded[0,3]]).T
u_fx_orig, u_fx_opt = mix_forward_thrust_and_yaw(input_vector, P_fx_yaw, thrust_x_min.T, thrust_x_max.T)


# create the thrust vector of each motor
mot1_thrust = np.array([u_fx_opt[0], 0, u_hover_opt[0]])
mot2_thrust = np.array([u_fx_opt[1], 0, u_hover_opt[1]])
mot3_thrust = np.array([0, 0, u_hover_opt[2]])


# draw the results
fig = plt.figure()
ax = plot_arrow(motor1_pos, motor1_pos + mot1_thrust, fig)
ax = plot_arrow(motor2_pos, motor2_pos + mot2_thrust, fig, ax)
ax = plot_arrow(motor3_pos, motor3_pos + mot3_thrust, fig, ax)
ax.plot3D([-1,1], [0,0], [0,0])
ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(0,1))
plt.show()



