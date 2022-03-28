import matplotlib.pylab as plt
from pyulog import ULog
from pyulog.px4 import PX4ULog
import numpy as np
import quaternion
from scipy import optimize

def getAllData(logfile, instance=1):
    log = ULog(logfile)

    v_local = np.matrix([getData(log, 'vehicle_local_position', 'vx'),
              getData(log, 'vehicle_local_position', 'vy'),
              getData(log, 'vehicle_local_position', 'vz')])

    t_v_local = ms2s(getData(log, 'vehicle_local_position', 'timestamp'))

    accel = np.matrix([getData(log, 'sensor_combined', 'accelerometer_m_s2[0]'),
              getData(log, 'sensor_combined', 'accelerometer_m_s2[1]'),
              getData(log, 'sensor_combined', 'accelerometer_m_s2[2]')])
    t_accel = ms2s(getData(log, 'sensor_combined', 'timestamp'))

    q = np.matrix([getData(log, 'vehicle_attitude', 'q[0]'),
              getData(log, 'vehicle_attitude', 'q[1]'),
              getData(log, 'vehicle_attitude', 'q[2]'),
              getData(log, 'vehicle_attitude', 'q[3]')])
    t_q = ms2s(getData(log, 'vehicle_attitude', 'timestamp'))

    dist_bottom = getData(log, 'vehicle_local_position', 'dist_bottom')
    t_dist_bottom = ms2s(getData(log, 'vehicle_local_position', 'timestamp'))

    (t_aligned, v_body_aligned, accel_aligned) = alignData(t_v_local, v_local, t_accel, accel, t_q, q, t_dist_bottom, dist_bottom)

    t_aligned -= t_aligned[0]

    return (t_aligned, v_body_aligned, accel_aligned)

def alignData(t_v, v_local, t_accel, accel, t_q, q, t_dist_bottom, dist_bottom):
    len_accel = len(t_accel)
    len_q = len(t_q)
    len_db = len(t_dist_bottom)
    i_a = 0
    i_q = 0
    i_db = 0
    v_body_aligned = np.empty((3,0))
    accel_aligned = np.empty((3,0))
    t_aligned = []

    for i_v in range(len(t_v)):
        t = t_v[i_v]
        while t_accel[i_a] <= t and i_a < len_accel-1:
            i_a += 1
        while t_q[i_q] <= t and i_q < len_q-1:
            i_q += 1
        while t_dist_bottom[i_db] <= t and i_db < len_db-1:
            i_db += 1

        if dist_bottom[i_db] < 1.0:
            continue

        qk = np.quaternion(q[0, i_q],q[1, i_q],q[2, i_q],q[3, i_q])
        q_vl = np.quaternion(0, v_local[0, i_v], v_local[1, i_v], v_local[2, i_v])
        q_vb = qk.conjugate() * q_vl * qk # Get velocity in body frame
        vb = quaternion.as_float_array(q_vb)[1:4]

        v_body_aligned = np.append(v_body_aligned, [[vb[0]], [vb[1]], [vb[2]]], axis=1)
        accel_aligned = np.append(accel_aligned, accel[:, i_a-1], axis=1)
        t_aligned.append(t)

    return (t_aligned, v_body_aligned, np.asarray(accel_aligned))

def getData(log, topic_name, variable_name, instance=0):
    variable_data = np.array([])
    for elem in log.data_list:
        if elem.name == topic_name:
            if instance == elem.multi_id:
                variable_data = elem.data[variable_name]
                break

    return variable_data

def ms2s(time_ms):
    return time_ms * 1e-6

def run(logfile):
    # TODO:
    # - accumulate accel instead of smaple picking
    # - use in air data only
    (t, v_body, a_body) = getAllData(logfile)

    rho = 1.15 # air densitiy
    rho15 = 1.225 # air density at 15 degC

    # Concatenate X and Y axes if the body is symmetric or if the bluff body drag is negligible (ballistic coefficient -> inf)
    U = np.append(v_body[0], v_body[1])
    Y = np.append(a_body[0], a_body[1])

    # x[0]: momentum drag, scales with v
    # x[1]: ballistic coefficient, scales with v^2
    computed_output = lambda x: -U * x[0] - 0.5 * rho * U**2 * np.sign(U) / x[1]
    J = lambda x: np.sum(np.power(abs(Y-computed_output(x)), 2.0))
    x0 = [0.1, 10]
    res = optimize.minimize(J, x0, method='nelder-mead', options={'disp': True})
    print(f"BCoef = {res.x[1]}, MCoef = {res.x[0] / np.sqrt(rho / rho15)}")

    # Plot data
    plt.figure(1)
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t, v_body[0])
    ax1.plot(t, v_body[1])

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(t, a_body[0])
    ax2.plot(t, a_body[1])
    ax2.plot(np.append(t, t), computed_output(res.x))
    plt.show()

if __name__ == '__main__':
    import os
    import argparse

    # Get the path of this script (without file name)
    script_path = os.path.split(os.path.realpath(__file__))[0]

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Estimate mag biases from ULog file')

    # Provide parameter file path and name
    parser.add_argument('logfile', help='Full ulog file path, name and extension', type=str)
    args = parser.parse_args()

    logfile = os.path.abspath(args.logfile) # Convert to absolute path

    run(logfile)
