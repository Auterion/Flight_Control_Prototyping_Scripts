# Multicopter control allocation
The algorithm implements the following idea: the signals going to the actuators are computed by the multiplication of the control vector **m** (desired rotational and linear accelerations/torques) by the control allocation matrix **P**. Then, some constraints (e.g.: vertical thrust or yaw acceleration) are relxed to minimize the saturation of the actuators. 
In PX4, the actuator effectiveness matrix **B** is defined in the toml files by the geometry of the vehicle and the control allocation matrix **P** is computed in the python script during compilation and is given to the mixer (nothing different so far).

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{m}&space;=&space;\mathbf{Bu}&space;\Leftrightarrow&space;\mathbf{u}&space;=&space;\mathbf{Pm}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{m}&space;=&space;\mathbf{Bu}&space;\Leftrightarrow&space;\mathbf{u}&space;=&space;\mathbf{Pm}" title="\mathbf{m} = \mathbf{Bu} \Leftrightarrow \mathbf{u} = \mathbf{Pm}" /></a>

Instead of decreasing the input and re-mixing several times as it is done at the moment, this algorithm makes use of the vectors given by the control allocation matrix **P** to directly manipulate the output vector. Those vectors are orthogonal (4D space) and give the direct information about how to change the outputs in order to modify an input. In the case of a symmetric quad, the result is the same as before: add the same value to the 4 motors to modify thrust without changing roll/pitch/yaw accelerations.
This strategy gives a more general approach and gives a good base for 6D control allocation of fully actuated vehicles.

## Saturation minimization
Given that each axes given by the columns of the control allocation are orthogonal, we only need to optimize a single variable at a time an the optimal gain ca be find in a few simple steps:
1. Find the gain <a href="https://www.codecogs.com/eqnedit.php?latex=k_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k_1" title="k_1" /></a> that unsaturates the most saturated actuator
2. Apply the gain to unsaturate it
3. repeat 1. to find <a href="https://www.codecogs.com/eqnedit.php?latex=k_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k_2" title="k_2" /></a>
4. compute the optimal gain

The optimal gain <a href="https://www.codecogs.com/eqnedit.php?latex=k_{opt}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k_{opt}" title="k_{opt}" /></a> that minimizes the saturations is given by

<a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{u}_{opt}&space;=&space;\frac{\mathbf{u}_1&space;&plus;&space;\mathbf{u}_2}{2}&space;\\&space;=&space;\frac{\mathbf{u}&space;&plus;&space;k_1\mathbf{\Delta}_u&space;&plus;&space;\mathbf{u}&space;&plus;&space;k_1&space;\mathbf{\Delta}_u&space;&plus;&space;k_2&space;\mathbf{\Delta}_u}{2}&space;\\&space;=&space;\mathbf{u}&space;&plus;&space;(k_1&space;&plus;&space;\frac{k_2}{2})\mathbf{\Delta}_u&space;\\&space;=&space;\mathbf{u}&space;&plus;&space;k_{opt}&space;\mathbf{\Delta}_u" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{u}_{opt}&space;=&space;\frac{\mathbf{u}_1&space;&plus;&space;\mathbf{u}_2}{2}&space;\\&space;=&space;\frac{\mathbf{u}&space;&plus;&space;k_1\mathbf{\Delta}_u&space;&plus;&space;\mathbf{u}&space;&plus;&space;k_1&space;\mathbf{\Delta}_u&space;&plus;&space;k_2&space;\mathbf{\Delta}_u}{2}&space;\\&space;=&space;\mathbf{u}&space;&plus;&space;(k_1&space;&plus;&space;\frac{k_2}{2})\mathbf{\Delta}_u&space;\\&space;=&space;\mathbf{u}&space;&plus;&space;k_{opt}&space;\mathbf{\Delta}_u" title="\mathbf{u}_{opt} = \frac{\mathbf{u}_1 + \mathbf{u}_2}{2} \\ = \frac{\mathbf{u} + k_1\mathbf{\Delta}_u + \mathbf{u} + k_1 \mathbf{\Delta}_u + k_2 \mathbf{\Delta}_u}{2} \\ = \mathbf{u} + (k_1 + \frac{k_2}{2})\mathbf{\Delta}_u \\ = \mathbf{u} + k_{opt} \mathbf{\Delta}_u" /></a>

By identification, we find that

<a href="https://www.codecogs.com/eqnedit.php?latex=k_{opt}&space;=&space;k_1&space;&plus;&space;\frac{k_2}{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k_{opt}&space;=&space;k_1&space;&plus;&space;\frac{k_2}{2}" title="k_{opt} = k_1 + \frac{k_2}{2}" /></a>

## Axis prioritization
The strategy used to prioritize the axes is to allocate the axes starting with the higher priority and ending with the lower priority.

Three modes of prioritization are currently supported:

**Normal (safe) mode** - Roll/Pitch accelerations are allocated, Z thrust is used to unsaturate the actuators reaching the higher saturation (not the lower saturation, no "boosting" in this mode), roll/pitch acceleration are reduced if needed to unsaturate, then yaw acceleration is allocated and reduced if needed to fit into the remaining space.

**Airmode XY** - Roll/Pitch accelerations are allocated, Z thrust is used to unsaturate the actuators (Z thrust can be increased), then yaw acceleration is allocated and reduced if needed to fit into the remaining space.

**Airmode XYZ** - Roll/Pitch/Yaw accelerations are allocated and Z thrust is used to unsaturate the actuators.

### When to use which mode?
The normal mode is the default mode as it is the safest one and should be used for all the new platform bringups.

The Airmode XY is useful for some fixedwings VTOLs (e.g.: quadplane) as it helps to maintain stability when the wing has lift but the control surfaces become uneffective. It can also be used on multicopters with weak yaw control.

The Airmode XYZ is useful for racer quads that have good yaw authority and require full attitude control even at zero and full throttle.
