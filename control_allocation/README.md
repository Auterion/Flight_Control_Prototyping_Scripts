# Multicopter control allocation
The general idea is the same as it is now: the output is computed and then we try to relax some constraints (e.g.: vertical thrust or yaw acceleration) to minimize the saturation. The actuator effectiveness matrix B is defined in the toml files and the control allocation matrix P is computed in the python script during compilation and is given to the mixer (nothing different so far).
However, instead of decreasing the input and re-mixing several times, directly change the mixed output using the vectors given by the control allocation matrix P. Those vectors are orthogonal (4D space) and give the direct information about how to change the outputs in order to modify an input (in the case of a symmetric quad, the result is the same as before: add the same value to the 4 motors to modify thrust without changing roll/pitch/yaw accelerations).
This gives a more general approach and gives a good base for 6D control allocation of fully actuated vehicles.

## Saturation minimization
Given that each axes given by the columns of the control allocation are orthogonal, we only need to optimize a single variable at a time an the optimal gain ca be find in a few simple steps:
1. Find the gain <a href="https://www.codecogs.com/eqnedit.php?latex=k_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k_1" title="k_1" /></a> that unsaturates the most saturated actuator
2. Apply the gain to unsaturate it
3. repeat 1. to find <a href="https://www.codecogs.com/eqnedit.php?latex=k_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k_2" title="k_2" /></a>
4. compute the optimal gain

The optimal gain <a href="https://www.codecogs.com/eqnedit.php?latex=k_{opt}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k_{opt}" title="k_{opt}" /></a> that minimizes the saturations is given by

<a href="https://www.codecogs.com/eqnedit.php?latex=u_{opt}&space;=&space;\frac{u_1&space;&plus;&space;u_2}{2}&space;\\&space;=&space;\frac{u&space;&plus;&space;k_1\Delta_u&space;&plus;&space;u&space;&plus;&space;k_1&space;\Delta_u&space;&plus;&space;k_2&space;\Delta_u}{2}&space;\\&space;=&space;u&space;&plus;&space;(k_1&space;&plus;&space;\frac{k_2}{2})\Delta_u&space;\\&space;=&space;u&space;&plus;&space;k_{opt}&space;\Delta_u" target="_blank"><img src="https://latex.codecogs.com/gif.latex?u_{opt}&space;=&space;\frac{u_1&space;&plus;&space;u_2}{2}&space;\\&space;=&space;\frac{u&space;&plus;&space;k_1\Delta_u&space;&plus;&space;u&space;&plus;&space;k_1&space;\Delta_u&space;&plus;&space;k_2&space;\Delta_u}{2}&space;\\&space;=&space;u&space;&plus;&space;(k_1&space;&plus;&space;\frac{k_2}{2})\Delta_u&space;\\&space;=&space;u&space;&plus;&space;k_{opt}&space;\Delta_u" title="u_{opt} = \frac{u_1 + u_2}{2} \\ = \frac{u + k_1\Delta_u + u + k_1 \Delta_u + k_2 \Delta_u}{2} \\ = u + (k_1 + \frac{k_2}{2})\Delta_u \\ = u + k_{opt} \Delta_u" /></a>

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

The Airmode XY is useful some fixedwings VTOLs (e.g.: quadplane) as it helps to maintain stability when the wing has lift but the control surfaces become uneffective.

The Airmode XYZ is useful for racer quads that have good yaw authority and require full attitude control even at zero and full throttle.
