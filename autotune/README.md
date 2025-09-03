# Installation using a virtual environment

## Using Poetry (recommended)
### Create a new environment and install the dependencies
1. Get [poetry](https://python-poetry.org/docs/): `curl -sSL https://install.python-poetry.org | python3 -`
2. In this directory, simply run `poetry install`

### Start the GUI
```
poetry run python3 autotune.py
```

## Using venv
### Create a new environment
```
python3.9 -m venv virtualenv-test
source virtualenv-test/bin/activate
```

### Install the dependencies
```
pip3 install numpy scipy pyulog control pyqt5
```

### Start the GUI
```
python3 autotune.py
```

![image](https://github.com/user-attachments/assets/fcdf5c25-d92d-4487-9736-e77f6576d180)

# Tuning Worflow

## Fixed-Wing Rate Tuning

**Prerequisites**:

- Requires valid airspeed measurement on vehicle
- Correctly set FW_AIRSPD_TRIM

### Flight Maneuvers

Using default PX4 gains. Perform these maneuvers separately for each axis: roll, pitch, and yaw.

1. Requires valid airspeed measurement, and a correctly set FW_AIRSPD_TRIM.
2. Fly in stabilized mode and start from level flight.
3. Apply a manual sine chirp input (sine wave with increasing frequency) on the selected axis. Avoid any other control inputs simultaneously.
4. For pitch and yaw: The amplitude of the input should not be too high, but should be enough to overcome noise.
5. Repeat the maneuver a few times if necessary to capture good data.

### Using the Tool

#### Load the flight log and select the relevant tuning loop (roll, pitch, or yaw):

![image](https://github.com/user-attachments/assets/f962a004-eae8-433f-9b7e-adc618dcbeae)

#### Window selection:

1. Select the portion of the log where the maneuver occurred.
2. Ensure the window starts from level flight and covers the full maneuver.
3. (Experimental) Use the coherence plot to check whether your input-output data have a strong enough linear relationship at the desired frequencies. For fixed-wings: \
   i. We are interested in frequencies between 0.5-10Hz. \
   ii. Values > 0.6 are sufficient. \
   iii. If the coherence function is oscillating dramatically, the identified model may be unreliable at those frequencies.
4. Once you have identified the window, load the selection into the tool.

<table>
  <tr>
    <td align="center">
      <b>Good window selection</b><br>
      <img src="https://github.com/user-attachments/assets/25c30295-6825-4cb4-828e-47eaa17e1557" width="400"/><br>
      <i>Starts from level flight and covers the full maneuver. Coherence is sufficient.</i>
    </td>
    <td align="center">
      <b>Bad window selection</b><br>
      <img src="https://github.com/user-attachments/assets/1e05d411-37cf-42ab-8d6f-0cbbbb00921c" width="400"/><br>
      <i>Does not isolate the maneuver. Coherence is insufficient. </i>
    </td>
  </tr>
</table>

#### Model Verification

1. Check that the estimated dynamic model correctly tracks the actual aircraft output.
2. If not, adjust the system order (number of poles and zeros) and/or delay parameters.
3. Verify that the system is minimum-phase (i.e., zeroes lie inside the unit circle).

<img src="https://github.com/user-attachments/assets/87168b77-1b52-4d6f-bb4d-ace3f76fe4ee" width="700"/>

#### Tuning Gains

1. Set all gains to zero
2. Increase the P-gain until the step response shows oscillations. Then reduce the P-gain by approximately 50%:

   <b>Increase P-gain</b><br>
   <img src="https://github.com/user-attachments/assets/db05a018-f9fb-4b3c-9f55-a4d7a577251f"/>

   <b>Decrease by 50%</b><br>
   <img src="https://github.com/user-attachments/assets/77dc38aa-f829-4878-a58e-32d775fa542c"/>

3. Adjust the I-gain to achieve a good step response with acceptable steady-state error.

   ![image](https://github.com/user-attachments/assets/78f3bca5-2642-44e2-b3f2-7444fc4ffa40)

4. Fine-tune both gains as needed. At the 1 second mark, a disturbance is injected into the system. The tuned response to this disturbance is visible in the step response plot.

5. Use the Bode plot to confirm that the resonant frequency peak remains below 0dB.

   ![image](https://github.com/user-attachments/assets/86d99295-43a8-47a9-b052-324ab5ac3082)

### Applying the Gains

1. The gains shown in the parallel form correspond to the PX4 rate controller parameters.
   ![image](https://github.com/user-attachments/assets/2db77238-bd33-4454-af1f-a38901571e88)

2. If the identified parameters are strongly different to the currently set ones (more than 30% for multiple axes), an incremental application while flying is recommended (not prior to takeoff, during takeoff, or during landing).

3. For all loops tuned with the strategy outlined in this guide: increase the respective integrator limit parameters to 1 (FW_RR_IMAX, FW_PR_IMAX, FW_YR_IMAX).
