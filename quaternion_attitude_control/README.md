# Quaternion attitude control
The algorithm implements an attitude control strategy for flying vehicles especially multicopters using unit quaternion rotation math.

## Setup
To run the script you need Python 3 (tested on 3.7.1) and the following python modules:
```
sudo python3 -m pip install numpy matplotlib pyquaternion
```

Command to run the script:
```
python3 quaternion_attitude_control_test.py
```

## Details
The algorithm is based on [this paper](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf) and [this master thesis](https://drive.google.com/uc?e=pdf&id=1jVABlvL4eGU_IM6f_tUnRhjHgKOIAlcP).

The visualization shows an abstract vehicle state for each time step in the form of three arrows. The common origin of the arrows is the vehicle's 3D position at the timestep and the arrows point in x-, y- and z-axis (red, gree, blue) direction of the vehicle's current body frame and hence represent its attitude relative to the world frame.
