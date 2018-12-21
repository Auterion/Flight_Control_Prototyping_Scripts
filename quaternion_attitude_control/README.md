# Quaternion attitude control
The algorithm implements an attitude control strategy for flying vehicles especially multicopters using unit quaternion rotation math.

To run the script you need Python 3 (tested on 3.7.1) and the following python modules:
```
python -m pip install numpy matplotlib pyquaternion
```

Command to run the script:
```
python quaternion_attitude_control_test.py
```

Based on [this paper](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf) and [this master thesis](https://drive.google.com/uc?e=pdf&id=1jVABlvL4eGU_IM6f_tUnRhjHgKOIAlcP).
