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

