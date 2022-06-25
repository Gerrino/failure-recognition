# failure-recognition
Repository for recognizing failures in machining

## Installation


## git

```bash
git clone https://github.com/Gerrino/failure-recognition.git
git pull
```

## Windows

```bash
cd failure-recognition-signal-processing
py -3 -m venv .venv # create a virtual environment
.venv/scripts/activate # activate the virtual environment
pip install setuptools
```

## Linux

```bash
cd failure-recognition-signal-processing
python3 -m venv .venv
source .venv/bin/activate
```


## How to install subpackages

I) Create + activate a virtual environment within a subpackage (containing a setup.py) (see https://packaging.python.org/en/latest/guides/packaging-namespace-packages/)

II) To install that subpackage run:
```bash
pip install -e .
```
III)  To install other subpackages within the failure-recognition namespace package run:
```bash
cd ../failure-recognition-die-casting
pip install .
```