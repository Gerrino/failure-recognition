"""Init module for the smac-recognizer"""

DEFAULT_MAX_INT = 100
DEFAULT_MIN_INT = 0
DEFAULT_MAX_FLOAT = 10
DEFAULT_MIN_FLOAT = -10
DEFAULT_INT = 1
DEFAULT_FLOAT = 0.1

PATH_DICT = {
    "features": "./examples/tsfreshlist.txt",
    "timeSeries": "./examples/NASA_tsfresh_format_all.csv",
    "label": "./examples/label.csv",
    "testSettings": "./examples/testsettings.csv",
}

SCENARIO_DICT = {
    "run_obj": "quality",  # we optimize quality (alternative runtime)
    "runcount-limit": 20,  # max. number of function evaluations; for this example set to a low number
    "deterministic": "true",
    "memory_limit": 3072,  # adapt this to reasonable value for your hardware
    # "cutoff": .1,
}
