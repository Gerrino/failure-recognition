"""Init module for the smac-recognizer"""

DEFAULT_MAX_INT = 100
DEFAULT_MIN_INT = 0
DEFAULT_MAX_FLOAT = 10
DEFAULT_MIN_FLOAT = -10
DEFAULT_INT = 1
DEFAULT_FLOAT = 0.1

PATH_DICT = {
    "features": "/home/gerritnoske/Documents/Projekt_Henne/tsfreshlist.txt",
    "timeSeries": "/home/gerritnoske/Documents/Projekt_Henne/NASA_tsfresh_format_all.csv",
    "label": "/home/gerritnoske/Documents/Projekt_Henne/label.csv",
    "testSettings": "/home/gerritnoske/Documents/Projekt_Henne/testsettings.csv",
}

SCENARIODICT = {
    "run_obj": "quality",  # we optimize quality (alternative runtime)
    "runcount-limit": 20,  # max. number of function evaluations; for this example set to a low number
    "deterministic": "true",
    "memory_limit": 3072,  # adapt this to reasonable value for your hardware
    # "cutoff": .1,
}
