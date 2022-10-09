"""Init module for the smac-recognizer"""

SMAC_SEED = 42

PATH_DICT = {
    "features": "./examples/tsfresh_features.json",
    "forest_params": "./examples/tsfresh_random_forest.json",
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
