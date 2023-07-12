"""Module for accessing/visualizing csv files in the tsfresh format
"""

from pathlib import Path
from typing import Union
import pandas as pd
from pandas import DataFrame as df
from matplotlib import pyplot as plt

TIME_ID = "timeseries_me_count"
TIME_SERIES_ID = "TimeSeries_ME_id"


def data_frame_from_csv_dir(csv_dir: Union[Path, str], csv_count: int = None):
    """Given a directory containing csv files create a
    pandas dataframe

    The csv files have the format: \d+_\d+.*csv

    Returns
    ---
    pd.DataFrame
    """
    csv_dir = Path(csv_dir)
    _csv_files = list(csv_dir.iterdir())
    _csv_files.sort(key=lambda x: int(x.name.split("_")[0]))
    print("Found csv files: ", len(_csv_files))
    data_frame = df()
    for i, _csv_file in enumerate(_csv_files):
        if i == csv_count:
            break
        read_frame = pd.read_csv(
            _csv_file, delimiter=";", encoding="ISO-8859-1")
        if i == 0:
            data_frame = read_frame
        else:
            data_frame = pd.concat([data_frame, read_frame], axis=0)
    return data_frame


def plot_column(data_frame: pd.DataFrame, column_name: str, max_timeseries: int = None):
    """Plot a column by name given a pandas dataframe in the tsfresh format
    """
    given_ids = list(set(data_frame[TIME_SERIES_ID]))
    if isinstance(max_timeseries, int):
        given_ids = given_ids[0:min(max_timeseries, len(given_ids)-1)]
    print(f"Plotting timeseries {given_ids}")
    for id in given_ids:
        time_vector = data_frame.query(f"{TIME_SERIES_ID} == {id}")
        plt.plot(time_vector[TIME_ID], time_vector[column_name])
    plt.show()
    pass


if __name__ == "__main__":
    csv_dir = Path(
        "examples/bueh_example/parts")
    data_frame = data_frame_from_csv_dir(csv_dir, csv_count=2)
    plot_column(data_frame, "19_Temp11", max_timeseries=None)
    pass
