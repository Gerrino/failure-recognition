"""
TimeSeries_ME_id: id of timeseries
timeseries_Number: global row counter
timeseries_me_count: time
01_Temp01
02_Temp02
"""

import csv
from math import sqrt
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
from failure_recognition.signal_processing.read_tsfresh_format import TIME_SERIES_ID, TIME_ID, data_frame_from_csv_dir
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame as df





def convert_flat_series(flat_rows: list, time_series_id: str =
                        TIME_SERIES_ID, time_id: str = TIME_ID) -> Tuple[list, Dict[int, any]]:
    time_series_map: Dict[int, list] = {}

    title_row = flat_rows.pop(0)
    time_index = title_row.index(time_id)
    id_index = title_row.index(time_series_id)

    for row in flat_rows:
        row: List
        time_value = int(row[time_index])
        id_value = str(row[id_index])
        if time_value == 0:
            time_series_map[id_value] = []
        time_series_map[id_value].append(row)
    time_series_map = pd.DataFrame(time_series_map)
    return title_row, time_series_map


def get_plausible_indices(vector: pd.DataFrame):
    """Get the indices of plausbile data points
    """
    vector = pd.DataFrame(vector)
    plausibility_factor = 1000
    plausibility_last = 10

    mean = vector.mean()
    valid_indices = []
    dropped_indices = []
    loop_dropped = True
    while loop_dropped:
        loop_dropped = False
        _index = 0
        while _index < vector.size:
            variance = vector.var()[0]
            std = sqrt(variance)
            value = vector.values[_index][0]
            plausible = True
            median = vector.median()[0]
            if value == 0:
                plausible = False
            if value > plausibility_factor*median:
                plausible = False
            if _index > 0:
                diff_last = abs(value - vector.values[_index-1][0])
                if diff_last > median * plausibility_last:
                    plausible = False
            if not plausible:
                loop_dropped = True
                vector = vector.drop(vector.index[_index])
                dropped_indices.append(_index)
                _index -= 1
            _index += 1
    return dropped_indices





def write_rows(csv_dir, all_csv_files):
    with open(csv_dir / "out.csv",  "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=";", quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        for i, rows in enumerate(all_csv_files):
            if i > 0 and rows[0][0] == 'TimeSeries_ME_id':
                rows = rows[1:]
            print(f"Wrote {len(rows)} rows!")
            csv_writer.writerows(rows)
        pass



time_id_6 = data_frame.query(f"{TIME_SERIES_ID} == 6")
# timeseries_map = convert_flat_series(data_frame)

# time_index = title_row.index(TIME_ID)
# time_row_map = {k: v[time_index, :] for k, v in timeseries_map.items()}


non_plausible = get_plausible_indices(time_id_6["13_V01"])
time_id_6_plausbie = time_id_6[TIME_ID].drop(
    time_id_6[TIME_ID].index[non_plausible])
time_id_6_v01_plausible = time_id_6["13_V01"].drop(
    time_id_6["13_V01"].index[non_plausible])
plt.plot(time_id_6[TIME_ID], time_id_6["13_V01"])
plt.plot(time_id_6_plausbie, time_id_6_v01_plausible)
plt.show()
pass
