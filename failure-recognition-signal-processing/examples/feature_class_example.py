"""Module providing information about features for any error class"""

from typing import Dict
import pandas as pd


if __name__ == "__main__":
    classification: pd.DataFrame = pd.read_csv("./examples/dumps/buehler_data_classification_20230601.tsv", delimiter="\t")
    df_feature_state: pd.DataFrame = pd.read_pickle("./examples/dumps/timeseries_zdg_feature_state.pkl")    
    df_timeseries: pd.DataFrame = pd.read_pickle("./examples/dumps/timeseries_zdg.pkl")

    timeseries_id_to_classification_index = {v: k for k, v in enumerate(list(classification.Zyklus_Nummer))}

    timeseries_id_to_feature_row_index = {_id: index for index, _id in enumerate(set(df_timeseries.id))}    
    
    if len(timeseries_id_to_feature_row_index) != df_feature_state.shape[0]:
        raise ValueError("Feature matrix has unexpected shape!")   

    class_name = "fehler"
    feat_name = "variance"

    #select specific feature
    feat_columns = df_feature_state[[f for f in df_feature_state.columns if str(f).endswith(feat_name)]]
    num_sensors = len(feat_columns.columns)
    #get feature row from timeseries id
    timeseries_id_to_feature_row = {_id: feat_columns.iloc[index,:] for index, _id in enumerate(set(df_timeseries.id))}
    #select classification column
    classification_column = classification[class_name]
    # calculate list of (timeseries_id, classification)
    classification_result_ids = [(classification.Zyklus_Nummer.iloc[row_index], cell_value) for row_index, cell_value in enumerate(classification_column)]
    # calculate list of ([features], classification)
    classification_features = [(pd.DataFrame(timeseries_id_to_feature_row[x[0]]).transpose(), x[1]) for x in classification_result_ids]

    #dict mapping class result to feature matrix of feature 'feat_name'
    classification_result_features: Dict[int, pd.DataFrame] = {}
    for class_feature in classification_features:
        class_value = class_feature[1]
        class_value_feature_row = class_feature[0]
        if class_value in classification_result_features:
            classification_result_features[class_value] = pd.concat([classification_result_features[class_value], class_value_feature_row], axis=0)
        else:
            classification_result_features[class_value] = pd.DataFrame(data=class_value_feature_row)

    #check len
    for class_value, class_value_count in {k: len([v for v in classification_column if v == k]) for k in set(classification_column)}.items():
        if classification_result_features[class_value].shape[0] != class_value_count:
            raise ValueError("Classification feature matrix has unexpected size (rows)!")
        if classification_result_features[class_value].shape[1] != num_sensors:
            raise ValueError("Classification feature matrix has unexpected size (columns)!")
        
    # show feature matrix for 'fehler == error_value'
    error_value = 1
    sensor_no = 0
    feature_matrix_error_value = classification_result_features[error_value]
    print(f"Feature matrix for error class {error_value}\n", feature_matrix_error_value)
    # show feature vector for 'fehler == 1' and the first sensor
    print(f"Feature vector for error class {error_value} of sensor {feature_matrix_error_value.columns[sensor_no]}\n", feature_matrix_error_value.iloc[:, sensor_no])