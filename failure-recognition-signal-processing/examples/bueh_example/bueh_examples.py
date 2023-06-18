import pandas as pd

if __name__ == "__main__":
    classification_dataframe = pd.read_csv("./examples/dumps/buehler_data_classification_20230601.tsv", sep="\t")
    timeseries_id_to_classification_index = {v: k for k, v in enumerate(list(classification_dataframe.Zyklus_Nummer))}
    classification_index_to_timeseries_timeseries_id = {v: k for k, v in timeseries_id_to_classification_index.items()}

    test_id = 6 
    test_classification = classification_dataframe.iloc[timeseries_id_to_classification_index[test_id]]