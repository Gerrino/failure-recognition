import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table
import yaml
from failure_recognition.signal_processing.db_schema import timeseries_me, SchemaInterface, timeseries_zdg
# mysql://gerri:G3ndo$$$@localhost/modeldb
import pandas as pd

def load_db_data(table: SchemaInterface, save_data: bool) -> pd.DataFrame:
    """Load a pandas dataframe from the database using the data base info provided in db_info.yaml
    """
 
    with open("examples/db_info.yaml", "r") as user_data_stream:
        db_info = yaml.safe_load(user_data_stream)
    engine = db.create_engine(f'mysql://{db_info["username"]}:{db_info["userpassword"]}@{db_info["host"]}/{db_info["schema"]}')
    connection = engine.connect()
    metadata = db.MetaData()
    metadata.reflect(engine)

    table_name = table.__tablename__
    series_data_frame = pd.read_sql_table(table_name, connection, db_info["schema"])
    
    series_data_frame.drop(columns=table.get_drop_list(), inplace=True)
    series_data_frame.rename(columns=table.get_rename_dict(), inplace=True)
    
    if save_data:
        series_data_frame.to_csv(table_name + ".csv")
        series_data_frame.to_pickle(table_name + ".pkl")

    return series_data_frame


if __name__ == "__main__":
    load_db_data(timeseries_zdg, save_data=True)