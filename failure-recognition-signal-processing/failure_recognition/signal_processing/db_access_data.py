import sqlalchemy as db
from sqlalchemy.orm import sessionmaker
import yaml
from failure_recognition.signal_processing.db_schema import timeseries_me
# mysql://gerri:G3ndo$$$@localhost/modeldb
import pandas as pd

def load_db_data():
    with open("examples/db_info.yaml", "r") as user_data_stream:
        db_info = yaml.safe_load(user_data_stream)
    engine = db.create_engine(f'mysql://{db_info["username"]}:{db_info["userpassword"]}@{db_info["host"]}/{db_info["schema"]}')
    connection = engine.connect()
    metadata = db.MetaData()
    metadata.reflect(engine)
    Session = sessionmaker(bind = engine)
    session = Session()
    #series: timeseries_me = session.query(timeseries_me).filter(timeseries_me.timeSeries_ME_id < 10).all()


    series_data_frame = pd.read_sql_table(timeseries_me.__tablename__, connection, db_info["schema"], chunksize=500)

    pass


if __name__ == "__main__":
    load_db_data()