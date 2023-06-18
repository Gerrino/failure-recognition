from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, Float
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey

Base = declarative_base()


class SchemaInterface:
    __tablename__: str = None

    @classmethod
    def get_rename_dict(cls) -> dict:
        return dict()

    @classmethod
    def get_drop_list(cls) -> list:
        return []


class timeseries_me(Base):
    __tablename__ = 'timeseries_me'
    timeseries_Number = Column(Integer, primary_key=True)
    TimeSeries_ME_id = Column(Integer)
    timeseries_me_count = Column(Integer)
    Temp01 = Column(Float, name="01_Temp01")
    Temp02 = Column(Float, name="02_Temp02")
    Temp03 = Column(Float, name="03_Temp03")
    Temp04 = Column(Float, name="04_Temp04")
    Temp05 = Column(Float, name="05_Temp05")
    Temp06 = Column(Float, name="06_Temp06")
    dP01 = Column(Float, name="07_dP01")
    dP02 = Column(Float, name="08_dP02")
    dP03 = Column(Float, name="09_dP03")
    dP04 = Column(Float, name="10_dP04")
    Temp07 = Column(Float, name="11_Temp07")
    Rf01 = Column(Float, name="12_Rf01")
    V01 = Column(Float, name="13_V01")
    Temp08 = Column(Float, name="14_Temp08")
    Rf02 = Column(Float, name="15_Rf02")
    V02 = Column(Float, name="16_V02")
    Temp09 = Column(Float, name="17_Temp09")
    Temp10 = Column(Float, name="18_Temp10")
    Temp11 = Column(Float, name="19_Temp11")
    Lubi01 = Column(Float, name="20_Lubi01")
    Q01 = Column(Float, name="21_Q01")
    Q02 = Column(Float, name="22_Q02")
    Blk1 = Column(Float, name="23_Blk1")
    Blk2 = Column(Float, name="24_Blk2")
    Blk3 = Column(Float, name="25_Blk3")
    Blk4 = Column(Float, name="26_Blk4")
    M = Column(Float, name="27_M")
    Masse_Ofen = Column(Float, name="28_Masse_Ofen")
    Temp12 = Column(Float, name="29_Temp12")
    Temp13 = Column(Float, name="30_Temp13")
    Q03 = Column(Float, name="31_Q03")
    Kolbenschmierzeit = Column(Float, name="Kolbenschmierzeit")
    Spruehzeit = Column(Float, name="Sprühzeit")
    #xMischungsverhaeltnis = Column(Float, name="Mischungsverhältnis")
    #xspez_Q_Spruehmittelkonzentrat = Column(Float, name="spez._Q_Sprühmittelkonzentrat")
    Temperiermittelmenge_FF = Column(Float, name="Temperiermittelmenge_FF")
    Temperiermittelmenge_BF = Column(Float, name="Temperiermittelmenge_BF")
    P01_Heizung_FF = Column(Float, name="P01_Heizung_FF")
    P02_Heizung_BF = Column(Float, name="P02_Heizung_BF")
    #xP03_Kuehlung_Giesskolben = Column(Float, name="P03_Kühlung_Giesskolben")
    P04_Wasser_Kuehlung_total = Column(Float, name="P04_Wasser_Kühlung_total")
    #xP05_Spruehen_Form = Column(Float, name="P05_Sprühen_Form")
    Diff_Temp10_Temp11 = Column(Float, name="Differenz_Temp10/Temp11")
    Steigung_Temp09 = Column(Float, name="Steigung_Temp09")
    Steigung_Temp10 = Column(Float, name="Steigung_Temp10")
    Steigung_Temp11 = Column(Float, name="Steigung_Temp11")

    @classmethod
    def get_rename_dict(cls) -> dict:
        ts_rename_map = {
            "01_Temp01": "Temp01",
            "02_Temp02": "Temp02",
            "03_Temp03": "Temp03",
            "04_Temp04": "Temp04",
            "05_Temp05": "Temp05",
            "06_Temp06": "Temp06",
            "07_dP01": "dp01_07",
            "08_dP02": "dp02_08",
            "09_dP03": "dp03_09",
            "10_dP04": "dp04_10",
            "11_Temp07": "Temp07_11",
            "12_Rf01": "Rf01_12",
            "13_V01": "V01_13",
            "14_Temp08": "Temp08_14",
            "15_Rf02": "Rf02_15",
            "16_V02": "V02_16",
            "17_Temp09": "Temp09_17",
            "18_Temp10": "Temp10_18",
            "19_Temp11": "Temp11_19",
            "20_Lubi01": "Lubi01_20",
            "21_Q01": "Q01_21",
            "22_Q02": "Q02_22",
            "23_Blk1": "Blk1_23",
            "24_Blk2": "Blk2_24",
            "25_Blk3": "Blk3_25",
            "26_Blk4": "Blk4_26",
            "27_M": "M_27",
            "28_Masse_Ofen": "Masse_Ofen_28",
            "29_Temp12": "Temp12_29",
            "30_Temp13": "Temp13_30",
            "31_Q03": "Q03_31",
            "spez._Q_Sprühmittelkonzentrat": "spez_Sprühmittelkonzentrat",
            "Differenz_Temp10/Temp11": "Differenz_Temp10Temp11"
        }
        ts_rename_map.update({
            cls.timeseries_me_count.name: "time",
            cls.TimeSeries_ME_id.name: "id"
        })
        return ts_rename_map

    @classmethod
    def get_drop_list(cls) -> list:
        return [cls.timeseries_Number.name]


class timeseries_zdg(Base):
    __tablename__ = 'timeseries_zdg'
    idtimeseries_zdg_id = Column(Integer, primary_key=True)
    timeseries_col = Column(Integer)
    timeseries_id = Column(Integer)
    p_Cav1 = Column(Float)
    p_Cav3 = Column(Float)
    p_Cav4 = Column(Float)
    p_cav5 = Column(Float)
    p_IM = Column(Float)
    p_s2 = Column(Float)
    sl = Column(Float)
    vl = Column(Float)

    @classmethod
    def get_rename_dict(cls) -> dict:
        return {
            cls.timeseries_id.name: "id",
            cls.timeseries_col.name: "time",
        }

    @classmethod
    def get_drop_list(cls) -> list:
        return [cls.idtimeseries_zdg_id.name]
