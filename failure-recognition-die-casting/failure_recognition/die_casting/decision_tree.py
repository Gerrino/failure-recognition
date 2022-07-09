from matplotlib import pyplot as plt
import pandas as pd
from failure_recognition.die_casting import PATH_DICT


 
plt.close("all")
timeseries = pd.read_csv(
    PATH_DICT["buehler"], decimal=".", sep='\t', header=0)

pass
 
