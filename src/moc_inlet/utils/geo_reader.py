import numpy as np
import pandas as pd

def geo_reader(filepath):
    walls = pd.read_csv(filepath)
    x1 = np.array(walls["x1"].values)
    y1 = np.array(walls["y1"].values)
    x2 = np.array(walls["x2"].values)
    y2 = np.array(walls["y2"].values)

    x1 = cleaner(x1)
    x2 = cleaner(x2)
    y1 = cleaner(y1)
    y2 = cleaner(y2)

    body1 = np.column_stack((x1, y1))
    body2 = np.column_stack((x2, y2))
    return body1, body2


def cleaner(array):
    return array[~np.isnan(array)]

