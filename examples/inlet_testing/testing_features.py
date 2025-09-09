import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from moc_inlet.moc_inlet import moc_inlet


solution = moc_inlet("examples/inlet_testing/sample_inlet3.csv", T=70, P=8729, M=4.03, theta=0.00, fluid="air",wave_res=5)











