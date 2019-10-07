import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import LRclasses0 as lr

p = 7

FF_terrain = lr.Data()
FF_terrain.GenerateDataTerrain()
FF_terrain.CreateDesignMatrix(p, split="True")
FF_terrain.Skl_OLS()
FF_terrain.Ridge(0)

lmb = np.linspace(0, 1, 10)
mse = np.zeros(10)
for i in range(0, 10):
    FF_terrain.Kfold_Crossvalidation(1, lmb[i], 10)
    mse[i] = FF_terrain.r2_kf
print(mse)
