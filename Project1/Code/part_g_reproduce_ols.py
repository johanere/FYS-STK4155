import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr
p=2


plt.style.use("seaborn-whitegrid")

Terrain = lr.Data(2)
Terrain.GenerateDataTerrain()
Terrain.CreateDesignMatrix(p,split="False")
Terrain.OLS_SVD()
Terrain.PlotPredicted("OLS")

Terrain2 = lr.Data(2)
Terrain2.GenerateDataTerrain()
Terrain2.CreateDesignMatrix(p,split="False")
Terrain2.Ridge(0.26)
Terrain2.PlotPredicted("Ridge")
