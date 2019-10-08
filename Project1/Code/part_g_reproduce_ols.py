import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr

p = 2


plt.style.use("seaborn-whitegrid")

Terrain = lr.Data(1)
Terrain.GenerateDataTerrain()
Terrain.CreateDesignMatrix(p, split="False")
Terrain.OLS_SVD()
Terrain.PlotPredicted("OLS")

Terrain2 = lr.Data(2)
Terrain2.GenerateDataTerrain()
Terrain2.CreateDesignMatrix(p, split="False")
Terrain2.Ridge(0.26)
Terrain2.PlotPredicted("Ridge")
n, m = np.shape(Terrain2.template)
deviationaver = 0
for i in range(0, 7):
    for j in range(0, 3):
        square_real = Terrain2.template[
            500 * i : 500 * (i + 1), 500 * j : 500 * (j + 1)
        ]
        square_pred = Terrain2.predicted_terrain[
            500 * i : 500 * (i + 1), 500 * j : 500 * (j + 1)
        ]
        deviationaver = deviationaver + abs(
            np.mean(np.ravel(square_real)) - np.mean(np.ravel(square_pred))
        )
deviationaver = deviationaver / (7 * 3)
print("Average deviation from 500x500 square mean height", deviationaver)
print("Average height of terrain:", np.mean(np.ravel(Terrain2.template)))
