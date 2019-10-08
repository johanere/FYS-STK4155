import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import LRclasses0 as lr

p_min = 1
p_max = 12
n = p_max - p_min
p = np.arange(p_min, p_max)
noise = [0, 0.3, 1]
noise_label = "1"  # label to avoid . in file name

coefficients = np.zeros(6)
mse = np.zeros((3, n))

FF_terrain = lr.Data(0)
for j in range(0, 3):
    FF_terrain.GenerateDataFF(30, 30, noise[j])
    for i in range(n):
        FF_terrain.CreateDesignMatrix(p[i], split="False")
        FF_terrain.Kfold_Crossvalidation(0, 20, 0)
        mse[j, i] = FF_terrain.mse_kf[0]
    print(len(mse[j, :]))
plt.style.use("seaborn-whitegrid")
plt.plot(p, mse[0, :], "k")
plt.plot(p, mse[1, :], "k--")
plt.plot(p, mse[2, :], "k:")
plt.legend(("No Noise", "Noise $\\sim N(0,0.3)$", "Noise $\\sim N(0,1)$"))
plt.xlabel("Model complexity --->")
plt.ylabel("MSE")
plt.show()
