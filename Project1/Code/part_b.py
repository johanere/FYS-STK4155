import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import LRclasses0 as lr

p_min = 4
p_max = 10
n = p_max - p_min
p = np.arange(p_min, p_max)
folds = [5, 10, 20, 40]
noise = 0.3

coefficients = np.zeros(6)
mse = np.zeros((4, n))

FF_terrain = lr.Data(0)
FF_terrain.GenerateDataFF(30, 30, noise)
for j in range(0, 4):
    for i in range(n):
        FF_terrain.CreateDesignMatrix(p[i], split="False")
        FF_terrain.Kfold_Crossvalidation(0, folds[j], 0)
        mse[j, i] = FF_terrain.mse_kf[0]
plt.style.use("seaborn-whitegrid")
plt.plot(p, mse[0, :], "k")
plt.plot(p, mse[1, :], "k--")
plt.plot(p, mse[2, :], "k:")
plt.plot(p, mse[3, :], "")
labels = []
for i in range(0, 4):
    labels.append("%s-Fold" % folds[i])
plt.legend((labels))
plt.xlabel("Model complexity --->")
plt.ylabel("MSE")
plt.savefig("../Results/Part_b/kfold_foldeval.pdf", bbox_inches="tight")
plt.show()
