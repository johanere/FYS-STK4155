import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr

p_min = 1
p_max = 10
n = p_max - p_min
p = np.arange(p_min, p_max)
noise = 0.3

mse = np.zeros((2, n))

p = np.arange(p_min, p_max)
FF_terrain = lr.Data(0)
for run in range(0, 50):
    FF_terrain.GenerateDataFF(30, 30, noise)
    for i in range(n):
        FF_terrain.CreateDesignMatrix(p[i], split="False")
        FF_terrain.Kfold_Crossvalidation(0, k=10, method=0)
        mse[0, i] = mse[0, i] + FF_terrain.mse_kf[0]
        mse[1, i] = mse[1, i] + FF_terrain.mse_kf_train[0]

mse[0, :] = mse[0, :] / 50.0
mse[1, :] = mse[1, :] / 50.0
plt.style.use("seaborn-whitegrid")
plt.plot(p, mse[1, :], "k--", p, mse[0, :], "k:")
plt.legend(("MSE Training Sample", "MSE Test Sample"))
plt.xlabel("Model complexity --->")
plt.ylabel("MSE")
plt.savefig("../Results/Part_c/bias_var_50runs.pdf", bbox_inches="tight")
plt.show()
