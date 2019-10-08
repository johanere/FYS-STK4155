import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr


p = [3, 6, 8, 12]
noise = 0.3

lmd_min = 0.1
lmd_max = 1
n_lmd = 20
lmd = np.linspace(lmd_min, lmd_max, n_lmd)

mse_ridge = np.zeros((2, n_lmd))
mse_lasso = np.zeros((2, n_lmd))

plt.style.use("seaborn-whitegrid")
fig, ax = plt.subplots(nrows=1, ncols=4)

FF_terrain = lr.Data()
FF_terrain.GenerateDataTerrain()
for j in range(0, 4):
    FF_terrain.CreateDesignMatrix(int(p[j]), split="False")
    for i in range(0, n_lmd):
        FF_terrain.Kfold_Crossvalidation(lmd[j], k=10, method=1)
        mse_ridge[0, j] = FF_terrain.r2_kf[0]
        mse_ridge[1, j] = FF_terrain.r2_kf_train[0]
        mse_lasso[0, j] = FF_terrain.r2_kf[1]
        mse_lasso[1, j] = FF_terrain.r2_kf_train[1]

    ax[j].plot(lmd, mse_ridge[1, :], lmd, mse_ridge[0, :])
    ax[j].plot(lmd, mse_lasso[1, :], lmd, mse_lasso[0, :])
    ax[j].legend(("Ridge Training ", "Ridge Test", "Lasso training", "Lasso Test"))
    ax[j].set_xlabel("$ \\lambda $")
    ax[j].set_ylabel("MSE")
    ax[j].title.set_text("complexity of order %s" % p[j])
plt.show()
