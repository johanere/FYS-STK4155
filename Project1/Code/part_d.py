import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import LRclasses0 as lr

"""polynomial parameters"""
p_min = 11
p_max = 16
n = p_max - p_min
p = np.arange(p_min, p_max)

"""lambda paramter"""
lmd_min=0.1
lmd_max=1
n_lmd=20
lmd=np.linspace(lmd_min,lmd_max,n_lmd)

"""placeholders"""
mse = np.zeros((n,n_lmd))
r2_ridge =  np.zeros((n,n_lmd))
r2_lasso =  np.zeros((n,n_lmd))
coefficients = np.zeros(n)

"""simulation"""
FF_terrain = lr.Data(0)
FF_terrain.GenerateDataFF(30, 30, 0.2)
for i in range(n): #loop over polynomials
    print(p[i])
    FF_terrain.CreateDesignMatrix(p[i], split="False")
    for j in range(0,n_lmd): #loop over lambdas
        FF_terrain.Kfold_Crossvalidation(lmd[j],10)
        mse[i,j]=FF_terrain.mse_kf[0]
        r2_ridge[i,j]=  FF_terrain.mse_kf[0]
        r2_lasso[i,j] = FF_terrain.mse_kf[1]


"""produce table for ridge MSE"""
table = []
headers = ["Polynomial degree"]
for value in lmd:
    headers.append("$ MSE(\\lambda=%.2f)$"%value)
for i in range(0, n):
    line = ["$ %s $" % (i + 1)]
    for j in range(0,n_lmd):
        line.append("$ %.4f $" % mse[i,j])
    table.append(line)

with open("../Results/part_d_mse.txt", "w") as outputfile:
    outputfile.write(tabulate(table, headers, tablefmt="latex_raw"))

"""produce subplots of r2 score ridge vs lasso"""
plt.style.use("seaborn-whitegrid")
fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True)
for i in range(0, n):
    x = lmd
    y1 = r2_ridge[i,:]
    y2 =r2_lasso[i,:]
    ax[i].plot(x, y1, label="Ridge")
    ax[i].plot(x, y2, "k--",label="Lasso")
    ax[i].set_ylabel("%s. order\n $ R^2$ " % (i + 1))
plt.savefig("..\\Results\\r2ridgelasso12.pdf", bbox_inches="tight")
