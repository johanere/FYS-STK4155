import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import LRclasses0 as lr

p_min = 1
p_max = 2
n = p_max - p_min
p = np.arange(p_min, p_max)

MSE_train = np.zeros(n)
MSE_test = np.zeros(n)

beta = np.zeros((n, 21))
betaints = np.zeros((n, 21))
coefficients = np.zeros(6)
mse = np.zeros(n)  # 0 = mse, 1=r2
r2 = np.zeros(n)

FF_terrain = lr.Data(0)
FF_terrain.GenerateDataFF(30, 30, 0.1)

for i in range(n):
    FF_terrain.CreateDesignMatrix(p[i], split="False")
    FF_terrain.OLS_SVD()
    FF_terrain.Train_Predict()
    FF_terrain.BetaConfidenceIntervals()
    for j in range(0, len(FF_terrain.beta)):
        beta[i, j] = FF_terrain.beta[j]
        betaints[i, j] = FF_terrain.beta_ci[j]
    coefficients[i] = np.size(FF_terrain.beta)
    r2[i] = FF_terrain.r2_train
    mse[i] = FF_terrain.MSE_train


table = []
headers = ["Polynomial degree", "$R^2$", "MSE"]
for j in range(0, n):
    line = ["$ %s $" % (j + 1)]
    line.append("$ %.2f $" % r2[j])
    line.append("$ %.2f $" % mse[j])
    table.append(line)

with open("../Results/part_a_r2_mse.txt", "w") as outputfile:
    outputfile.write(tabulate(table, headers, tablefmt="latex_raw"))
