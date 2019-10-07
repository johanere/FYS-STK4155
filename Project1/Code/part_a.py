import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import LRclasses0 as lr

p_min = 1
p_max = 6
n = p_max - p_min
p = np.arange(p_min, p_max)
noise=1
noise_label="1" #label to avoid . in file name
beta = np.zeros((n, 21))
betaints = np.zeros((n, 21))
coefficients = np.zeros(6)
mse = np.zeros(n)
r2 = np.zeros(n)

FF_terrain = lr.Data(0)
FF_terrain.GenerateDataFF(30, 30, noise)
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
    line.append("$ %.4f $" % r2[j])
    line.append("$ %.4f $" % mse[j])
    table.append(line)

with open("../Results/Part_a/r2_mse_noise%s.txt"%noise_label, "w") as outputfile:
    outputfile.write(tabulate(table, headers, tablefmt="latex_raw"))


plt.style.use("seaborn-whitegrid")
fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True)
ticks = []
for i in range(0, 21):
    ticks.append("$\\beta_{%s}$" % i)
plt.xticks(range(21), ticks)
for i in range(0, 5):
    points = int(coefficients[i])
    x = np.arange(0, points, 1)
    y = np.zeros(points)
    error = np.zeros(points)
    for j in range(0, points):
        error[j] = betaints[i, j]  # /beta[i,j]
        y[j] = beta[i, j]
    ax[i].errorbar(x, y, yerr=error, fmt=".k")
    ax[i].set_xlim([0, 21])
    ax[i].set_ylabel("%s. order\n $ CI_{0.95}(\\beta_j) $ " % (i + 1))

plt.savefig("../Results/Part_a/betas_franke_all_noise%s.pdf"%noise_label, bbox_inches="tight")

points = int(coefficients[4])
x = np.arange(0, points, 1)
y = np.zeros(points)
error = np.zeros(points)
plt.figure()
for j in range(0, points):
    error[j] = betaints[i, j]  # /beta[i,j]
    y[j] = beta[i, j]
plt.errorbar(x, y, yerr=error, fmt=".k")
plt.xticks(range(21), ticks)
plt.ylabel("$ CI_{0.95}(\\beta_j) $ ")
plt.savefig("../Results/Part_a/Part_abetas_franke_p5_noise%s.pdf"%noise_label, bbox_inches="tight")
