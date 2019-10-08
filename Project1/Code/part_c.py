import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr

p_min = 1
p_max = 12
n = p_max - p_min

MSE_train = np.zeros(n)
MSE_test = np.zeros(n)
var = np.zeros(n)
bias = np.zeros(n)

p = np.arange(p_min, p_max)
print(p)
runs = 80
for run in range(0, runs):
    FF_terrain = lr.Data(0)
    FF_terrain.GenerateDataFF(30, 30, 0.1)
    for i in range(n):
        FF_terrain.CreateDesignMatrix(p[i], "True")
        FF_terrain.OLS_SVD()
        FF_terrain.Train_Predict()
        FF_terrain.Test_Predict()
        MSE_train[i] = MSE_train[i] + FF_terrain.MSE_train
        MSE_test[i] = MSE_test[i] + FF_terrain.MSE_test
        var[i] = var[i] + FF_terrain.var
        bias[i] = bias[i] + FF_terrain.bias
MSE_train = MSE_train / float(runs)
MSE_test = MSE_test / float(runs)

plt.style.use("seaborn-whitegrid")
plt.plot(p, MSE_train, "k--", p, MSE_test, "k:")
plt.legend(("MSE Training Sample", "MSE Test Sample"))
plt.xlabel("Model complexity --->")
plt.ylabel("MSE")
plt.savefig("..\\Results\\bias_var_MSE.pdf", bbox_inches="tight")
plt.figure()
plt.plot(p, var, "k--")
plt.plot(p, bias, "k:")
plt.legend(("Variance", "Bias"))
plt.xlabel("Model complexity [p] --->")
plt.ylabel("MSE")
plt.savefig("..\\Results\\bias_var.pdf", bbox_inches="tight")
