import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr

"""polynomial parameters"""
p_min = 1
p_max = 6
n = p_max - p_min
p = np.arange(p_min, p_max)

"""lambda paramter"""
lmd_min = 0.0
lmd_max = 1
n_lmd = 7
lmd = np.linspace(lmd_min, lmd_max, n_lmd)

"""placeholders"""
mse = np.zeros((n, n_lmd))
r2_ridge = np.zeros((n, n_lmd))
r2_lasso = np.zeros((n, n_lmd))
coefficients = np.zeros(n)

"""simulation"""
FF_terrain = lr.Data(0)
FF_terrain.GenerateDataFF(30, 30, 0.1)
FF_terrain.CreateDesignMatrix(5)
FF_terrain.Skl_Lasso(0.1)
