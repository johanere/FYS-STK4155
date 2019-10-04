import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr

n=3

p=[0,1,2]

FF_terrain=lr.Data(0)
FF_terrain.GenerateDataFF(n=100, m=100, noise=0)
for i in range(n):
    FF_terrain.CreateDesignMatrix(p[i],"False")
    FF_terrain.OLS_SVD()
    FF_terrain.Train_Predict()
    FF_terrain.BetaConfidenceIntervals()
    print(FF_terrain.beta_ci)
