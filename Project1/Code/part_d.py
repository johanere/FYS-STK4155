import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr


p =[3,6,8,12]
noise=0.3

lmd_min=0.1
lmd_max=1
n_lmd=20
lmd=np.linspace(lmd_min,lmd_max,n_lmd)

mse_ridge=np.zeros((2,n_lmd))
mse_lasso=np.zeros((2,n_lmd))

plt.style.use("seaborn-whitegrid")
fig, ax = plt.subplots(nrows=1, ncols=4)

FF_terrain = lr.Data(0)
FF_terrain.GenerateDataFF(30, 30, noise)
for j in range(0,4):
    FF_terrain.CreateDesignMatrix(int(p[j]),split="False")
    for i in range(0,n_lmd):
        FF_terrain.Kfold_Crossvalidation(lmd[i],k=10,method=1)
        mse_ridge[0,i] = mse_ridge[0,i]+FF_terrain.mse_kf[0]
        mse_ridge[1,i] = mse_ridge[1,i]+FF_terrain.mse_kf_train[0]
        mse_lasso[0,i] = mse_lasso[0,i]+FF_terrain.mse_kf[1]
        mse_lasso[1,i] = mse_lasso[1,i]+FF_terrain.mse_kf_train[1]

    mse_ridge=mse_ridge-np.mean(mse_ridge)
    mse_lasso=mse_lasso-np.mean(mse_lasso)
    ax[j].plot(lmd, mse_ridge[1,:], lmd, mse_ridge[0,:] )
    ax[j].plot(lmd, mse_lasso[1,:], lmd, mse_lasso[0,:] )
    ax[j].legend(("Ridge Training MSE ", "Ridge Test MSE ","Lasso Training MSE ", "Lasso Test MSE "))
    ax[j].set_xlabel("$ \\lambda $")
    ax[j].set_ylabel("MSE")
    ax[j].title.set_text('complexity: %sth order'%p[j])
#plt.savefig("../Results/Part_d/ridgelasso_lambdaest_p%s.pdf"%p, bbox_inches="tight")
plt.show()
