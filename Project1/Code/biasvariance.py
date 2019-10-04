import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr

p_min=1
p_max=10
n=p_max-p_min

MSE_train=np.zeros(n)
MSE_test=np.zeros(n)
p=np.arange(p_min,p_max)

for run in range(0,100):
    FF_terrain=lr.Data(0)
    FF_terrain.GenerateDataFF()
    for i in range(n):
        FF_terrain.CreateDesignMatrix(p[i])
        FF_terrain.OLS()
        FF_terrain.Train_Predict()
        FF_terrain.Test_Predict()
        MSE_train[i]=MSE_train[i]+FF_terrain.MSE_train
        MSE_test[i]=MSE_test[i]+FF_terrain.MSE_test
    print(run)
MSE_train=MSE_train/300.0
MSE_test=MSE_test/300

print(MSE_test)

plt.plot(p,MSE_train,'k--',p,MSE_test,'k:')
plt.legend(('Training Sample',"Test Sample"))
plt.xlabel('Model complexity --->')
plt.ylabel('MSE')
plt.show()
