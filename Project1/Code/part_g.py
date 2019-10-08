import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr

p_min = 1
p_max = 14
n = p_max - p_min
p = np.arange(p_min, p_max)

r2_train=np.zeros(n)
r2_test=np.zeros(n)

plt.style.use("seaborn-whitegrid")
dataset=1
Terrain2 = lr.Data(dataset)
Terrain2.GenerateDataTerrain()

for j in range(0,n):
    Terrain2.CreateDesignMatrix(int(p[j]),split="False")
    Terrain2.Kfold_Crossvalidation(0,k=10,method=0)
    if Terrain2.r2_kf[0]>0:
        r2_test[j] = Terrain2.r2_kf[0]
    if Terrain2.r2_kf_train[0]>0:
        r2_train[j] = Terrain2.r2_kf_train[0]



plt.style.use("seaborn-whitegrid")
plt.plot(p, r2_test, "k--", p, r2_train, "k:")
plt.legend(("$R^2$ Test", "$R^2$ Training"))
plt.xlabel("Model complexity --->")
plt.ylabel("$R^2$")
plt.savefig("../Results/Part_g/OLSr2_terraintest%s.pdf"%dataset, bbox_inches="tight")
plt.show()

p=2
lmd_min=0.0
lmd_max=1
n_lmd=20
lmd=np.linspace(lmd_min,lmd_max,n_lmd)
r2_train=np.zeros(20)
bestlambda=0
bestr2=0
Terrain2.CreateDesignMatrix(int(p),split="False")
for i in range(0,n_lmd):
        Terrain2.Kfold_Crossvalidation(lmd[i],k=10,method=1)
        if Terrain2.r2_kf[0]>0:
            r2_train[i]=Terrain2.r2_kf[0]

for i in range(0,n_lmd):
    if r2_train[i]>bestr2:
        bestr2=r2_train[i]
        bestlambda=lmd[i]

print(bestlambda)
print(bestr2)
