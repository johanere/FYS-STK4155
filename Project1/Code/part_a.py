import numpy as np
import matplotlib.pyplot as plt


import LRclasses0 as lr

p_min=0
p_max=6
n=p_max-p_min

runs=1

MSE_train=np.zeros(n)
MSE_test=np.zeros(n)

p=np.arange(p_min,p_max)

beta=np.zeros((6,24))
betaints=np.zeros((5,24))

for run in range(0,runs):
    FF_terrain=lr.Data(0)
    FF_terrain.GenerateDataFF()
    for i in range(n):
        FF_terrain.CreateDesignMatrix(p[i],"False")
        FF_terrain.OLS()
        FF_terrain.Train_Predict()
        FF_terrain.BetaConfidenceIntervals()
        if run==runs-1:
            for j in range(0,len(FF_terrain.beta)):
                beta[i,j]=FF_terrain.beta[j]
                betaints[i,j]=FF_terrain.beta_ci[j]

print(beta[0,0])


from tabulate import tabulate

table=[]
headers = ["Polynomial degree"]
for j in range(0,6):
    line=["$ %s $"%j]
    for k in range(0,24):
        if betaints[j,k] != 0:
            line.append("$ %.2f \pm %.2f $"%(beta[i,k],betaints[i,k]))
        else:
            line.append(" - ")
    table.append(line)
for k in range(0,24):
    headers.append("$\\beta_{%s} $"%k)

with open('../DataFiles/part_a.txt', 'w') as outputfile:
    outputfile.write(tabulate(table, headers, tablefmt="latex_raw"))
