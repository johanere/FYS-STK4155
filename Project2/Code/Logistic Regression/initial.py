import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import confusion_matrix




cancer = load_breast_cancer()
X = cancer.data
y=  cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=1)

"""moon test"""
"""
from sklearn.datasets import make_moons
X, y = make_moons(20, noise=0.2, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
"""


max_iters=3000
eta=0.01
beta=np.ones(np.size(X_train,1)) #initalize beta vector
costfunc=np.zeros(max_iters)
for iter in range(max_iters):
    prediction =1/(1+np.exp(-X_train@beta))
    gradient= -X_train.T@(y_train-prediction)
    beta -= eta*gradient
    costfunc[iter] = -np.sum(y_train*(X_train@beta)-np.log(1+np.exp(X_train@beta)))

#plt.plot(np.arange(0,max_iters,1),costfunc)
#plt.show()


compare_X=X_train
compare_y=y_train
def MSE(z1, z2):
    MSE = mean_squared_error(z1, z2, sample_weight=None, multioutput="uniform_average")
    return MSE

def r2(z1, z2):
    r2 = r2_score(z1, z2, sample_weight=None, multioutput="uniform_average")
    return r2

""" compare predictions with y""" """
for i in range(len(y_train)):
    print(1/(1+np.exp(-compare_X@beta))[i],compare_y[i])
"""
def Accuracy(z1, z2,treshold=0.5):
    prediction_to_binary=np.where(z1>treshold,1,0)
    Confusion=confusion_matrix(compare_y, prediction_to_binary, labels=None, sample_weight=None)
    accuracy = (Confusion[0,0]+Confusion[1,1])/(np.sum(Confusion))
    return accuracy

#ADD gains chart


prediction = 1/(1+np.exp(-X_train@beta)) #np.exp(X_train@beta)/(1+np.exp(X_train@beta))

#clf = linear_model.LogisticRegressionCV(cv=5)
#clf.fit(X_train,y_train)
#SKfit=clf.predict(compare_X)
#print("own MSE:" ,MSE(prediction,compare_y), "SKL MSE:",MSE(SKfit,compare_y))
#print("own r2:" ,r2(prediction,compare_y), "SKL r2:",r2(SKfit,compare_y))

print(Accuracy(prediction,compare_y))
#print(Accuracy(SKfit,compare_y))




"""
    gradients =  -np.transpose(XTrain)@(prediction-yTrain))

def GD(X_train,y_train):
    beta=np.ones(np.shape(y))
    eta = 0.1
    Niterations = 1000
    for iter in range(Niterations):
        gradients = 2.0/m*xb.T.dot(xb.dot(beta)-y)
        beta -= eta*gradients

Logistic_Regression_fit(X_train,y_train)
"""
