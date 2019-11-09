import numpy as np
import matplotlib.pyplot as plt


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
from metrics import gains_plot_area, Confusion_and_accuracy

def GD(X,y,max_iters=1000,eta=0.1,plot=False):
    beta=np.ones(np.size(X,1)) #initalize beta vector
    costfunc=np.zeros(max_iters)
    N=float(np.size(y))
    for iter in range(max_iters):
        prediction =1/(1+np.exp(-X@beta))
        gradient= -1.0/N*X.T@(y-prediction)
        beta -= eta*gradient
        costfunc[iter] = -np.sum(y*(X@beta)-np.log(1+np.exp(X@beta)))
    if plot==True:
        plt.style.use("seaborn-whitegrid")
        plt.plot(np.arange(0,max_iters,1)[200:],costfunc[200:],"k")
        plt.xlabel("epochs")
        plt.ylabel("cost function")
        plt.show()
    return beta

def learning_schedule(t):
    t0 = 1
    t1 = 7
    return 0.1#t0/(t+t1)


def SGD(X,y,n_epochs=1000,batch_size=20,eta=0.1,lmd=0.0,plot=False):
    n = np.size(y) #number of data point
    m=int(n/batch_size) #number of batches
    random_index = np.random.randint(m)
    beta=np.ones(np.size(X,1)) #initalize beta vector
    data_indices = np.arange(n)
    if plot==True:
        costfunc=np.zeros(n_epochs)
    for epoch in range(n_epochs):
        for i in range(m):
            chosen_datapoints = np.random.choice(
                data_indices, size=batch_size, replace=False)
            Xi = X[chosen_datapoints]
            yi = y[chosen_datapoints]
            prediction =1/(1+np.exp(-Xi@beta))
            gradient= -1.0/float(batch_size)*Xi.T@(yi-prediction)
            if lmd > 0.0:
                gradient += lmd * gradient
            eta = learning_schedule(epoch*m+i)
            beta = beta - eta*gradient
        if plot==True:
            costfunc[epoch] = -np.sum(y*(X@beta)-np.log(1+np.exp(X_train@beta)))
    if plot==True:
        plt.style.use("seaborn-whitegrid")
        plt.plot(np.arange(0,n_epochs,1)[200:],costfunc[200:],"k")
        plt.xlabel("epochs")
        plt.ylabel("cost function")
        plt.show()
    return beta


def scale_data(X_train,X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test

#set parameters for fit
max_iters=5000
eta_gd=0.1
n_epochs=1000
batch_size=20
eta_sgd=0.1
lmd=0.0
#-----Load data-----
cancer = load_breast_cancer()
X = cancer.data
y=  cancer.target
#X, y = make_moons(100, noise=0.2, random_state=7)
#-----split with stratification-----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=1,stratify=y)
#-----pre process data------
X_train,X_test=scale_data(X_train,X_test) #scale with standard scaler
#SCALE Y AS WELL?
#scale 0-1?
#sm = SMOTE(random_state=42)
#X_train, y_train = sm.fit_resample(X, y)
#-----Minimize cost functions-----
beta_GD= GD(X_train,y_train,max_iters,eta_gd,plot=True)
beta_SGD=SGD(X_train,y_train,n_epochs,batch_size,eta_sgd,lmd,plot=False)
#----- make SKL fit for comparison-----
clf = linear_model.LogisticRegressionCV(cv=5,max_iter=5000,solver="lbfgs",n_jobs=-1) #use all available cores for CV
clf.fit(X_train,y_train)

#---predict
X_true=X_test #set predictors and data to use as true X
y_true=y_test #set targets to use as true y
y_predicted_GD = 1/(1+np.exp(-X_true@beta_GD))
y_predicted_SGD = 1/(1+np.exp(-X_true@beta_SGD))
y_predicted_SKL=clf.predict(X_test)
#-----evaluate fits-----
Confusion_GD, accuracy_GD=Confusion_and_accuracy(y_true,y_predicted_GD)
area_score_GD=gains_plot_area(y_true,y_predicted_GD,"gd")
"""
Confusion_SGD, accuracy_SGD=Confusion_and_accuracy(y_true,y_predicted_SGD)
area_score_SGD=gains_plot_area(y_true,y_predicted_SGD)

Confusion_SGD, accuracy_SGD=Confusion_and_accuracy(y_true,y_predicted_SKL)
area_score_SGD=gains_plot_area(y_true,y_predicted_SKL)
"""
print("Fit GD")
print("Fit using SGD")
print("Fit GD")
