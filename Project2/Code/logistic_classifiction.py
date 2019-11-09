import numpy as np
import matplotlib.pyplot as plt


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons

from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE

from metrics_and_preprocessing import gains_plot_area, Confusion_and_accuracy, scale_data_standard


#-----Define functions
def GD(X,y,max_iters=1000,eta=0.1,plot=False): #Gradient decent
    beta=np.ones(np.size(X,1)) #initalize beta vector
    costfunc=np.zeros(max_iters)
    N=float(np.size(y))
    for iter in range(max_iters):
        prediction =1/(1+np.exp(-X@beta))
        gradient= -1.0/N*X.T@(y-prediction)
        beta -= eta*gradient
        costfunc[iter] = -np.sum(y*(X@beta)-np.log(1+np.exp(X@beta)))
    if plot==True:
        print("Cost function SGD at last epoch:", costfunc[-1])
        plt.style.use("seaborn-whitegrid")
        plt.plot(np.arange(0,max_iters,1)[200:],costfunc[200:],"k")
        plt.xlabel("epochs")
        plt.ylabel("cost function")
        plt.show()
    return beta

def learning_schedule(t): #Learning schedule for SGD
    t0 = 1
    t1 = 10
    return t0/(t+t1)


def SGD(X,y,n_epochs=1000,batch_size=20,eta=0.1,lmd=0.0,plot=False): #stochastic GD
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
            eta = eta#learning_schedule(epoch*m+i)
            beta = beta - eta*gradient
        if plot==True:
            costfunc[epoch] = -np.sum(y*(X@beta)-np.log(1+np.exp(X_train@beta)))
    if plot==True:
        plt.style.use("seaborn-whitegrid")
        plt.plot(np.arange(0,n_epochs,1)[200:],costfunc[200:],"k")
        plt.xlabel("epochs")
        plt.ylabel("cost function")
        plt.show()
        print("Cost function SGD at last epoch:", costfunc[-1])
    return beta

#-----set parameters
max_iters=5000
eta_gd=0.1
n_epochs=1000
batch_size=20
eta_sgd=0.1
lmd=0.0
max_iter_SK=5000
solver_SK="lbfgs"
#-----Load data
cancer = load_breast_cancer()
X = cancer.data
y=  cancer.target
#X, y = make_moons(100, noise=0.2, random_state=7)
#-----split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=1,stratify=y)
#-----pre process data
X_train,X_test=scale_data_standard(X_train,X_test) #scale with standard scaler
#SCALE Y AS WELL?
#scale 0-1?
#sm = SMOTE(random_state=42)
#X_train, y_train = sm.fit_resample(X, y)
##-----Minimize cost functions
beta_GD= GD(X_train,y_train,max_iters,eta_gd,plot=True)
beta_SGD=SGD(X_train,y_train,n_epochs,batch_size,eta_sgd,lmd,plot=True)
#----- make SKL fit for comparison-----
clf = linear_model.LogisticRegressionCV(cv=5,max_iter=max_iter_SK,solver=solver_SK,n_jobs=-1) #use all available cores for CV
clf.fit(X_train,y_train)

#----- Predict
X_true=X_test #set predictors and data to use as true X
y_true=y_test #set targets to use as true y
y_predicted_GD = 1/(1+np.exp(-X_true@beta_GD))
y_predicted_SGD = 1/(1+np.exp(-X_true@beta_SGD))
y_predicted_SKL=clf.predict(X_test)
#----- Eevaluate model fits
Confusion_GD, accuracy_GD=Confusion_and_accuracy(y_true,y_predicted_GD)
area_score_GD=gains_plot_area(y_true,y_predicted_GD,"gd")

Confusion_SGD, accuracy_SGD=Confusion_and_accuracy(y_true,y_predicted_SGD)
area_score_SGD=gains_plot_area(y_true,y_predicted_SGD,"sgd")

Confusion_SKL, accuracy_SKL=Confusion_and_accuracy(y_true,y_predicted_SKL)
area_score_SKL=gains_plot_area(y_true,y_predicted_SKL,"skl")

#----- Print parameters and scores to terminal
print("Fit GD using")
print("iters:", max_iters,"eta:", eta_gd)
print("Confusion\n",Confusion_GD,"acc",accuracy_GD,"area score",area_score_GD)
print("Fit SGD using")
print("n_epochs:", n_epochs,"batch_size:", batch_size, "eta_sgd",eta_sgd,"lmd",lmd)
print("Confusion\n",Confusion_SGD,"acc",accuracy_SGD,"area score",area_score_SGD)
print("Fit SKL using")
print("max_iter:", max_iter_SK,"solver",solver_SK )
print("Confusion\n",Confusion_SKL,"acc",accuracy_SKL,"area score",area_score_SKL)
