import numpy as np
import matplotlib.pyplot as plt


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE

def Confusion_and_accuracy(y_true, y_pred,treshold=0.5):
    prediction_to_binary=np.where(y_pred>treshold,1,0) #make classifications
    Confusion=confusion_matrix(y_true, prediction_to_binary, labels=None, sample_weight=None)
    accuracy = (Confusion[0,0]+Confusion[1,1])/(np.sum(Confusion)) #calculate accuracy score
    return Confusion, accuracy

""""prov med andreordens deriverte!"""
def GD(X,y,max_iters=1000,eta=1,plot=False):
    beta=np.ones(np.size(X,1)) #initalize beta vector
    costfunc=np.zeros(max_iters)
    N=float(np.size(y))
    for iter in range(max_iters):
        prediction =1/(1+np.exp(-X@beta))
        gradient= -1.0/N*X.T@(y-prediction)
        beta -= eta*gradient
        costfunc[iter] = -np.sum(y*(X@beta)-np.log(1+np.exp(X@beta)))
    if plot==True:
        plt.plot(np.arange(0,max_iters,1)[1000:],costfunc[1000:])
        plt.show()
    return beta

def learning_schedule(t):
    t0 = 1
    t1 = 5
    return 0.1#t0/(t+t1)

def SGD(X,y,plot): #m-number of mini batches
    M = 5   #size of each minibatch
    n = np.size(y) #number of minibatches
    m=int(n/M)
    n_epochs=3000
    random_index = np.random.randint(m)
    beta=np.ones(np.size(X,1)) #initalize beta vector
    if plot==True:
        costfunc=np.zeros(n_epochs)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            Xi = X[random_index:random_index+M]
            yi = y[random_index:random_index+M]
            prediction =1/(1+np.exp(-Xi@beta))
            gradient= -1.0/float(M)*Xi.T@(yi-prediction)
            eta = learning_schedule(epoch*m+i)
            beta = beta - eta*gradient
        if plot==True:
            costfunc[epoch] = -np.sum(y*(X@beta)-np.log(1+np.exp(X_train@beta)))
    if plot==True:
        plt.plot(np.arange(0,n_epochs,1)[200:],costfunc[200:])
        plt.show()
    return beta


def scale_data(X_train,X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test



"""Load data"""
cancer = load_breast_cancer()
X = cancer.data
y=  cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=1,stratify=y) #split with stratification

"""moon test"""
"""
X, y = make_moons(100, noise=0.2, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
"""

"""pre process data"""
X_train,X_test=scale_data(X_train,X_test)
"""SCALE Y AS WELL!""
#scale 0-1
#sm = SMOTE(random_state=42)
#X_train, y_train = sm.fit_resample(X, y)
""" Minimize cost function"""
beta= GD(X_train,y_train,plot=True)
#beta1=SGD(X_train,y_train,plot=True)

""" fit model """
X_true=X_test # set data to compare with
y_true=y_test # set targets to compare with
y_predicted = 1/(1+np.exp(-X_true@beta))
own_Confusion, own_accuracy=Confusion_and_accuracy(y_true,y_predicted)
print("GD Own accuracy: ",own_accuracy)
print("Confusion: \n",own_Confusion)
"""
beta1=SGD(X_train,y_train,plot=True)
y_predicted = 1/(1+np.exp(-X_true@beta1))
own_Confusion, own_accuracy=Confusion_and_accuracy(y_true,y_predicted)
print("SGD Own accuracy: ",own_accuracy)
print("Confusion: \n",own_Confusion)
"""
""" make SKL fit for comparison """
clf = linear_model.LogisticRegressionCV(cv=5,max_iter=5000,solver="lbfgs",n_jobs=-1) #use all available cores for CV
clf.fit(X_train,y_train)
SKfit=clf.predict(X_true)
skl_Confusion, skl_accuracy=Confusion_and_accuracy(y_true,SKfit)
print("Skl accuracy: ",skl_accuracy)

print("Confusion: \n",skl_Confusion)
