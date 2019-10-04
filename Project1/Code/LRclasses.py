from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate
import sklearn.linear_model as skl
from sklearn.model_selection import KFold

import Read_Write_Terrain


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def r2(z1, z2):
    r2 = r2_score(z1, z2, sample_weight=None, multioutput="uniform_average")
    return r2


def MSE(z1, z2):
    MSE = mean_squared_error(z1, z2, sample_weight=None, multioutput="uniform_average")
    return MSE


class Data:
    def __init__(self, p=5):
        """Initiates class. Provide degree of polynomial, p"""
        self.p = p

    def GenerateDataFF(self, N=30, noise=0):
        """Generates a random NxN-grid and computes targets [with noise] from Franke Function. Provide N and noise [true]/false"""
        self.N = N
        self.n = N ** 2  # Total number of data points
        self.noise = noise  # 1-noise N(0,inf)
        x = np.random.uniform(0, 1, N)
        y = np.random.uniform(0, 1, N)
        x, y = np.meshgrid(x, y)
        self.x = np.ravel(x)
        self.y = np.ravel(y)
        noiseval = np.zeros(self.n)
        if self.noise == 1:
            noiseval = 0.5 * np.random.normal(0, 1, self.n)
        z = FrankeFunction(self.x, self.y) + noiseval
        self.z = z

    def GenerateDataTerrain(self, terrain):
        n,m=np.shape(terrain)
        #er det smart med linspace paa 1?
        y=np.linspace(1,n,n)
        x=np.linspace(1,m,m)
        x,y=np.meshgrid(x,y)
        self.x = np.ravel(x)
        self.y = np.ravel(y)
        self.z=np.ravel(terrain1)

    def CreateDesignMatrix(self):
        """  Sets up design matrix X. Splits X in training and test data"""
        x = self.x
        y = self.y
        p = self.p
        N = len(x)
        l = int((p + 1) * (p + 2) / 2)  # Number of elements in beta
        X = np.ones((N, l))
        for i in range(1, p + 1):
            q = int((i) * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = x ** (i - k) * y ** k
            self.X = X
            self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(
                X, self.z, test_size=0.2
            )

    def OLS(self):
        """Stores a beta vector using OLS on currently set training data"""
        X = self.X_train
        z = self.z_train
        self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)

    def Ridge(self, ridge_lambda=0):
        """Stores a beta vector using Ridge on currently set training data. Provide Lambda"""
        X = self.X_train
        z = self.z_train
        dim = np.shape(np.linalg.pinv(X.T.dot(X)))[0]
        self.beta = (
            np.linalg.pinv((X.T.dot(X)) + np.identity(dim) * ridge_lambda)
            .dot(X.T)
            .dot(z)
        )

    def Train_Predict(self):
        """Target prediction utilizing training data and current beta. Stores MSE and r2 scores"""
        self.z_train_predict = np.dot(self.X_train, self.beta)
        self.MSE_train = MSE(Data.z_train, self.z_train_predict)
        self.r2_train = 0
        if self.n > 4:
            self.r2_train = r2(Data.z_train, self.z_train_predict)

    def Test_Predict(self):
        """Target prediction utilizing test data and current beta. Stores MSE and r2 scores"""
        self.z_test_predict = np.dot(self.X_test, self.beta)
        self.MSE_test = MSE(self.z_test, self.z_test_predict)
        self.r2_test = 0
        if self.n > 4:
            self.r2_test = r2(self.z_test, self.z_test_predict)

    def Skl_OLS(self, k=10):
        clf = skl.LinearRegression().fit(self.X_train, self.z_train)
        ytilde = clf.predict(self.X_train)
        self.MSE_SKL = MSE(Data.z_train, ytilde)
        self.r2_SKL = 0
        #if self.n > 4:
            self.r2_SKL = r2(Data.z_train, ytilde)
        cv_results = cross_validate(clf, self.X, self.z, cv=k)
        print("OLS score \n",cv_results["test_score"])

    def Skl_Lasso(self, k=10):
        lasso = skl.Lasso().fit(self.X_train, self.z_train)
        ytilde = lasso.predict(self.X_train)
        self.MSE_SKL_Lasso = MSE(Data.z_train, ytilde)
        self.r2_SKL_Lasso = 0
        if self.n > 4:
            self.r2_SKL_Lasso = r2(Data.z_train, ytilde)
        cv_results = cross_validate(lasso, self.X, self.z, cv=k)
        print("Lasso score \n", cv_results["test_score"])

    def Kfold_Crossvalidation(self, method=0, llambda=0, k=10):
        "k fold cross validation. 0-OLS, 1-Ridge. Gets MSE from test predict"
        self.kscores = np.zeros((2, k))
        kfold = KFold(k, shuffle=True)
        j = 0
        for train_inds, test_inds in kfold.split(self.X):
            self.X_train, self.X_test = self.X[train_inds], self.X[test_inds]
            self.z_train, self.z_test = self.z[train_inds], self.z[test_inds]
            if method == 0:
                self.OLS()
            if method == 1:
                self.Ridge(llambda)
            self.Train_Predict()
            self.Test_Predict()
            self.kscores[0, j] = self.MSE_test
            self.kscores[1, j] = self.r2_test
            j += 1


np.random.seed(3)
"""
Data = Data(p=5)  # Initiate with polynomial degree
Data.GenerateDataFF(
    N=100, noise=1
)  # N - square of number of data points, set noise == 1 to add noise
Data.CreateDesignMatrix()
Data.OLS()
Data.Train_Predict()
Data.Test_Predict()
Data.Skl_Lasso()
Data.Skl_OLS()
#Data.SklOLS()

#print("MSE SKL Train: %f" % Data.MSE_SKL)
#print("MSE Own Train: %f" % Data.MSE_train)
#print("MSE Own Test:  %f" % Data.MSE_test)
Data.Kfold_Crossvalidation()
#print(Data.kscores)
Data.Kfold_Crossvalidation(method=1, llambda=1, k=10)
"""
Data = Data(p=5)
terrain1=Read_Write_Terrain.read_terrain()
print(np.shape(terrain1))
Read_Write_Terrain.plot_terrain(terrain1)

Data.GenerateDataTerrain(terrain1)
Data.CreateDesignMatrix()
Data.OLS()
Data.Skl_OLS()
print("MSE SKL Train: %f" % Data.MSE_SKL)
