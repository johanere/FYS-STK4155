import numpy as np
from random import random, seed
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_validate, KFold
import sklearn.linear_model as skl


import Read_Write_Terrain

# np.random.seed(1412) #1411


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


def SVDinv(A):
    """ Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    """
    U, s, VT = np.linalg.svd(A)

    D = np.zeros((len(U), len(VT)))
    for i in range(0, len(VT)):
        D[i, i] = s[i]
    UT = np.transpose(U)
    V = np.transpose(VT)
    invD = np.linalg.inv(D)
    return np.matmul(V, np.matmul(invD, UT))


class Data:
    def __init__(self, terrainsource="../DataFiles//SRTM_data_Norway_1.tif"):
        """Initiates class. Load provided name of terrain data. 0 if no data is needed"""
        if terrainsource != 0:
            terrain = Read_Write_Terrain.read_terrain(
                terrainsource
            )  # Load terrain matrix
            self.terrain = terrain[::30, ::30]
            # Read_Write_Terrain.plot_terrain(self.terrain) #plot current terrain

    def GenerateDataFF(self, n=20, m=20, noise=0.1):
        """Generates an MxN-grid and computes targets [with noise] from Franke Function. Provide N and noise [true]/false"""
        self.n = n  # Total number of data points Y-direction
        self.m = m  # Total number of data points in X-direction
        self.N = m * n  # Total number of data points
        self.noise = noise  # 1-noise N(0,inf)
        y = np.random.uniform(0, 1, m)  # Generate uniformly spaced values of y in [0,1]
        x = np.random.uniform(0, 1, n)  # Generate uniformly spaced values of x in [0,1]
        x, y = np.meshgrid(x, y)  # Generate meshgrid
        self.x = np.ravel(x)  # Generate x vector
        self.y = np.ravel(y)  # Generate y vector
        noiseval = np.zeros(self.N)
        if self.noise > 0:
            noiseval = noise * np.random.normal(0, 1, self.N)
        z = np.ravel(FrankeFunction(self.x, self.y) + noiseval)
        self.z = z

    def GenerateDataTerrain(self):
        # SNIPP
        terrain = self.terrain
        n, m = np.shape(terrain)
        self.n = n  # Total number of data points Y-direction
        self.m = m  # Total number of data points in X-direction
        self.N = m * n
        print(n, m)
        # er det smart med linspace paa 1?
        y = np.arange(0, n, 1)
        x = np.arange(0, m, 1)
        x, y = np.meshgrid(x, y)
        self.x = np.ravel(x)
        self.y = np.ravel(y)
        self.z = np.ravel(terrain)

    def PlotTerrain(self):
        self.z_predict = self.z_predict.reshape(self.n, self.m)
        # Read_Write_Terrain.plot_terrain(self.z) #plot current terrain

    def CreateDesignMatrix(
        self, p, split="True"
    ):  # set split to False to not split data
        """  Sets up design matrix X. Splits X in training and test data"""
        x = self.x
        y = self.y
        N = len(x)
        l = int((p + 1) * (p + 2) / 2)  # Number of elements in beta
        X = np.ones((N, l))
        self.p = p
        for i in range(1, p + 1):
            q = int((i) * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = x ** (i - k) * y ** k
        self.X = X
        self.X_train, self.X_test, self.z_train, self.z_test = train_test_split(
            X, self.z, test_size=0.2
        )
        if split == "False":
            self.X_train = self.X
            self.z_train = self.z

    def OLS(self):
        """Stores a beta vector using OLS on currently set training data"""
        X = self.X_train
        z = self.z_train
        self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)

    def OLS_SVD(self):
        """Stores a beta vector using OLS on currently set training data"""
        X = self.X_train
        z = self.z_train
        self.beta = SVDinv(X.T.dot(X)).dot(X.T).dot(z)

    def Ridge(self, lmd=0):
        """Stores a beta vector using Ridge on currently set training data. Provide Lambda"""
        X = self.X_train
        z = self.z_train
        dim = np.shape(np.linalg.pinv(X.T.dot(X)))[0]
        self.beta = (
            SVDinv((X.T.dot(X)) + np.identity(dim) * lmd).dot(X.T).dot(z)
        )

    def Train_Predict(self):
        """Target prediction utilizing training data and current beta. Stores MSE and r2 scores"""
        self.z_train_predict = np.dot(self.X_train, self.beta)
        self.MSE_train = MSE(self.z_train, self.z_train_predict)
        self.r2_train = 0
        if self.N > 4:
            self.r2_train = r2(self.z_train, self.z_train_predict)

    def Test_Predict(self):
        """Target prediction utilizing test data and current beta. Stores MSE and r2 scores"""
        self.z_test_predict = np.dot(self.X_test, self.beta)
        self.MSE_test = MSE(self.z_test, self.z_test_predict)
        self.r2_test = 0
        if self.N > 4:
            self.r2_test = r2(self.z_test, self.z_test_predict)
        """"Calculate bias and variance"""


    def Skl_OLS(self, k=10):
        clf = skl.LinearRegression().fit(self.X_train, self.z_train)
        self.z_train_predict = clf.predict(self.X_train)

    def Skl_Lasso(self, lmd, k=0):
        lasso = skl.Lasso(alpha=lmd).fit(self.X_train, self.z_train)
        """training values"""
        self.z_train_predict = lasso.predict(self.X_train)
        self.MSE_train = MSE(self.z_train, self.z_train_predict)
        self.r2_train = 0
        if self.N > 4:
            self.r2_train = r2(self.z_train, self.z_train_predict)
        """test values"""
        self.z_test_predict = lasso.predict(self.X_test)
        self.MSE_test = MSE(self.z_test, self.z_test_predict)
        self.r2_test = 0
        if self.N > 4:
            self.r2_test = r2(self.z_test, self.z_test_predict)

    def Kfold_Crossvalidation(self, lmd, k=10,method=1):
        kscores_mse_test = np.zeros((2,k))
        kscores_mse_train = np.zeros((2,k))
        kscores_r2 = np.zeros((2,k))
        kfold = KFold(k, shuffle=True)
        """Retain original train/test split"""
        X = self.X
        z = self.z
        full_X_train = self.X_train
        full_X_test = self.X_test
        full_Z_train = self.z_train
        full_Z_test = self.z_test

        j = 0
        for train_inds, test_inds in kfold.split(full_X_train):
            self.X_train, self.X_test = X[train_inds], X[test_inds]
            self.z_train, self.z_test = z[train_inds], z[test_inds]
            if method==1:
                self.Ridge(lmd) #Run ridge
                self.Train_Predict()
                kscores_mse_train[0,j]=self.MSE_train
                self.Test_Predict()
                kscores_mse_test[0,j] = self.MSE_test #store values for Ridge
                kscores_r2[0,j] = self.r2_test
                self.Skl_Lasso(lmd) #Run lasso
                kscores_mse_train[1,j]=self.MSE_train
                kscores_mse_test[1,j] = self.MSE_test  #store values for  Lasso
                kscores_r2[1,j] = self.r2_test
            if method==0:
                self.OLS_SVD()
                self.Train_Predict()
                kscores_mse_train[0,j]=self.MSE_train
                self.Test_Predict()
                kscores_mse_test[0,j] = self.MSE_test

            j += 1
        self.mse_kf_train = np.mean(kscores_mse_train,axis=1, keepdims=True)
        self.mse_kf = np.mean(kscores_mse_test,axis=1, keepdims=True)
        self.r2_kf = np.mean(kscores_r2,axis=1, keepdims=True)
        """Reset to original train/test split"""
        self.X_train = full_X_train
        self.X_test = full_X_test
        self.z_train = full_Z_train
        self.z_test = full_Z_test

    def BetaConfidenceIntervals(self):
        sigma2 = np.mean(np.var(self.z_train_predict, axis=0, keepdims=True))
        X = self.X
        V = np.linalg.pinv(X.T.dot(X))
        self.beta_ci = np.zeros(len(self.beta))
        for j in range(0, len(self.beta)):
            self.beta_ci[j] = 1.96 * np.sqrt(V[j, j]) * sigma2

        """Target prediction utilizing training data and current beta. Stores MSE and r2 scores"""
        self.z_train_predict = np.dot(self.X_train, self.beta)
        self.MSE_train = MSE(self.z_train, self.z_train_predict)
        self.r2_train = 0
        if self.N > 4:
            self.r2_train = r2(self.z_train, self.z_train_predict)


# Terrain=Data(("../DataFiles//SRTM_data_Norway_1.tif"))
