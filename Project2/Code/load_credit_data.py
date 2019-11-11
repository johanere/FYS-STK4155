from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

def load_CC_data(which_onehot = 1): #load with outlier deletion and onehot encoding
    nanDict = {}
    df = pd.read_excel("default_of_credit_card_clients.xls", header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace=True)

    X = df.loc[:, df.columns != 'defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns == 'defaultPaymentNextMonth'].values

    #find and remove outliers in the categories with known value ranges
    outlier_gender1 = np.where(X[:,1] < 1)[0]
    outlier_gender2 = np.where(X[:,1] > 2)[0]

    outlier_education1 = np.where(X[:,2] < 1)[0]
    outlier_education2 = np.where(X[:,2] > 4)[0]

    outlier_marital1 = np.where(X[:,3] < 1)[0]
    outlier_marital2 = np.where(X[:,3] > 3)[0]

    inds = np.concatenate((outlier_gender1,
                           outlier_gender2,
                           outlier_education1,
                           outlier_education2,
                           outlier_marital1,
                           outlier_marital2))

    outlier_rows = np.unique(inds)

    X = np.delete(X, outlier_rows, axis=0)
    y = np.delete(y, outlier_rows, axis=0)

    

    #split data into categorical and continuous features
    if which_onehot==1: #marriage, sex and education one-hot

        categorical_inds = (1, 2, 3)
        continuous_inds = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)


    elif which_onehot==2: #all categories one-hot + age and defaulting last months

        categorical_inds = (1, 2, 3, 5, 6, 7, 8, 9, 10)
        continuous_inds = (0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)

    else:
        print('specify onehot option 1 or 2')
        exit(0)

    X_cat = X[:,categorical_inds]
    X_cont = X[:, continuous_inds]

    #onehot encode categorical data
    onehotencoder = OneHotEncoder(categories="auto", sparse=False)
    preprocessor = ColumnTransformer(
            remainder="passthrough",
            transformers=[
                ('onehot', onehotencoder, list(range(X_cat.shape[1])))])

    X_cat = preprocessor.fit_transform(X_cat)

    #join categorical and continuous features
    X = np.concatenate((X_cont, X_cat), axis=1)

    cont_feat_inds = list(range(X_cont.shape[1]))

    return X, np.ravel(y), cont_feat_inds
