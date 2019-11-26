import pandas as pd
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import matplotlib.pyplot as plt


np.random.seed(3)

# load data
df = pd.read_csv("winequality-white.csv", sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

# stratified split
X_train, X_test, y_train, y_test = sklms.train_test_split(
    X, y, stratify=y, test_size=0.33
)

# pre-process data
scaler = sklpre.MinMaxScaler()  # alt: StandardScaler()
encoder = sklpre.OneHotEncoder(categories="auto", sparse=False)  # onehot-encoder

# fikse saa 10 kategorier alltid?!

# Scale design matrix (no categorical)
X_train = pd.DataFrame(
    scaler.fit_transform(X_train.to_numpy()), columns=X_train.columns
)  # to_numpy not needed?

X_test = pd.DataFrame(
    scaler.fit_transform(X_test.to_numpy()), columns=X_test.columns
)  # to_numpy not needed?

y_train = pd.DataFrame(
    encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)),
    columns=encoder.categories_,
)
y_test = pd.DataFrame(
    encoder.fit_transform(y_test.to_numpy().reshape(-1, 1)), columns=encoder.categories_
)

n_predictors = np.shape(X_train)[1]
n_categories = np.shape(y_train)[1]

from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
