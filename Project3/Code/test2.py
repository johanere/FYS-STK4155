import pandas as pd
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import matplotlib.pyplot as plt


#load data
df=pd.read_csv("winequality-white.csv",sep=";")
print(df)

X = df.drop("quality", axis=1)
y = df["quality"]


#stratified split
X_train, X_test, y_train, y_test = sklms.train_test_split(
    X, y, stratify=y, test_size=0.33
)

#smote upsamlping
from imblearn.over_sampling import SMOTE

print(np.shape(X_train))

sm = SMOTE(sampling_strategy='all',random_state=12,k_neighbors=2)
X_train, y_train = sm.fit_sample(X_train, y_train)
print(np.shape(X_train))

#scale design matrices (no categorical)
scaler = sklpre.MinMaxScaler()#StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X.columns
) #to_numpy not needed?

X_test = pd.DataFrame(
    scaler.fit_transform(X_test.to_numpy()), columns=X_test.columns
) #to_numpy not needed?

y_train = pd.DataFrame(
    y_train
)


encoder = sklpre.OneHotEncoder(categories="auto", sparse=False)
# Plotting histogram of outcomes
fig, ax = plt.subplots(figsize=(10 * 1.618, 10))

y_train.hist(ax=ax, density=False)

y_train = pd.DataFrame(
    encoder.fit_transform(y_train.reshape(-1, 1)),
    columns=encoder.categories_,
)
y_test = pd.DataFrame(
    encoder.fit_transform(y_test.to_numpy().reshape(-1, 1)),
    columns=encoder.categories_,
)

plt.show()

"""
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
"""
