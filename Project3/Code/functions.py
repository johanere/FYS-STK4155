import pandas as pd
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms

import numpy as np



def load_wine_split(wine=1):
    if wine==1:
        data="../DataFiles/winequality-white.csv"
    elif wine==2:
        data="../DataFiles/winequality-white.csv",
    else:
        exit('Program aborted. Provide wine type 1 (white) or 2 (red) in function load_wine_split')
    df = pd.read_csv(data, sep=";")
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # stratified split
    X_train, X_test, y_train, y_test = sklms.train_test_split(
        X, y, stratify=y, test_size=0.33
    )

    # pre-process data
    scaler = sklpre.MinMaxScaler()  # alternative: StandardScaler()
    encoder = sklpre.OneHotEncoder(categories="auto", sparse=False)  # onehot-encoder

    # Scale design matrix (no categorical)
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train.to_numpy()), columns=X_train.columns
    )  # to_numpy not needed?

    X_test = pd.DataFrame(
        scaler.fit_transform(X_test.to_numpy()), columns=X_test.columns
    )  # to_numpy not needed?

    y_train  = pd.DataFrame(
        encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)),
        columns=encoder.categories_,
    )

    y_test = pd.DataFrame(
        encoder.fit_transform(y_test.to_numpy().reshape(-1, 1)), columns=encoder.categories_
    )

    return X_train,X_test,y_train,y_test


from keras import optimizers as kerasopt
from keras import regularizers as kerasreg
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

def create_model(
    nl1=0,
    nl2=1,
    nl3=0,
    nn1=12,
    nn2=11,
    nn3=10,
    lr=0.01,
    decay=0.0,
    l1=0.01,
    l2=0.01,
    act="relu",
    dropout=0,
    input_shape=11,
    output_shape=7, #synes jo ikke på putput!
):
    # set optimizer
    opt = kerasopt.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=decay) #decay?
    # set regularizers
    reg = kerasreg.l1_l2(l1=l1, l2=l2)

    model = Sequential()
    model.add(
        Dense(nn1, input_dim=input_shape, activation=act, kernel_regularizer=reg) #dropout her?
    )  # first layer
    if dropout != 0:
        model.add(Dropout(dropout))
    for i in range(nl1):  # add up to nl1 layers with nn1 nodes
        model.add(Dense(nn1, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout)) #hva er egentlig dropout? Hvor lav terskel for å fjerne nevroner?

    for i in range(nl2):  # add up to nl2 layers with nn2 nodes
        model.add(Dense(nn2, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout))

    for i in range(nl3):  # add up to nl3 layers with nn3 nodes
        model.add(Dense(nn3, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout)) #dropout her?
    model.add(Dense(output_shape, activation="softmax"))  # ouput layer
    if dropout != 0:
        model.add(Dropout(dropout))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model
