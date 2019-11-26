import pandas as pd
import numpy as np
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.metrics as sklmet
import matplotlib.pyplot as plt
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from functions import multilabel_f1

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
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=decay) #decay?
    # set regularizers
    reg = keras.regularizers.l1_l2(l1=l1, l2=l2)

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
    """remember to change metric.
    What is the function of metrics in compile, other than being metric for ex randomizedsearch if no metric is specified?
    dropout on first and last layer?
    """
    return model


np.random.seed(3)

# load data
df = pd.read_csv("winequality-white.csv", sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

# stratified split
X_train, X_test, y_train, y_test = sklms.train_test_split(
    X, y, stratify=y, test_size=0.90
)

# pre-process data
scaler = sklpre.MinMaxScaler()  # alt: StandardScaler()
encoder = sklpre.OneHotEncoder(categories="auto", sparse=False)  # onehot-encoder

# fikse saa 10 kategorier alltid?!
# PD-struktur fornuftig?
# Scale design matrix (no categorical)
X_train = pd.DataFrame(
    scaler.fit_transform(X_train.to_numpy()), columns=X_train.columns
)  # to_numpy not needed?

X_test = pd.DataFrame(
    scaler.fit_transform(X_test.to_numpy()), columns=X_test.columns
)  # to_numpy not needed?

#one hot - nokså nødvendig ved flere kategorier?
y_train = pd.DataFrame(
    encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)),
    columns=encoder.categories_,
)
y_test = pd.DataFrame(
    encoder.fit_transform(y_test.to_numpy().reshape(-1, 1)), columns=encoder.categories_
)


n_predictors = np.shape(X_train)[1]
n_categories = np.shape(y_train)


model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10, verbose=1)

model.fit(X_train, y_train)
ypred=model.predict(X_train) #feil form på!
print(ypred)
print(y_train)
"""
# learning rate
lr = [1e-2, 1e-3]  # , 1e-4]
# deacy ???
decay = [1e-6, 1e-9]  # ,0]

# activation
activation = ["relu", "sigmoid"]

# numbers of layers
nl1 = [0, 1]  # ,2]
nl2 = [0, 1]  # ,2]
nl3 = [0]  # ,1]#,2]

# neurons in each layer
nn1 = [12]  # ,8]#,20]
nn2 = [11]  # ,8]#,14]
nn3 = [10, 6]  # ,9]

# dropout and regularisation
dropout = [0, 0.1, 0.2, 0.3]
l1 = [0, 0.01, 0.003, 0.001, 0.0001]
l2 = [0, 0.01, 0.003, 0.001, 0.0001]

dropout = [0.1, 0.2]
l1 = [0]  # ,0.1]
l2 = [0]  # ,0.1]
# dictionary summary

param_grid = dict(
    nl1=nl1,
    nl2=nl2,
    nl3=nl3,
    nn1=nn1,
    nn2=nn2,
    nn3=nn3,
    act=activation,
    l1=l1,
    l2=l2,
    lr=lr,
    decay=decay,
    dropout=dropout,
)




grid = sklms.RandomizedSearchCV(
    estimator=model,
    cv=sklms.KFold(3),
    param_distributions=param_grid,
    verbose=1,
    n_iter=10,
    n_jobs=-1,
    scoring="f1_weighted"
)  # scoring='f1_weighted') - default uses estimator scoring
grid_result = grid.fit(X_train, y_train)
cv_results_df = pd.DataFrame(grid_result.cv_results_)
cv_results_df.to_csv("gridsearch.csv")

best_params = grid_result.best_params_
best_model = grid_result.best_estimator_

print("Best fit params:", best_params)
"""
