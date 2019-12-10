import pandas as pd
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.metrics as sklmet
import matplotlib.pyplot as plt
import numpy as np
import keras


from functions import create_model, load_wine_split

# load data
np.random.seed(3)
X_train, X_test, y_train, y_test = load_wine_split(wine=1)
# set parameters
lr = [1e-2, 1e-5]  # learning rate
decay = [1e-6, 0]  # deacy 1e-9,
activation = ["relu"]  # , "sigmoid"] # activation

nl1 = [0]  # ,2] # numbers of layers of type nl1
nl2 = [0, 1]  # ,2] # numbers of layers of type nl2
nl3 = [0]  # ,2] # numbers of layers of type nl3

nn1 = [12]  # neurons in each layer of type nl1
nn2 = [11, 8]  # neurons in each layer of type nl2
nn3 = [10]  # neurons in each layer of type nl3

dropout = [0, 0.1]  # dropout 0.2 0.3
l1 = [0, 0.001]  # Regularizer 1
l2 = [0, 0.001]  # Regularizer 2 0.0001

epochs = [3]  # ,9]

"""
lr = [1e-5] # learning rate
decay = [1e-6] # deacy 1e-9,
activation = ["relu"] # activation
nl1 = [0]#,2] # numbers of layers of type nl1
nl2 = [0, 1] # ,2] # numbers of layers of type nl2
nl3 = [0] #,2] # numbers of layers of type nl3
nn1 = [12] # neurons in each layer of type nl1
nn2 = [11] # neurons in each layer of type nl2
nn3 = [10] # neurons in each layer of type nl3
dropout = [0,0.1] #dropout 0.2 0.3
l1 = [ 0.001] #Regularizer 1
l2 = [0.001] #Regularizer 2 0.0001
epochs=[3]
"""
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
    epochs=epochs,
)


from keras import optimizers as kerasopt
from keras import regularizers as kerasreg
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


from sklearn.metrics import mean_absolute_error, make_scorer

MAD_score = make_scorer(mean_absolute_error)


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
    output_shape=7,  # synes jo ikke på putput!
):
    # set optimizer
    opt = kerasopt.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=decay)  # decay?
    # set regularizers
    reg = kerasreg.l1_l2(l1=l1, l2=l2)

    model = Sequential()
    model.add(
        Dense(
            nn1, input_dim=input_shape, activation=act, kernel_regularizer=reg
        )  # dropout her?
    )  # first layer
    if dropout != 0:
        model.add(Dropout(dropout))
    for i in range(nl1):  # add up to nl1 layers with nn1 nodes
        model.add(Dense(nn1, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(
                Dropout(dropout)
            )  # hva er egentlig dropout? Hvor lav terskel for å fjerne nevroner?

    for i in range(nl2):  # add up to nl2 layers with nn2 nodes
        model.add(Dense(nn2, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout))

    for i in range(nl3):  # add up to nl3 layers with nn3 nodes
        model.add(Dense(nn3, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout))  # dropout her?
    model.add(Dense(output_shape, activation="softmax"))  # ouput layer
    if dropout != 0:
        model.add(Dropout(dropout))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10, verbose=1)
# model.fit(X_train.values, y_train.values)
# y_pred=model.predict_proba(X_train.values, verbose=1)
"""
grid = sklms.RandomizedSearchCV(
    estimator=model,
    cv=sklms.KFold(3),
    param_distributions=param_grid,
    verbose=1,
    n_iter=20,
    n_jobs=-1,
    scoring='neg_mean_absolute_error'
) #GridSearchCV
"""

grid = sklms.GridSearchCV(
    estimator=model,
    cv=sklms.KFold(3),
    param_grid=param_grid,
    verbose=1,
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
)


grid_result = grid.fit(X_train.values, np.argmax(y_train.values, axis=1))
cv_results_df = pd.DataFrame(grid_result.cv_results_)
cv_results_df.to_csv("gridsearch.csv")

best_params = grid_result.best_params_
best_model = grid_result.best_estimator_
best_score = grid_result.best_score_
print("Best fit params:", best_params)
print("Best score:", best_score)

y_pred = best_model.predict(X_test.to_numpy(), verbose=0)
y_test_flat = np.argmax(y_test.values, axis=1)

MAD_best = mean_absolute_error(y_test_flat, y_pred)
print("Best MAD", MAD_best)
