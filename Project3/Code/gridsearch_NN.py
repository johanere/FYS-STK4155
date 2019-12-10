import pandas as pd
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.metrics as sklmet
from sklearn.metrics import mean_absolute_error, make_scorer
import sklearn.svm as sksvm
import numpy as np
from load_wine_data import load_wine_data
from datetime import datetime
import keras.optimizers as kerasopt
import keras.regularizers as kerasreg
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

np.random.seed(21)
# set parameters for choice of data abd pre-processing
wine = 1
scaler = 1
resampling = 0
drop_classes = True

# set time stamp
now = datetime.now()
head = now.strftime("%d%m_%H%M")
head = ("RS%s_dropC%s__f1macro" % (resampling, drop_classes)) + head

# load data
X_train, X_test, y_train, y_test = load_wine_data(
    wine=wine, scaler=scaler, resampling=resampling, drop_classes=drop_classes
)

# Set grid search parameters
n_predictors = np.shape(X_train)[1]
n_categories = np.shape(y_train)

# define neural network creator
def create_model(
    nl1=0,
    nl2=1,
    nl3=0,
    nn1=12,
    nn2=11,
    nn3=10,
    lr=0.01,
    decay=0.0,
    act="relu",
    dropout=0,
    input_shape=11,
    output_shape=5,
    beta_1=0.9,
    beta_2=0.999,
    l1=0.01,
    l2=0.01,
):
    # set optimizer
    opt = kerasopt.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
    # set regularizers
    reg = kerasreg.l1_l2(l1=0.01, l2=0.01)
    model = Sequential()
    model.add(Dense(nn1, input_dim=input_shape, activation=act))  # first layer
    if dropout != 0:
        model.add(Dropout(dropout))
    for i in range(nl1):  # add up to nl1 layers with nn1 nodes
        model.add(Dense(nn1, activation=act))
        if dropout != 0:
            model.add(Dropout(dropout))

    for i in range(nl2):  # add up to nl2 layers with nn2 nodes
        model.add(Dense(nn2, activation=act))
        if dropout != 0:
            model.add(Dropout(dropout))

    for i in range(nl3):  # add up to nl3 layers with nn3 nodes
        model.add(Dense(nn3, activation=act))
        if dropout != 0:
            model.add(Dropout(dropout))
    model.add(Dense(output_shape, activation="softmax"))  # ouput layer
    if dropout != 0:
        model.add(Dropout(dropout))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


# initiate model
model = KerasClassifier(build_fn=create_model, verbose=1)


# set parameters
lr = [1e-2, 1e-4]  # learning rate
decay = [0.001]  # for adam optimizer
activation = ["relu"]
dropout = [0.1]
epochs = [10, 50, 100]
batch_size = [10, 50, 100]

nl1 = [1]  # numbers of layers of type nl1
nl2 = [0]  # numbers of layers of type nl2
nl3 = [0]  # numbers of layers of type nl3

nn1 = [9]  # neurons in each layer of type nl1
nn2 = [7]  #  neurons in each layer of type nl2
nn3 = [0]  # neurons in each layer of type nl3

l1 = [0]  #  kernel regularizer 1
l2 = [0]  #  kernel regularizer 2

beta_1 = [0.9]  # regularizer 1 for adam optimizer
beta_2 = [0.9]  #  regularizer 2 for adam optimizer

# compile dictionary of parameters
param_grid = dict(
    nl1=nl1,
    nl2=nl2,
    nl3=nl3,
    nn1=nn1,
    nn2=nn2,
    nn3=nn3,
    act=activation,
    lr=lr,
    decay=decay,
    dropout=dropout,
    epochs=epochs,
    batch_size=batch_size,
    beta_1=beta_1,
    beta_2=beta_2,
    l1=l1,
    l2=l2,
)

# construct hyper-para grid
grid = sklms.GridSearchCV(
    estimator=model,
    cv=sklms.KFold(3),
    param_grid=param_grid,
    verbose=1,
    n_jobs=7,
    scoring="f1_weighted",
)

grid_result = grid.fit(X_train.values, np.argmax(y_train.values, axis=1))
cv_results_df = pd.DataFrame(grid_result.cv_results_)
cv_results_df.to_csv("NN_gridsearch_results/NN_gridsearch_%s.csv" % head)

best_params = grid_result.best_params_
best_model = grid_result.best_estimator_
best_score = grid_result.best_score_
print("Best fit params:", best_params)
print("Best score:", best_score)

y_pred = best_model.predict(X_test.to_numpy())
y_test_flat = np.argmax(y_test.to_numpy(), axis=1)
acc = sklmet.accuracy_score(y_test_flat, y_pred)  # accuracy score
print("acc", acc)

# save results and model
f = open("NN_gridsearch_results/NN_gridsearch_%s.txt" % head, "w+")
f.write("Best parameters: ")
f.write(np.array2string(np.asarray(best_params)))
f.write("\n")
f.write("Best score: ")
f.write(np.array2string(np.asarray(best_score)))
f.write("\nBest test score: ")
f.write("%s" % acc)
f.close()
