"""
check
"""


from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import sklearn.svm as sksvm
import sklearn.metrics as sklmet
import sklearn.utils as sklut
import numpy as np
from load_wine_data import load_wine_data
import keras.optimizers as kerasopt
import keras.regularizers as kerasreg
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


np.random.seed(21)



wine=1
scaler=1
resampling=2
drop_classes=True
grid=False
training=False

if resampling==0:
    head="nores"
    title="FFNN without resampling"
    best_parameters =  {'act': 'relu', 'batch_size': 50, 'beta_1': 0.9, 'beta_2': 0.9, 'decay': 0.001, 'dropout': 0.03, 'epochs': 20, 'l1': 0, 'l2': 0, 'lr': 0.01, 'nl1': 1, 'nl2': 1, 'nl3': 0, 'nn1': 9, 'nn2': 7, 'nn3': 0}

elif resampling==2:
    head="adasyn_adjust_f"
    title="FFNN with ADASYN"
    best_parameters =  {'act': 'relu', 'batch_size': 800, 'beta_1': 0.9, 'beta_2': 0.99, 'decay': 0.001, 'dropout': 0.0, 'epochs': 1000, 'l1': 0.3, 'l2': 0.3, 'lr': 0.05, 'nl1': 1, 'nl2': 1, 'nl3': 1, 'nn1': 10, 'nn2': 9, 'nn3':8}
if grid==True:
    if resampling==0:
        head="grid_nores"
        title="FFNN without resampling. Grid-results"
        best_parameters = {'act': 'relu', 'batch_size': 10, 'beta_1': 0.9, 'beta_2': 0.9, 'decay': 0.001, 'dropout': 0.1, 'epochs': 50, 'l1': 0, 'l2': 0, 'lr': 0.01, 'nl1': 1, 'nl2': 0, 'nl3': 0, 'nn1': 9, 'nn2': 7, 'nn3': 0}
    elif resampling==2:
        head="grid_adasyn"
        title="FFNN with ADASYN. Grid-results"
        best_parameters={'act': 'relu', 'batch_size': 200, 'beta_1': 0.9, 'beta_2': 0.9, 'decay': 0, 'dropout': 0.1, 'epochs': 8, 'l1': 0, 'l2': 0, 'lr': 0.01, 'nl1': 1, 'nl2': 0, 'nl3': 0, 'nn1': 11, 'nn2': 7, 'nn3': 0}


#remember to adjust output_shape for nine true or false


X_train, X_test, y_train, y_test = load_wine_data(wine=wine, scaler=scaler,resampling=resampling,drop_classes=drop_classes)

# Set grid search parameters
n_predictors = np.shape(X_train)[1]
n_categories = np.shape(y_train)

#define neural network creator
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
    beta_2=0.999
):
    # set optimizer
    opt = kerasopt.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
    # set regularizers

    model = Sequential()
    model.add(
        Dense(
            nn1, input_dim=input_shape, activation=act     )
    )  # first layer
    if dropout != 0:
        model.add(Dropout(dropout))
    for i in range(nl1):  # add up to nl1 layers with nn1 nodes
        model.add(Dense(nn1, activation=act))
        if dropout != 0:
            model.add(
                Dropout(dropout)
            )

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

#construct model
model = KerasClassifier(build_fn=create_model, act=best_parameters["act"],  batch_size=best_parameters["batch_size"], decay=best_parameters["decay"], dropout=best_parameters["dropout"], epochs=best_parameters["epochs"],
 lr=best_parameters["lr"] , nl1=best_parameters["nl1"] ,   nl2=best_parameters["nl2"] , nl3=best_parameters["nl3"]
, nn1=best_parameters["nn1"] , nn2=best_parameters["nn2"] , nn3=best_parameters["nn3"] ,verbose=1) # l1=best_parameters["l1"] , l2=best_parameters["l2"] ,

#fit
model.fit(X_train.values, y_train.values)
#predict
if training==True:
    X_test=X_train #switch training and test
y_pred=model.predict(X_test.values, verbose=1)



y_test_flat = np.argmax(y_test.to_numpy(), axis=1)
y_train_flat = np.argmax(y_train.to_numpy(), axis=1)
if training==True:
    y_test_flat=y_train_flat
    head=head+"training"
    title=title+" on training data"
#metrics'

f1=sklmet.f1_score(y_test_flat, y_pred,average='weighted')
acc = sklmet.accuracy_score(y_test_flat, y_pred)
title=title+"\n Accuracy: $%.2f$, F1 (weighted) =$%.2f$"%(acc,f1)

#plot confusion matrix
from mlxtend.evaluate import confusion_matrix
cm = confusion_matrix(y_target=y_test_flat,
                      y_predicted=y_pred)
fig, ax = plot_confusion_matrix(cm)
ax.set_xticklabels([''] + list(np.unique(y_test_flat)+3))
ax.set_yticklabels([''] + list(np.unique(y_test_flat)+3))
ax.set_title(title)
plt.savefig("../Results/Confusion/conf_NN_%s.jpg"%head,bbox_inches='tight' )
plt.show()

floof=model.get_params()
print(floof)
