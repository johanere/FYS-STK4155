import numpy as np
import matplotlib.pyplot as plt


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons

from imblearn.over_sampling import SMOTE  # doctest: +NORMALIZE_WHITESPACE

from metrics_and_preprocessing import (
    table_parameters_measurements,
    gains_plot_area,
    Confusion_and_accuracy,
    scale_data_standard,
    gains_area,
)


def GD(X, y, max_iters=1000, eta=0.1, plot=False):  # Gradient decent
    beta = np.ones(np.size(X, 1))  # initalize beta vector
    costfunc = np.zeros(max_iters)
    N = float(np.size(y))
    for iter in range(max_iters):
        prediction = 1 / (1 + np.exp(-X @ beta))
        gradient = -1.0 / N * X.T @ (y - prediction)
        beta -= eta * gradient
        costfunc[iter] = -np.sum(y * (X @ beta) - np.log(1 + np.exp(X @ beta)))
    if plot == True:
        plt.figure()
        print("Cost function GD at last epoch:", costfunc[-1])
        plt.style.use("seaborn-whitegrid")
        plt.plot(np.arange(0, max_iters, 1)[200:], costfunc[200:], "k")
        plt.xlabel("epochs")
        plt.ylabel("cost function")
        plt.savefig("../Results/cost_gd.pdf")
    return beta


def learning_schedule(t):  # Learning schedule for SGD
    t0 = 1
    t1 = 10
    return t0 / (t + t1)


def SGD(
    X, y, n_epochs=1000, batch_size=20, eta=0.1, lmd=0.0, plot=False
):  # stochastic GD
    n = np.size(y)  # number of data point
    m = int(n // batch_size)  # number of batches
    if m < 1:
        print("match size too large")
    random_index = np.random.randint(m)
    beta = np.ones(np.size(X, 1))  # initalize beta vector
    data_indices = np.arange(n)
    if plot == True:
        costfunc = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        for i in range(m):
            chosen_datapoints = np.random.choice(
                data_indices, size=batch_size, replace=False
            )
            Xi = X[chosen_datapoints]
            yi = y[chosen_datapoints]
            prediction = 1 / (1 + np.exp(-Xi @ beta))
            gradient = -1.0 / float(batch_size) * Xi.T @ (yi - prediction)
            if lmd > 0.0:
                gradient += lmd * gradient
            eta = eta  # learning_schedule(epoch*m+i)
            beta = beta - eta * gradient
        if plot == True:
            costfunc[epoch] = -np.sum(
                y * (X @ beta) - np.log(1 + np.exp(X_train @ beta))
            )
    if plot == True:
        plt.figure()
        plt.style.use("seaborn-whitegrid")
        plt.plot(np.arange(0, n_epochs, 1)[200:], costfunc[200:], "k")
        plt.xlabel("epochs")
        plt.ylabel("cost function")
        plt.savefig("../Results/cost_sgd.pdf")
        print("Cost function SGD at last epoch:", costfunc[-1])
    return beta


# -----set parameters
initial = True  # set to true for initial testing
gridsearch = False  # set to true for grid search
costplots = True  # set to true for cost function plots
CC = False  # set to false to use cancer data
# Set true for different aspects of pre processing IOT include them in fitting
PP_Stratify = True
PP_Scaling = True
PP_Smote = True

use_SGD = False  # True to re-fit using SGD

max_iters = 10000
eta_gd = 0.1

n_epochs = 5000
batch_size = 1000
eta_sgd = 0.1
lmd = 0.0

max_iter_SK = 5000
solver_SK = "lbfgs"


# -----Load data
if CC == True:
    from load_credit_data import load_CC_data

    X, y, col = load_CC_data(1)

if CC == False:
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

# -----split data
if PP_Stratify == True:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1
    )
# -----pre process data
if PP_Scaling == True:
    X_train, X_test = scale_data_standard(X_train, X_test)  # scale with standard scaler
if PP_Smote == True:
    sm = SMOTE(random_state=12, ratio=1.0)
    X_train, y_train = sm.fit_sample(X_train, y_train)


X_true = X_test  # set predictors and data to use as true X
y_true = y_test  # set targets to use as true y

##-----Minimize cost functions
accuracy_GD = 0
area_score_GD = 0
accuracy_SGD = 0
area_score_SGD = 0
accuracy_SKL = 0
area_score_SKL = 0

if initial == True:
    beta_GD = GD(X_train, y_train, max_iters, eta_gd, plot=costplots)

    # ----- make SKL fit for comparison-----
    clf = linear_model.LogisticRegressionCV(
        cv=5, max_iter=max_iter_SK, solver=solver_SK, n_jobs=-1
    )  # use all available cores for CV
    clf.fit(X_train, y_train)

    # ----- Predict
    y_predicted_GD = 1 / (1 + np.exp(-X_true @ beta_GD))
    y_predicted_SKL = clf.predict(X_test)
    # ----- Eevaluate model fits
    Confusion_GD, accuracy_GD = Confusion_and_accuracy(y_true, y_predicted_GD)
    area_score_GD = gains_plot_area(y_true, y_predicted_GD, "gd")

    Confusion_SKL, accuracy_SKL = Confusion_and_accuracy(y_true, y_predicted_SKL)
    area_score_SKL = gains_plot_area(y_true, y_predicted_SKL, "skl")

    # ----- Print parameters and scores to terminal
    print("Fit GD using")
    print("iters:", max_iters, "eta:", eta_gd)
    print("Confusion\n", Confusion_GD, "acc", accuracy_GD, "area score", area_score_GD)
    if use_SGD == True:
        beta_SGD = SGD(
            X_train, y_train, n_epochs, batch_size, eta_sgd, lmd, plot=costplots
        )
        y_predicted_SGD = 1 / (1 + np.exp(-X_true @ beta_SGD))
        Confusion_SGD, accuracy_SGD = Confusion_and_accuracy(y_true, y_predicted_SGD)
        area_score_SGD = gains_plot_area(y_true, y_predicted_SGD, "sgd")
        print("Fit SGD using")
        print(
            "n_epochs:",
            n_epochs,
            "batch_size:",
            batch_size,
            "eta_sgd",
            eta_sgd,
            "lmd",
            lmd,
        )
        print(
            "Confusion\n",
            Confusion_SGD,
            "acc",
            accuracy_SGD,
            "area score",
            area_score_SGD,
        )
    print("Fit SKL using")
    print("max_iter:", max_iter_SK, "solver", solver_SK)
    print(
        "Confusion\n", Confusion_SKL, "acc", accuracy_SKL, "area score", area_score_SKL
    )

    table_parameters_measurements(
        "../Results/parameters.txt",
        accuracy_GD,
        area_score_GD,
        max_iters,
        eta_gd,
        accuracy_SGD,
        area_score_SGD,
        n_epochs,
        batch_size,
        eta_sgd,
        lmd,
        accuracy_SKL,
        area_score_SKL,
        max_iter_SK,
    )

# visual representation of grid search - uses seaborn heatmap
if gridsearch == True:

    import seaborn as sns

    eta_vals = np.logspace(-4, 1, 6)
    lmbd_vals = np.logspace(-4, 1, 6)
    sns.set()
    print(len(eta_vals))
    train_area = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    for i in range(len(eta_vals)):
        print("on eta: ", eta_vals[i])
        for j in range(len(lmbd_vals)):
            beta_loop = SGD(
                X_train,
                y_train,
                n_epochs,
                batch_size,
                eta_vals[i],
                lmbd_vals[j],
                plot=False,
            )
            y_predicted_loop = 1 / (1 + np.exp(-X_true @ beta_loop))
            area_sc, accuracy_sc = gains_area(y_true, y_predicted_loop)

            train_area[i][j] = area_sc
            test_accuracy[i][j] = accuracy_sc

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        train_area,
        annot=True,
        ax=ax,
        cmap="viridis",
        xticklabels=eta_vals,
        yticklabels=lmbd_vals,
    )
    ax.set_title("Gains area ratio")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        test_accuracy,
        annot=True,
        ax=ax,
        cmap="viridis",
        xticklabels=eta_vals,
        yticklabels=lmbd_vals,
    )
    ax.set_title("Test Accuracy")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    plt.show()
