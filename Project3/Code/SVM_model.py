"""
check
"""

from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import sklearn.svm as sksvm
import sklearn.metrics as sklmet
import numpy as np
from load_wine_data import load_wine_data

np.random.seed(21)


wine = 1
scaler = 1
resampling = 0
drop_classes = True


if resampling == 0:
    head = "nores"
    title = "SVC without resampling"
    best_param = {
        "C": 1,
        "coef0": 0.8,
        "degree": 10,
        "gamma": "scale",
        "kernel": "poly",
        "shrinking": True,
    }
elif resampling == 2:
    head = "adasyn"
    title = "SVC with ADASYN"
    best_param = {
        "C": 1,
        "coef0": 1,
        "degree": 8,
        "gamma": "scale",
        "kernel": "poly",
        "shrinking": True,
    }


clf = sksvm.SVC(
    C=best_param["C"],
    coef0=best_param["coef0"],
    degree=best_param["degree"],
    gamma=best_param["gamma"],
    kernel=best_param["kernel"],
    shrinking=best_param["shrinking"],
)


X_train, X_test, y_train, y_test = load_wine_data(
    wine=wine, scaler=scaler, resampling=resampling, drop_classes=drop_classes
)

y_test_flat = np.argmax(y_test.to_numpy(), axis=1)
y_train_flat = np.argmax(y_train.to_numpy(), axis=1)

# fit
clf.fit(X_train.to_numpy(), y_train_flat)
# predict
y_pred = clf.predict(X_test.to_numpy())

f1 = sklmet.f1_score(y_test_flat, y_pred, average="weighted")
acc = sklmet.accuracy_score(y_test_flat, y_pred)
title = title + "\n Accuracy: $%.2f$, F1 (weighted) =$%.2f$" % (acc, f1)

# plot confusion matrix
from mlxtend.evaluate import confusion_matrix

cm = confusion_matrix(y_target=y_test_flat, y_predicted=y_pred)
fig, ax = plot_confusion_matrix(cm)
ax.set_xticklabels([""] + list(np.unique(y_test_flat) + 4))
ax.set_yticklabels([""] + list(np.unique(y_test_flat) + 4))
ax.set_title(title)
plt.savefig("../Results/Confusion/conf_SVM_%s.jpg" % head, bbox_inches="tight")
plt.show()
