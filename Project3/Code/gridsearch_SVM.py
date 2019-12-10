import pandas as pd
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import sklearn.metrics as sklmet
import sklearn.svm as sksvm
import numpy as np
from load_wine_data import load_wine_data
from datetime import datetime

np.random.seed(21)
# set parameters for choice of data abd pre-processing
wine = 1
scaler = 1
resampling = 2
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
C = [0.8, 0.9, 1]
kernel = ["rbf", "poly"]
degree = [6, 7, 8, 9, 10]
gamma = ["scale"]
coef0 = [0, 0.3, 0.7, 0.8, 0.9, 1]
shrinking = [True]
# compile dictionary of parameters
param_grid = dict(
    C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
)

# initiate model
clf = sksvm.SVC()
# construct hyper-para grid
grid = sklms.GridSearchCV(
    estimator=clf,
    cv=sklms.KFold(5),
    param_grid=param_grid,
    verbose=1,
    n_jobs=-1,
    scoring="f1_macro",
)

grid_result = grid.fit(
    X_train.values, np.argmax(y_train.values, axis=1)
)  # np.argmax(y_train.values, axis=1)
cv_results_df = pd.DataFrame(grid_result.cv_results_)
cv_results_df.to_csv("SVM_gridsearch_results/SVM_gridsearch_%s.csv" % head)

best_params = grid_result.best_params_
best_model = grid_result.best_estimator_
best_score = grid_result.best_score_
print("Best fit params:", best_params)
print("Best score:", best_score)

y_pred = best_model.predict(X_test.to_numpy())
y_test_flat = np.argmax(y_test.to_numpy(), axis=1)
acc = sklmet.accuracy_score(y_test_flat, y_pred)
print("acc", acc)
# save results and model
f = open("SVM_gridsearch_results/SVM_gridsearch_%s.txt" % head, "w+")
f.write("Best parameters: ")
f.write(np.array2string(np.asarray(best_params)))
f.write("\n")
f.write("Best score: ")
f.write(np.array2string(np.asarray(best_score)))
f.write("\nBest test score: ")
f.write("%s" % acc)
f.close()
