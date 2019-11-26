

import sklearn.svm as sklsvm
import sklearn.model_selection as sklms
import sklearn.metrics as sklmet
import pandas as pd

from sklearn.datasets import load_breast_cancer, make_moons


cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = sklms.train_test_split(
    X, y, stratify=y, test_size=0.33
)

#what parameters are neccecary? We've only learned about shrinkage. What's even C?
kernel=('linear','rbf') # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ - how to chose? try all?
C=[1,10]
#shrinking=['True','False']
param_grid = dict(
kernel=kernel,
C=C
#shrinking =
)

svc = sklsvm.SVC(gamma="scale")
grid = sklms.GridSearchCV(svc, param_grid, cv=2,verbose=20)
grid_result = grid.fit(X_train, y_train)


cv_results_df = pd.DataFrame(grid_result.cv_results_)
cv_results_df.to_csv("gridsearch_SVM.csv")
best_params = grid_result.best_params_
best_model = grid_result.best_estimator_

print("Best fit params:", best_params)

svm_predict=best_model.predict(X_test)
print(svm_predict)
f1=sklmet.f1_score(y_test,svm_predict)
print(f1)
