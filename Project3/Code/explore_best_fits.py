import pandas as pd

df = pd.read_csv("gridsearch.csv", sep=",")
best_fit = df[df.mean_test_score == df.mean_test_score.max()]
best_parameters = best_fit[
    [
        "param_nn3",
        "param_nn2",
        "param_nn1",
        "param_nl3",
        "param_nl2",
        "param_nl1",
        "param_lr",
        "param_l2",
        "param_l1",
        "param_dropout",
        "param_decay",
        "param_act",
    ]
]
print(best_parameters)
