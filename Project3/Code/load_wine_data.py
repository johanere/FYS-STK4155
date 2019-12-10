import pandas as pd
import sklearn.preprocessing as sklpre
import sklearn.model_selection as sklms
import numpy as np


def load_wine_data(wine=1, scaler=0, resampling=0, drop_classes=True):
    # select data set
    if wine == 1:
        data = "../DataFiles/winequality-white.csv"
    elif wine == 2:
        data = ("../DataFiles/winequality-red.csv",)
    else:
        exit(
            "Program aborted. Provide wine type 1 (white) or 2 (red) in function load_wine_split"
        )

    # load data set
    df = pd.read_csv(data, sep=";")
    if drop_classes == False:
        X = df.drop("quality", axis=1)
        y = df["quality"]

    # remove quality = 9
    elif drop_classes == True:
        df_filtered = df[df["quality"] <= 8]
        df_filtered = df_filtered[df_filtered["quality"] >= 4]
        X = df_filtered.drop("quality", axis=1)
        y = df_filtered["quality"]

    if drop_classes == False and resampling > 0:
        exit(
            "Program aborted. Ccategory nine has too few neighbours for upsampling, set drop_classes=True, or resampling=0"
        )

    # stratified split
    X_train, X_test, y_train, y_test = sklms.train_test_split(
        X, y, stratify=y, test_size=0.33
    )

    # resample
    if resampling == 1:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE("not majority")
        X_sm, y_sm = smote.fit_sample(X_train, y_train)

        X_train = pd.DataFrame(X_sm, columns=X_train.columns)
        y_train = pd.DataFrame(y_sm)

    elif resampling == 2:
        from imblearn.over_sampling import ADASYN

        ada = ADASYN(
            random_state=42
        )  # https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html
        X_res, y_res = ada.fit_resample(X_train, y_train)

        X_train = pd.DataFrame(X_res, columns=X_train.columns)
        y_train = pd.DataFrame(y_res)

    # scale
    if scaler == 0:
        scaler = sklpre.MinMaxScaler()
    elif scaler == 1:
        scaler = sklpre.StandardScaler()
    else:
        exit("in load_wine_split, provide scaler 1 or 0")

    encoder = sklpre.OneHotEncoder(categories="auto", sparse=False)

    # make data frames
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train.to_numpy()), columns=X_train.columns
    )
    X_test = pd.DataFrame(
        scaler.fit_transform(X_test.to_numpy()), columns=X_test.columns
    )
    y_train = pd.DataFrame(
        encoder.fit_transform(y_train.to_numpy().reshape(-1, 1)),
        columns=encoder.categories_,
    )
    y_test = pd.DataFrame(
        encoder.fit_transform(y_test.to_numpy().reshape(-1, 1)),
        columns=encoder.categories_,
    )

    return X_train, X_test, y_train, y_test
