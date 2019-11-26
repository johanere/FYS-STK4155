import pandas as pd
import numpy as np
import sklearn.model_selection as sklms
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv("winequality-white.csv", sep=";")
print("Dataframe")
print(df)

X = df.drop("quality", axis=1)
y = df["quality"]

np.random.seed(3)

# stratified split
X_train, X_test, y_train, y_test = sklms.train_test_split(
    X, y, stratify=y, test_size=0.33
)

print("Category count")
print(df["quality"].value_counts())

# plot
fig, ax = plt.subplots(figsize=(10 * 1.618, 10))
y_train.hist(ax=ax, density=False)
plt.xlabel("Quality")
plt.ylabel("Observations")
plt.savefig("../Results/hist_y_train.pdf")

# plot
fig, ax = plt.subplots(figsize=(10 * 1.618, 10))
y.hist(ax=ax, density=False)
plt.xlabel("Quality")
plt.ylabel("Observations")
plt.savefig("../Results/hist_y.pdf")


n_features = np.shape(df)[1]
# Plotting correlation matrix
fig, ax = plt.subplots(figsize=(10 * 1.618, 10))
sns.heatmap(df.corr(), cmap="YlGnBu", ax=ax, square=True, annot=True, cbar=False)
ax.set_ylim([n_features, 0])
ax.set_xlim([0, n_features])
fig.tight_layout()
plt.savefig("../Results/corrolation_X.pdf")
