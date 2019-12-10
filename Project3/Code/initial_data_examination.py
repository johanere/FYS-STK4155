"""
check
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(21)

data = "../DataFiles/winequality-white.csv"
df = pd.read_csv(data, sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]





n_features = np.shape(df)[1]

# Plot correlation matrix
fig, ax = plt.subplots(figsize=(10 * 1.618, 10))
sns.heatmap(df.corr(), cmap="YlGnBu", ax=ax, square=True, annot=True, cbar=False)
ax.set_ylim([n_features, 0])
ax.set_xlim([0, n_features])
fig.tight_layout()
plt.savefig("../Results/corrolation_X.pdf")


plt.rcParams.update({'font.size': 24})
# barplot of targets
fig, ax = plt.subplots(figsize=(10 * 1.618, 10))
plt.bar([3, 4, 5, 6, 7, 8, 9], df["quality"].value_counts().sort_index().values)
plt.xlabel("Quality")
plt.ylabel("Observations")
plt.savefig("../Results/bar_y.pdf")

print("Categories count:", df["quality"].value_counts().sort_index().values)
