import numpy as np
import matplotlib.pyplot as plt

n = 200  # data points
m = 1  # predictors
deg = 2  # polynomial degree


# Generate data
x = np.random.rand(n, 1)
y = 5 * x * x + 0.1 * np.random.randn(n, 1)

# Setting up design matrix
X = np.ones((n, deg + 1))
for i in range(1, deg + 1):
    X[:, i] = x[:, 0] ** i

#solve for beta
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

#fit series
def fit(x):
    sum=0
    for i in range(0,deg+1):
        sum+=beta[i]*x**i
    return sum
domain=np.linspace(0,1,50)
f=fit(domain)

# Plot
plt.scatter(x,y)
plt.plot(domain,f,'r-')
plt.show()
