import numpy as np
import pandas as pd

x = np.random.rand(100, 1)
y = 5*x*x+0.1*np.random.randn(100, 1)

df=pd.DataFrame(y)
print(df)

beta=np.linalg.inv(x.T.DOT(X)).dot(X.T).dot(y)
