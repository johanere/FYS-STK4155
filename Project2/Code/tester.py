import numpy as np

a=np.zeros(10)

b=np.zeros((6,1))
print(np.shape(a))
a=np.row_stack(a)
print(np.shape(a))
print(np.shape(b.T))

c=a@b.T
print(c)
