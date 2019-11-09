import numpy as np
n=5
p=4
neurons=3
cat=2
error_output=np.ones((n,cat))
output_weights=np.ones((neurons,cat))
a_h=np.ones((n,neurons))*3

print(error_output)
print(output_weights.T)
print(a_h)
print("--")
error_hidden = np.matmul(error_output,output_weights.T)
print(error_hidden)
error_hidden=error_hidden* a_h #* (1 - a_h)

print(error_hidden)
