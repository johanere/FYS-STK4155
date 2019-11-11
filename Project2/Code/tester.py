import numpy as np

p=50
cat=2
h_layers=4

a=(p-cat)//h_layers

neuron=p
for i in range(h_layers):
    neuron-=a
    print(neuron)
