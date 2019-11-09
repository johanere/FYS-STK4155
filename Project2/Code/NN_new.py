import numpy as np
import matplotlib.pyplot as plt

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            layers,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data
        self.layers=layers
        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.weights=[np.random.randn(self.n_features, self.layers[0])] #initialize w
        self.biases=[np.zeros(self.layers[0]) + 0.01]
        self.z=[np.zeros(self.layers[0])]
        self.a=[np.zeros(self.layers[0])]
        for i in range(1,len(self.layers)):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i-1]))
            self.biases.append(np.zeros(self.layers[i]) + 0.01)
            self.z.append(np.zeros(self.layers[i]))
            self.a.append(np.zeros(self.layers[i]))
        print(self.biases)

    def feed_forward(self):
        # feed-forward for training
        self.z[0]=self.X_data@self.weights[0]+weights[0]
        self.a[0] = sigmoid(self.z[0])
        for i in range(1,len(self.layers)):
            self.z[i]=self.a[i-1]@self.weights[i]+weights[i]
            self.a[i] = sigmoid(self.z[i])
X=np.random.random((5,5))
y=np.random.randint(2, size=5)

y_nehot = to_categorical_numpy(y)

sizes=[4,3,2]


NN=NeuralNetwork(X,y,sizes,epochs=10,batch_size=100,eta=0.1,lmbd=0.0)
