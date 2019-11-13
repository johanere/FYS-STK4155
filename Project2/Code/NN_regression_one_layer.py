import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier

from metrics_and_preprocessing import MSE, r2, scale_data_standard,scale_data_minmax,to_categorical_numpy

from functions_regression import FrankeFunction, create_X

from sklearn.linear_model import LinearRegression

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

#hidden neurons mean of input and output
class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_targets=1, #regression
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_targets = n_targets
        self.n_hidden_neurons =  int(np.mean([self.n_features,self.n_targets]))
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd



        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons) #W
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_targets)
        self.output_bias = np.zeros(self.n_targets) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = self.X_data @ self.hidden_weights + self.hidden_bias
        self.a_h = tanh(self.z_h)
        self.z_o =  self.a_h @ self.output_weights + self.output_bias #w_ij^L a_i^l
        self.output = self.z_o #activation

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = tanh(z_h)
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        output=z_o #activation function f(z)
        return output

    def backpropagation(self):
        n_datapoint=float(np.shape(self.Y_data)[0])
        error_output = self.output - self.Y_data #delta

        error_hidden = np.matmul(error_output, self.output_weights.T)*(1-tanh(self.z_h)**2)
        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient*1/n_datapoint
        self.output_bias -= self.eta * self.output_bias_gradient*1/n_datapoint
        self.hidden_weights -= self.eta * self.hidden_weights_gradient*1/n_datapoint
        self.hidden_bias -= self.eta * self.hidden_bias_gradient*1/n_datapoint

    def predict(self, X):
        output = self.feed_forward_out(X)
        return output

    def predict_probabilities(self, X):
        output = self.feed_forward_out(X)
        return output

    def train(self):
        data_indices = np.arange(self.n_inputs)

        costfunc=np.zeros(self.epochs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()

            self.X_data = self.X_data_full
            self.Y_data = self.Y_data_full
            self.feed_forward()
        #    costfunc[i] = 0.5*np.sum(self.output - self.Y_data)**2 ###
    #    plt.plot(np.arange(0,self.epochs,1)[20:],costfunc[20:])
    #    plt.show()
    #    print("Cost function NN at last epoch:", costfunc[-1])

#-----set parameters
nosplit=False

epochs_NN=1000
batch_size_NN=50
eta_NN=0.1
lmd_NN=0.1
#-----Load data
m =30
n= 30
p= 5
noise_val=0.3


y = np.random.uniform(0, 1, m)
x = np.random.uniform(0, 1, n)
x, y = np.meshgrid(x, y)
x = np.ravel(x)  # Generate x vector
y = np.ravel(y)  # Generate y vector
X = create_X(x, y, p)
z= FrankeFunction(x, y) + noise_val * np.random.normal(0, 1, m*n)
z=np.row_stack(y)

#-----split data
if nosplit==True:
    X_train=X
    X_test=X
    y_train=z
    y_test=z

else:
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.25,random_state=1) #split with stratification

#-----pre process data
X_train,X_test=scale_data_minmax(X_train,X_test)

#---
MSE_score_best=1
eta_best=0
lmd_best=0
eta_vals = np.logspace(-1, -0.5, 3)
lmbd_vals = np.logspace(-2, 0, 4)
for i in range(len(eta_vals)):
    print("on eta: " ,eta_vals[i])
    for j in range(len(lmbd_vals)):
        #-----initiate and train network
        NN=NeuralNetwork(X_train,y_train,n_targets=1,epochs=epochs_NN,batch_size=batch_size_NN,eta=eta_vals[i],lmbd=lmbd_vals[j])
        NN.train()
        y_pred_NN1=NN.predict(X_test)
        MSE_score=MSE(y_test,y_pred_NN1)
        print(MSE_score)
        if MSE_score<MSE_score_best:
            MSE_score_best=MSE_score
            eta_best=eta_vals[i]
            lmd_best=lmbd_vals[j]

reg = LinearRegression().fit(X_train, y_train)
y_skl=reg.predict(X_test)
MSE_score_sk=MSE(y_test,y_skl)
#----- Predict

y_pred_NN1=NN.predict(X_test)
MSE_score=MSE(y_test,y_pred_NN1)
#----- Eevaluate model fit



#----- Print parameters and scores to terminal


print("NN averaged MSE",MSE_score_best, "at eta,lambda",eta_best,lmd_best)

print("Sklearn ols MSE",MSE_score_sk)
#-----SKL NN
