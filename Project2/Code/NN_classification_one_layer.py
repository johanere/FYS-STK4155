import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier

from metrics_and_preprocessing import gains_plot_area, Confusion_and_accuracy, scale_data_standard,scale_data_minmax,to_categorical_numpy


def sigmoid(x):
    return 1/(1 + np.exp(-x))

#hidden neurons mean of input and output
class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            #n_hidden_neurons=50,
            n_categories=2, #binary
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_categories = n_categories
        self.n_hidden_neurons =  int(np.mean([self.n_features,self.n_categories]))
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd


        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons) #W
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = self.X_data @ self.hidden_weights + self.hidden_bias

        self.a_h = sigmoid(self.z_h)
        self.z_o =  self.a_h @ self.output_weights + self.output_bias #w_ij^L a_i^l
        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        return probabilities

    def backpropagation(self):
        n_datapoint=float(np.shape(self.Y_data)[0])

        error_output = self.probabilities - self.Y_data #delta
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

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
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

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
            costfunc[i] = 1/2*np.sum((self.probabilities[:,0]-self.Y_data[:,0])**2) ###
        plt.plot(np.arange(0,self.epochs,1)[200:],costfunc[200:])
        plt.show()
        print("Cost function NN at last epoch:", costfunc[-1])

#-----set parameters
epochs_NN=1500
batch_size_NN=20
eta_NN=0.1
lmd_NN=0.0
#-----Load data
cancer = load_breast_cancer()
X = cancer.data
y=  cancer.target
#X, y = make_moons(100, noise=0.2, random_state=7)
#-----split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=1,stratify=y) #split with stratification

#-----pre process data
X_train,X_test=scale_data_standard(X_train,X_test)
y_train_onehot, y_test_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test)

#-----initiate and train network
NN=NeuralNetwork(X_train,y_train_onehot,n_categories=2,epochs=epochs_NN,batch_size=batch_size_NN,eta=eta_NN,lmbd=lmd_NN)
NN.train()

#----- Predict
X_true=X_test #set predictors and data to use as true X
y_true=y_test #set targets to use as true y
y_pred_NN1=NN.predict(X_true)

#----- Eevaluate model fit
Confusion_NN1, accuracy_NN1=Confusion_and_accuracy(y_true,y_pred_NN1)
area_score_NN1=gains_plot_area(y_true,y_pred_NN1,"NN")


#----- Print parameters and scores to terminal
print("NN")
print("epochs:", epochs_NN,"batch_size:", batch_size_NN,"eta",eta_NN,"lmd",lmd_NN)
print("Confusion\n",Confusion_NN1,"acc",accuracy_NN1,"area score",area_score_NN1)


#-----SKL NN
"""
# store models for later use
eta_vals = np.logspace(-4, 1, 6) #-5, 1,7
lmbd_vals = np.logspace(-4, 1, 6)
n_hidden_neurons = 50
epochs=1000
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
        dnn.fit(X_train, y_train)

        DNN_scikit[i][j] = dnn

        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", dnn.score(X_test, y_test))
        print()
"""
