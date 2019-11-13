import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier

from metrics_and_preprocessing import r2,MSE, gains_plot_area, Confusion_and_accuracy, scale_data_standard,scale_data_minmax,to_categorical_numpy
from functions_regression import FrankeFunction, create_X

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
            n_hidden_layers=2,
            n_categories=1, #binary
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data


        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_categories = n_categories
        self.n_hidden_layers=n_hidden_layers
        self.layers=np.round(np.linspace(self.n_features,self.n_categories,n_hidden_layers+2))[1:-1]
        self.layers = [ int(x) for x in self.layers ]

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.create_biases_and_weights()


    def create_biases_and_weights(self):
        self.w=[]
        self.b=[]
        self.a=[] #activations
        self.error_hidden=[]
        self.w.append(np.random.randn(self.n_features, self.layers[0]))
        self.b.append(np.zeros(self.layers[0]) + 0.01)
        self.a.append(np.zeros(self.layers[0]))
        self.error_hidden.append(np.zeros(self.layers[0]))
        for layer in range(1,self.n_hidden_layers):
            self.w.append(np.random.randn(self.layers[layer-1], self.layers[layer])) #W
            self.b.append(np.zeros(self.layers[layer]) + 0.01)
            self.a.append(np.zeros(self.layers[layer]))
            self.error_hidden.append(np.zeros(self.layers[layer]))
        self.w_o=np.random.randn(self.layers[-1], self.n_categories)
        self.b_o=np.zeros(self.n_categories) + 0.01


    def feed_forward(self):
        # feed-forward for training
        z=self.X_data @ self.w[0] + self.b[0]
        self.a[0]=tanh(z)
        for layer in range(1,self.n_hidden_layers):
            z=self.a[layer-1] @ self.w[layer] + self.b[layer]
            self.a[layer]=tanh(z)

        self.z_o =  self.a[-1] @ self.w_o + self.b_o #w_ij^L a_i^l

        self.output = self.z_o

    def feed_forward_out(self, X):
        # feed-forward for training
        z=X @ self.w[0] + self.b[0]
        self.a[0]=sigmoid(z)
        for layer in range(1,self.n_hidden_layers):
            z=self.a[layer-1] @ self.w[layer] + self.b[layer]
            self.a[layer]=sigmoid(z)

        z_o =  self.a[-1] @ self.w_o + self.b_o #w_ij^L a_i^l

        output = z_o
        return output

    def backpropagation(self):
        n_datapoint=float(np.shape(self.Y_data)[0])

        error_layer = self.output - self.Y_data #delta

        w_g = np.matmul(self.a[-1].T, error_layer) #grad WL
        b_g= np.sum(error_layer, axis=0)

        error_layer=np.matmul(error_layer, self.w_o.T) * (1-tanh(self.a[-1])**2)

        self.w_o -= self.eta * w_g*1/n_datapoint
        self.b_o -= self.eta * b_g*1/n_datapoint

        w_g = np.matmul(self.a[-2].T, error_layer) #grad Wl-1
        b_g = np.sum(error_layer, axis=0)


        for i in range(self.n_hidden_layers-2,0,-1):
            error_layer=np.matmul(error_layer, self.w[i+1].T)  * (1-tanh(self.a[i])**2)
            self.w[i+1] -= self.eta * w_g*1/n_datapoint
            self.b[i+1] -= self.eta * b_g*1/n_datapoint
            w_g = np.matmul(self.a[i-1].T, error_layer) #grad Wl-1
            b_g = np.sum(error_layer, axis=0)

        i=0
        error_layer=np.matmul(error_layer, self.w[i+1].T)  * (1-tanh(self.a[i])**2)
        self.w[i+1] -= self.eta * w_g*1/n_datapoint
        self.b[i+1] -= self.eta * b_g*1/n_datapoint
        w_g = np.matmul(self.X_data.T, error_layer) #grad Wl-1
        b_g = np.sum(error_layer, axis=0)
        self.w[0] -= self.eta * w_g*1/n_datapoint
        self.b[0] -= self.eta * b_g*1/n_datapoint

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
            costfunc[i] =costfunc[i] = 0.5*np.sum(self.output - self.Y_data)**2
        plt.plot(np.arange(0,self.epochs,1)[200:],costfunc[200:])
        plt.show()
        print("Cost function NN at last epoch:", costfunc[-1])

#-----set parameters
CC=False #set to false to use cancer data
gridsearch_eta=False #use gridsearch for eta and lambda
gridsearch_epoch=False  #use gridsearch for epochs and batch size 
np.random.seed(3)
epochs_NN=1500
batch_size_NN=20
eta_NN=0.1
lmd_NN=0.1

PP_Stratify=False
PP_Scaling=True
PP_Smote=False

#-----Load data
m = 30
n= 30
p=5
y = np.random.uniform(0, 1, m)
x = np.random.uniform(0, 1, n)
x, y = np.meshgrid(x, y)
x = np.ravel(x)  # Generate x vector
y = np.ravel(y)  # Generate y vector
y = FrankeFunction(x, y)
X = create_X(x, y, p)
y=np.row_stack(y)

print(np.shape(X))
#-----split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=1)#,stratify=y) #split with stratification

#-----pre process data
if PP_Stratify==True:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=1,stratify=y)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=1)
#-----pre process data
if PP_Scaling==True:
    X_train,X_test=scale_data_standard(X_train,X_test) #scale with standard scaler
if PP_Smote==True:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=12, ratio = 1.0)
    X_train, y_train = sm.fit_sample(X_train, y_train)

#y_train,y_test=scale_data_minmax(y_train,y_test)

#-----initiate and train network

#-----initiate and train network
if gridsearch_eta==False and gridsearch_epoch==False:
    NN=NeuralNetwork(X_train,y_train,n_hidden_layers=2,n_categories=1,epochs=epochs_NN,batch_size=batch_size_NN,eta=eta_NN,lmbd=lmd_NN)

    NN.train()

    #----- Predict
    X_true=X_test #set predictors and data to use as true X
    y_true=y_test #set targets to use as true y
    y_pred_NN1=NN.predict(X_true)

    #----- Print parameters and scores to terminal
    print("NN")
    print("epochs:", epochs_NN,"batch_size:", batch_size_NN,"eta",eta_NN,"lmd",lmd_NN)

    mse_score=MSE(y_true,y_pred_NN1)
    print(mse_score)
