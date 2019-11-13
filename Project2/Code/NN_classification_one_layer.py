import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier

from metrics_and_preprocessing import (
    gains_plot_area,
    Confusion_and_accuracy,
    scale_data_standard,
    scale_data_minmax,
    to_categorical_numpy,
    gains_area,
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# hidden neurons mean of input and output
class NeuralNetwork:
    def __init__(
        self,
        X_data,
        Y_data,
        # n_hidden_neurons=50,
        n_categories=2,  # binary
        epochs=10,
        batch_size=100,
        eta=0.1,
        lmbd=0.0,
    ):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_categories = n_categories
        self.n_hidden_neurons = int(np.mean([self.n_features, self.n_categories]))
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(
            self.n_features, self.n_hidden_neurons
        )  # W
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = self.X_data @ self.hidden_weights + self.hidden_bias

        self.a_h = sigmoid(self.z_h)
        self.z_o = self.a_h @ self.output_weights + self.output_bias  # w_ij^L a_i^l
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
        n_datapoint = float(np.shape(self.Y_data)[0])

        error_output = self.probabilities - self.Y_data  # delta
        error_hidden = (
            np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)
        )

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)  # grad WL
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(
            self.X_data.T, error_hidden
        )  # grad Wl-1
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient * 1 / n_datapoint
        self.output_bias -= self.eta * self.output_bias_gradient * 1 / n_datapoint
        self.hidden_weights -= self.eta * self.hidden_weights_gradient * 1 / n_datapoint
        self.hidden_bias -= self.eta * self.hidden_bias_gradient * 1 / n_datapoint

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self, plot=False):
        data_indices = np.arange(self.n_inputs)

        costfunc = np.zeros(self.epochs)

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
            costfunc[i] = -np.sum(
                self.Y_data[:, 0] * np.log(self.probabilities[:, 0])
                + (1 - self.Y_data[:, 0]) * np.log(1 - self.probabilities[:, 0])
            )
        if plot == True:
            plt.figure()
            plt.style.use("seaborn-whitegrid")
            plt.plot(np.arange(0, self.epochs, 1)[200:], costfunc[200:], "k")
            plt.xlabel("epochs")
            plt.ylabel("cost function")
            plt.savefig("../Results/cost_NN.pdf")
            print("Cost function NN at last epoch:", costfunc[-1])
            plt.show()


# -----set parameters
CC = True  # set to false to use cancer data
gridsearch_eta = False  # use parallel gridsearch for eta and lambda
gridsearch_epoch = True
epochs_NN = 1000
batch_size_NN = 1000
eta_NN = 0.1
lmd_NN = 1

PP_Stratify = False
PP_Scaling = True
PP_Smote = True

# -----Load data
if CC == True:
    from load_credit_data import load_CC_data

    X, y, col = load_CC_data(1)

if CC == False:
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

# -----split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1
)  # ,stratify=y) #split with stratification

# -----pre process data
if PP_Stratify == True:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1, stratify=y
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1
    )
# -----pre process data
if PP_Scaling == True:
    X_train, X_test = scale_data_standard(X_train, X_test)  # scale with standard scaler
if PP_Smote == True:
    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=12, ratio=1.0)
    X_train, y_train = sm.fit_sample(X_train, y_train)

y_train_onehot, y_test_onehot = (
    to_categorical_numpy(y_train),
    to_categorical_numpy(y_test),
)

# -----initiate and train network
if gridsearch_eta == False and gridsearch_epoch == False:
    NN = NeuralNetwork(
        X_train,
        y_train_onehot,
        n_categories=2,
        epochs=epochs_NN,
        batch_size=batch_size_NN,
        eta=eta_NN,
        lmbd=lmd_NN,
    )
    NN.train(plot=True)

    # ----- Predict
    X_true = X_test  # set predictors and data to use as true X
    y_true = y_test  # set targets to use as true y
    y_pred_NN1 = NN.predict(X_true)

    # ----- Eevaluate model fit
    Confusion_NN1, accuracy_NN1 = Confusion_and_accuracy(y_true, y_pred_NN1)
    area_score_NN1 = gains_plot_area(y_true, y_pred_NN1, "NN")

    # ----- Print parameters and scores to terminal
    print("NN")
    print(
        "epochs:", epochs_NN, "batch_size:", batch_size_NN, "eta", eta_NN, "lmd", lmd_NN
    )
    print(
        "Confusion\n", Confusion_NN1, "acc", accuracy_NN1, "area score", area_score_NN1
    )

# visual representation of grid search - uses seaborn heatmap
if gridsearch_eta == True:  # not parallel

    X_true = X_test  # set predictors and data to use as true X
    y_true = y_test  # set targets to use as true y

    import seaborn as sns

    eta_vals = np.logspace(-4, 1, 6)
    lmbd_vals = np.logspace(-4, 1, 6)
    sns.set()

    train_area = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    for i in range(len(eta_vals)):
        print("on eta: ", eta_vals[i])
        for j in range(len(lmbd_vals)):
            NN = NeuralNetwork(
                X_train,
                y_train_onehot,
                n_categories=2,
                epochs=epochs_NN,
                batch_size=batch_size_NN,
                eta=eta_vals[i],
                lmbd=lmbd_vals[j],
            )
            NN.train(plot=False)
            y_pred_NN1 = NN.predict(X_true)
            area_sc, accuracy_sc = gains_area(y_true, y_pred_NN1)
            train_area[i][j] = area_sc
            test_accuracy[i][j] = accuracy_sc

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        train_area,
        annot=True,
        ax=ax,
        cmap="viridis",
        xticklabels=eta_vals,
        yticklabels=lmbd_vals,
    )
    ax.set_title("Gains area ratio")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    plt.savefig("../Results/gridsearch_NN_area.pdf")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        test_accuracy,
        annot=True,
        ax=ax,
        cmap="viridis",
        xticklabels=eta_vals,
        yticklabels=lmbd_vals,
    )
    ax.set_title("Test Accuracy")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")
    plt.savefig("../Results/gridsearch_NN_acc.pdf")
    plt.show()

if gridsearch_epoch == True:  # not parallel

    X_true = X_test  # set predictors and data to use as true X
    y_true = y_test  # set targets to use as true y

    import seaborn as sns

    epochs = np.asarray([100, 500, 1000, 1500])
    batch_size = np.asarray([500, 1000, 1500, 2000])
    sns.set()

    train_area = np.zeros((len(epochs), len(batch_size)))
    test_accuracy = np.zeros((len(epochs), len(batch_size)))

    for i in range(len(epochs)):
        print("on epoch: ", epochs[i])
        for j in range(len(batch_size)):
            NN = NeuralNetwork(
                X_train,
                y_train_onehot,
                n_categories=2,
                epochs=epochs[i],
                batch_size=batch_size[j],
                eta=eta_NN,
                lmbd=lmd_NN,
            )
            NN.train(plot=False)
            y_pred_NN1 = NN.predict(X_true)
            area_sc, accuracy_sc = gains_area(y_true, y_pred_NN1)
            train_area[i][j] = area_sc
            test_accuracy[i][j] = accuracy_sc
            print("on batch: ", epochs[j])

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        train_area,
        annot=True,
        ax=ax,
        cmap="viridis",
        xticklabels=epochs,
        yticklabels=batch_size,
    )
    ax.set_title("Gains area ratio, $\eta=%.2f$, $\lambda=%.2f$" % (eta_NN, lmd_NN))
    ax.set_xlabel("epochs")
    ax.set_ylabel("batch size")
    plt.savefig("../Results/gridsearch_NN_area_epocs.pdf")
    plt.show()
