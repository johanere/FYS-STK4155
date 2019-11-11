import numpy as np
import matplotlib.pyplot as plt

def gains_area(y_true, y_pred):
    N=float(len(y_true)) #Number of entries
    index_true = np.argsort(y_true)[::-1]
    index_pred = np.argsort(y_pred)[::-1]

    y_true_sorted_pred = y_true[index_pred] #predict index
    y_true_sorted_true = y_true[index_true] #test index

    gains_optimal = np.cumsum(y_true_sorted_pred)/np.sum(y_true_sorted_pred)
    gains_model = np.cumsum(y_true_sorted_true)/np.sum(y_true_sorted_true)
    baserate = np.arange(1, N+1)/N

    A1 = np.trapz(gains_model, baserate) #Area under model curve
    A2 = np.trapz(gains_optimal, baserate) #Area under optimal curve
    A3 = np.trapz(baserate, baserate) #Area under base rate

    score = (A1-A3)/(A2-A3) #Area opt and brate / Area model and brate

    prediction_to_binary=np.where(y_pred>0.5,1,0)
    Confusion=confusion_matrix(y_true, prediction_to_binary, labels=None, sample_weight=None)
    accuracy = (Confusion[0,0]+Confusion[1,1])/(np.sum(Confusion))
    return score, accuracy

def gains_plot_area(y_true, y_pred,model):
    N=float(len(y_true)) #Number of entries
    index_true = np.argsort(y_true)[::-1]
    index_pred = np.argsort(y_pred)[::-1]

    y_true_sorted_pred = y_true[index_pred] #predict index
    y_true_sorted_true = y_true[index_true] #test index

    gains_optimal = np.cumsum(y_true_sorted_true)/np.sum(y_true_sorted_true)
    gains_model = np.cumsum(y_true_sorted_pred)/np.sum(y_true_sorted_true)
    baserate = np.arange(1, N+1)/N

    A1 = np.trapz(gains_model, baserate) #Area under model curve
    A2 = np.trapz(gains_optimal, baserate) #Area under optimal curve
    A3 = np.trapz(baserate, baserate) #Area under base rate

    score = (A1-A3)/(A2-A3) #Area opt and brate / Area model and brate

    plt.figure()
    plt.style.use("seaborn-whitegrid")
    plt.plot(baserate*100, gains_optimal*100,'--k',label="Optimal",color = '0.4')
    plt.plot(baserate*100, gains_model*100,'k',label="Model")
    plt.plot(baserate*100, baserate*100,'k',label="Baseline",color = '0.75')
    plt.legend(loc=0)
    plt.ylabel("Percentage of positive outcomes [%]")
    plt.xlabel("Percentage of population [%]")
    plt.savefig("../Results/gains_%s.pdf"%model)
    return score

from sklearn.metrics import confusion_matrix
def Confusion_and_accuracy(y_true, y_pred,treshold=0.5):
    prediction_to_binary=np.where(y_pred>treshold,1,0) #make classifications
    Confusion=confusion_matrix(y_true, prediction_to_binary, labels=None, sample_weight=None)
    accuracy = (Confusion[0,0]+Confusion[1,1])/(np.sum(Confusion)) #calculate accuracy score
    return Confusion, accuracy

from sklearn.preprocessing import StandardScaler,  MinMaxScaler


def scale_data_standard(X_train,X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test


def scale_data_minmax(X_train,X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)
    return scaled_X_train, scaled_X_test

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

from sklearn.metrics import r2_score, mean_squared_error
def r2(z1, z2): #true pred
    r2 = r2_score(z1, z2, sample_weight=None, multioutput="uniform_average")
    return r2


def MSE(z1, z2): #true pred
    MSE = mean_squared_error(z1, z2, sample_weight=None, multioutput="uniform_average")
    return MSE


def table_parameters_measurements(filename,accuracy_GD,area_score_GD,max_iters,eta_gd,accuracy_SGD,area_score_SGD,n_epochs, batch_size, eta_sgd,lmd,accuracy_SKL,area_score_SKL,max_iter_SK):
    f= open(filename,"w+")
    f.write("GD - iters: %.2f eta: %.2f\n" %(max_iters,eta_gd))
    f.write("Accuracy: %.2f "%accuracy_GD)
    f.write("Area score: %.2f \n"%area_score_GD)


    f.write("GDS - n_epochs: %.2f, batch size: %.2f, eta: %.2f, lambda: %.2f \n" %(n_epochs, batch_size, eta_sgd,lmd))
    f.write("Accuracy: %.2f "%accuracy_SGD)
    f.write("Area score: %.2f \n"%area_score_SGD)


    f.write("SK - iters: %.2f \n"%max_iter_SK)
    f.write("Accuracy: %.2f "%accuracy_SKL)
    f.write("Area score: %.2f \n"%area_score_SKL)
    f.close()
