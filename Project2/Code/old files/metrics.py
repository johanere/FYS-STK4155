import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def gains_plot_area(y_true, y_pred,model):
    N=float(len(y_true)) #Number of entries
    index_true = np.argsort(y_true)[::-1]
    index_pred = np.argsort(y_pred)[::-1]

    y_true_sorted_pred = y_true[index_pred] #predict index
    y_true_sorted_true = y_true[index_true] #test index

    gains_optimal = np.cumsum(y_true_sorted_pred)/np.sum(y_true_sorted_pred)
    gains_model = np.cumsum(y_true_sorted_true)/np.sum(y_true_sorted_true)
    baserate = np.arange(1, N+1)/N

    A1 = np.trapz(gains_optimal, baserate) #Area under optimal curv
    A2 = np.trapz(gains_model, baserate) #Area under optimal curv
    A3 = np.trapz(baserate, baserate) #Area under base rate

    score = (A1-A3)/(A2-A3) #Area opt and brate / Area model and brate


    plt.style.use("seaborn-whitegrid")
    plt.plot(baserate*100, gains_model*100,'--k',label="Optimal",color = '0.4')
    plt.plot(baserate*100, gains_optimal*100,'k',label="Model")
    plt.plot(baserate*100, baserate*100,'k',label="Baseline",color = '0.75')
    plt.legend(loc=0)
    plt.ylabel("Percentage of positive outcomes [%]")
    plt.xlabel("Percentage of population [%]")
    plt.savefig("..\\Results\\gains_%s.pdf"%model)
    plt.show()

    return score

def Confusion_and_accuracy(y_true, y_pred,treshold=0.5):
    prediction_to_binary=np.where(y_pred>treshold,1,0) #make classifications
    Confusion=confusion_matrix(y_true, prediction_to_binary, labels=None, sample_weight=None)
    accuracy = (Confusion[0,0]+Confusion[1,1])/(np.sum(Confusion)) #calculate accuracy score
    return Confusion, accuracy

"""
def Gains_plot(y_true,y_prob):
    index=np.argsort(y_true)[::-1]
    y_pred=np.where(y_prob>0.5,1,0)
    y_true_sorted=y_true[index]
    y_pred_sorted=y_pred[index]

    print(y_true)
    print(y_true_sorted)
    print(y_pred)
    print(y_pred_sorted)

    n_inputs=len(y_true)
    N=n_inputs-10
    wizard=np.zeros(n_inputs)
    model=np.zeros(n_inputs)
    population=np.linspace(0,n_inputs,n_inputs)
    baserate=np.linspace(0,np.sum(y_true_sorted),n_inputs)
    for i in range(10,n_inputs):
        wizard[i]=np.sum(y_true_sorted[:i])
        model[i]=np.sum(y_pred_sorted[:i])
    plt.plot(population,wizard)
    plt.plot(population,baserate)
    plt.plot(population,model)
    plt.show()
"""
