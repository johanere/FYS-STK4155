import numpy as np
import matplotlib.pyplot as plt


def cum_gains(Y_predict, Y_test):
    Yf_predict = Y_predict
    Yf_test = Y_test
    idx1 = np.argsort(Yf_predict)[::-1]
    idx2 = np.argsort(Yf_test)[::-1]

    Yf_test1 = Yf_test[idx1]
    Yf_test2 = Yf_test[idx2]

    gains1 = np.cumsum(Yf_test1) / np.sum(Yf_test1)
    percent1 = np.arange(1, len(Yf_test1) + 1) / len(Yf_test1)

    gains2 = np.cumsum(Yf_test2) / np.sum(Yf_test2)
    percent2 = np.arange(1, len(Yf_test2) + 1) / len(Yf_test2)

    A1 = np.trapz(gains1, percent1)
    print(A1)
    A2 = np.trapz(gains2, percent2)
    print(A2)
    A3 = np.trapz(percent1, percent1)
    print(A3)
    score = (A1 - A3) / (A2 - A3)
    print("notm", score)
    # plt.plot(percent1, gains1)
    # plt.plot(percent2, gains2)
    # plt.plot(percent1, percent1)
    # plt.show()
    # exit()

    return score
