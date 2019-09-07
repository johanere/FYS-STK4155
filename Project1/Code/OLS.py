import numpy as np
import matplotlib.pyplot as plt


class OLS:
    def __init__(self, m=3, n=6, set=0):
        self.m = m
        self.n = n
        self.set = set

    def construct_matr(self):
            self.x= #predictors
            self.y= #reponse
            self.A[:, i] = self.x ** i

    def QR(self):
        q, self.r = np.linalg.qr(self.A)
        self.c=self.q.T.dot(self.y)
        for i in range(0, self.m):
            if abs(self.r[i, i]) < 1.0e-12:
                print("In QR: R is singular. Program terminated.")  # FIKS DETTE  TRENGS DETTE=
                break
        #self.b = np.zeros(self.n)
        #b[n - 1] = self.y[n - 1] / self.q[n, m]


matr = OLS()
matr.construct_matr()
matr.QR()
matr.BWsub()
print(matr.c)

"""
A = construct_matr(n, m, 0)
q, r = np.linalg.qr(A)

print(q)
print(r)
"""
