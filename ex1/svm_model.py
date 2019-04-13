from __future__ import print_function
import numpy as np
from cvxopt import matrix, solvers


def svm_linear(X, y):
    def fit(X, y):
        N = len(y)
        V = X
        Q = matrix(V.dot(V.T))

        p = matrix(-np.ones((2*N), 1))
        # build A,b,G,h
        G = matrix(-np.eye((2*N)))
        h = matrix(np.zeros(2*N, 1))
        A = matrix(y.T)
        b = matrix(np.zeros(1, 1))
        solvers.options['show_progress'] = False
        sol = solvers.qp(Q, p, G, h, A, b)
        l = np.array(sol['x'])  # lambda

        epsilon = 1e-6  # just a small number, greater than 1e-9
        S = np.where(l > epsilon)[0]
        VS = V[:, S]
        XS = X[:, S]
        yS = y[:, S]
        lS = l[S]
        # calculate w and b
        w = VS.dot(lS)
        b = np.mean(yS.T - w.T.dot(XS))
        return w, b
    fit(X, y)

    def coef(w):
        return w

    def bias(b):
        return b
