import random
import time
import math
import numpy as np

def makeX(N,Ntreat,Nblock,n):

    X = np.zeros((N,Ntreat), dtype=int)
    nnprev = 0
    for i in range(Nblock):
        for j in range(Ntreat):
            nn = int(n[i][j])
            for jj in range(Ntreat):
                if (j==jj):
                    X[nnprev:(nnprev+nn),jj] = np.ones(nn)
                else:
                    X[nnprev:(nnprev+nn),jj] = np.zeros(nn)
            nnprev = nnprev+nn

    return X

def makeZ(N,Ntreat,Nblock,n):

    Z = np.zeros((N,Nblock), dtype=int)
    nnprev = 0
    for i in range(Nblock):
        nn = 0
        for j in range(Ntreat):
            nn += int(n[i][j])
        for ii in range(Nblock):
            if (i==ii):
                Z[nnprev:(nnprev+nn),ii] = np.ones(nn)
            else:
                Z[nnprev:(nnprev+nn),ii] = np.zeros(nn)
        nnprev = nnprev+nn

    return Z

def makeU(Nblock,a):
    return np.diag(a)

def makeS(N,Nblock,Ntreat,n,sigma):

    S = np.zeros((N,N))
    nnprev = 0
    for i in range(Nblock):
        for j in range(Ntreat):
            nn = int(n[i][j])
            S[nnprev:(nnprev+nn),nnprev:(nnprev+nn)] = np.diag(np.ones(nn)*sigma[j])
            nnprev += nn

    return S
