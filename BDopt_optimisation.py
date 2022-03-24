import random
import math
import numpy.matlib
import numpy as np
from scipy.stats import ncx2
from scipy.stats.distributions import chi2
from utils import makeX, makeS, makeU, makeZ

def objfunc(nalloc,a,sigma,priorSigma,ncur,K,T):
    """ objective function for D_A-optimality """
     X = makeX(ncur,K,T,nalloc)
     Z = makeZ(ncur,K,T,nalloc)
     U = makeU(T,a)
     Sigma = makeS(ncur,T,K,nalloc,sigma)

     V = np.add(Sigma,np.dot(np.dot(Z,U),Z.T))

     d = np.identity(ncur)
     Vinv = np.linalg.solve(V, d)

     M = np.dot(np.dot(X.T,Vinv),X)

     dM = np.identity(K)
     Minv = np.linalg.solve(M, dM)

     A = np.zeros((K,1))
     A[0][0] = -1
     A[1][0] = 1
     AMAT = np.dot(np.dot(A.T,Minv),A)

     detAMAT = np.linalg.det(AMAT)
     f = math.log(detAMAT)

     return f

def opt(N,Ntreat,Nblock,a,sigma,priorSigma):
    """ standard grid search optimisation to minimise the objective function for D_A-optimality (objfunc) """
    if( Nblock==1 ):
        fmin = 100
        nmin = np.zeros((Nblock,Ntreat))
        # perform search over the design space D={[1,2,...]x[1,2,...];i0+i1=N}
        for i0 in range(1,N):
            for i1 in range(1,N):
                        if( i0+i1==N ):
                            n = [[i0,i1]]
                            f = objfunc(n,a,sigma,priorSigma,N,Ntreat,Nblock)
                            if( f<fmin ):
                                fmin = f
                                nmin = np.copy(n)

    if( Nblock==2 ):
        fmin = 100
        nmin = np.zeros((Nblock,Ntreat))
        # perform search over the design space
        for i0 in range(1,N):
            for i1 in range(1,N):
                for i2 in range(1,N):
                    for i3 in range(1,N):
                        if( i0+i1+i2+i3==N ):
                            n = [[i0,i1],[i2,i3]]
                            f = objfunc(n,a,sigma,priorSigma,N,Ntreat,Nblock)
                            if( f<fmin ):
                                fmin = f
                                nmin = np.copy(n)

    if( Nblock==3 ):
        fmin = 100
        nmin = np.zeros((Nblock,Ntreat))
        # perform search over the design space
        for i0 in range(1,N):
            for i1 in range(1,N):
                for i2 in range(1,N):
                    for i3 in range(1,N):
                        for i4 in range(1,N):
                            for i5 in range(1,N):
                                if( i0+i1+i2+i3+i4+i5==N ):
                                    n = [[i0,i1],[i2,i3],[i4,i5]]
                                    f = objfunc(n,a,sigma,priorSigma,N,Ntreat,Nblock)
                                    if( f<fmin ):
                                        fmin = f
                                        nmin = np.copy(n)

    return nmin

def power(nalloc,a,sigma,N,K,T,p,incr):
    """ calculate power for certain p_0 and p_1 in [0,30] """
    end = int(30.0/incr)
    pwr = np.empty(end)
    pos = 0

    X = makeX(N,K,T,nalloc)
    Z = makeZ(N,K,T,nalloc)
    U = makeU(T,a)
    Sigma = makeS(N,T,K,nalloc,sigma)
    V = np.add(Sigma,np.dot(np.dot(Z,U),Z.T))

    d = np.identity(N)
    Vinv = np.linalg.solve(V, d)

    M = np.dot(np.dot(X.T,Vinv),X)

    dM = np.identity(K)
    Minv = np.linalg.solve(M, dM)

    A = np.zeros((K,1))
    A[0][0] = -1
    A[1][0] = 1
    AMAT = np.dot(np.dot(A.T,Minv),A)

    f = np.linalg.det(AMAT)

    for i in range(end):
        pc = p-pos
        ncp = pow(pc,2)/f # *N
        x = chi2.ppf(0.95,1)
        pwr[i] = 1-ncx2.cdf(x, 1, ncp)
        pos = pos+incr

    return pwr

def powerind(nalloc,a,sigma,N,K,T,p0,p1):
    """ calculate power for certain p_0 and p_1 """
    X = makeX(N,K,T,nalloc)
    Z = makeZ(N,K,T,nalloc)
    U = makeU(T,a)
    Sigma = makeS(N,T,K,nalloc,sigma)

    V = np.add(Sigma,np.dot(np.dot(Z,U),Z.T))

    d = np.identity(N)
    Vinv = np.linalg.solve(V, d)

    M = np.dot(np.dot(X.T,Vinv),X)

    dM = np.identity(K)
    Minv = np.linalg.solve(M, dM)

    A = np.zeros((K,1))
    A[0][0] = -1
    A[1][0] = 1
    AMAT = np.dot(np.dot(A.T,Minv),A)

    f = np.linalg.det(AMAT)

    pc = p1-p0
    ncp = pow(pc,2)/f*N
    x = chi2.ppf(0.95,1)
    pwr = 1-ncx2.cdf(x, 1, ncp)

    return pwr

# BLUE estimator of treatment effects for Ntreats=2
def est(n,sumy,a,sigma,N,K,T):

    X = makeX(N,K,T,n)
    Z = makeZ(N,K,T,n)
    U = makeU(T,a)
    Sigma = makeS(N,T,K,n,sigma)
    V = np.add(Sigma,np.dot(np.dot(Z,U),Z.T))
    d = np.identity(N)
    Vinv = np.linalg.solve(V, d)
    M = np.dot(np.dot(X.T,Vinv),X)
    detM = np.linalg.det(M)

    dt11 = M[0,0]
    dt12 = M[0,1]
    dt21 = M[1,0]
    dt22 = M[1,1]

    rho = np.zeros((T,K))
    for i in range(T):
        for j in range(K):
            rho[i,j] = float(n[i][j])

    g = np.zeros(T)
    for i in range(T):
        g[i] = 1.0 + a[i]*rho[i,0]/sigma[0] + a[i]*rho[i,1]/sigma[1]

    dt11 = 0
    dt22 = 0
    dt12 = 0
    for i in range(T):
        dt11 += rho[i,0]/sigma[0]-(a[i]*pow(rho[i,0],2.0))/(pow(sigma[0],2.0)*g[i])
        dt22 += rho[i,1]/sigma[1]-(a[i]*pow(rho[i,1],2.0))/(pow(sigma[1],2.0)*g[i])
        dt12 -= rho[i,0]*rho[i,1]*a[i]/(g[i]*sigma[0]*sigma[1])
    dt21 = dt12;
    detM = dt11*dt22-dt12*dt21

    psest = np.zeros(K)
    for i in range(T):
        psest[0] += ((dt22/sigma[0] - dt22*rho[i,0]*a[i]/(g[i]*pow(sigma[0],2.0)) + dt12*rho[i,1]*a[i]/(g[i]*sigma[0]*sigma[1]))*sumy[i][0] + (-dt22*rho[i,0]*a[i]/(g[i]*sigma[0]*sigma[1]) - dt12/sigma[1] + dt12*rho[i,1]*a[i]/(g[i]*pow(sigma[1],2.0)))*sumy[i][1])/(detM)
        psest[1] += ((-dt21/sigma[0] + dt21*rho[i,0]*a[i]/(g[i]*pow(sigma[0],2.0)) - dt11*rho[i,1]*a[i]/(g[i]*sigma[0]*sigma[1]))*sumy[i][0] + (dt21*rho[i,0]*a[i]/(g[i]*sigma[0]*sigma[1]) + dt11/sigma[1] - dt11*rho[i,1]*a[i]/(g[i]*pow(sigma[1],2.0)))*sumy[i][1])/(detM)

    return (psest[1]-psest[0])
