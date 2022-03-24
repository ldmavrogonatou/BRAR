import random
import math
import numpy.matlib
import numpy as np
from scipy.stats import ncx2
from scipy.stats.distributions import chi2
from utils import makeX, makeS, makeU, makeZ

def posterior_update(nalloc,S,a,sigma,priormu,priorSigma,ncur,K,T,mustar,Sigmastar):

    # generate matrices for calculations
    aall = np.ones(T)*a
    X = makeX(ncur,K,T,nalloc)
    Z = makeZ(ncur,K,T,nalloc)
    U = makeU(T,aall)
    Sigma = makeS(ncur,T,K,nalloc,sigma)
    V = np.add(Sigma,np.dot(np.dot(Z,U),Z.T))

    d = np.identity(ncur)
    Vinv = np.linalg.solve(V, d)
    d = np.identity(K)
    Sigmabinv = np.linalg.solve(priorSigma, d)

    M = np.dot(np.dot(X.T,Vinv),X)
    Minv = np.linalg.solve(M,d)

    Sigmastarinv = np.linalg.solve(np.add(Sigmabinv,Minv), d)
    d = np.identity(K)

    Sigmastar = Minv
    mustar = np.dot(Sigmastar,np.add(np.dot(Sigmabinv,priormu),np.dot(S,M)))

    return mustar,Sigmastar

def objfunc(ncur,N,K,T,mustar,Sigmastar):
    """ Objective function for outcome-oriented allocation (optimise with respect to treatment effect)"""
    simulation_time = 100
    max_time = [0] * K
    for _ in range(simulation_time):
        sample = np.random.multivariate_normal(mustar,Sigmastar)
        max_time[argmin(sample)] += 1.0 # smaller treatment effect is better in this case study

    pre_calc_indices_TS = sample_from_probs(normalize(list(map(lambda p: math.pow(p, ncur/(2.0*T)), normalize(max_time))))) # turn counts into probabilities

    return pre_calc_indices_TS

def sample_from_probs(probs):

    assert abs(sum(probs) - 1.0) < 0.000001
    K = len(probs)
    r = random.random()
    indices = [0.0 for _ in range(K)]
    acc = 0.0
    for kk in range(K):
        acc += probs[kk]
        if acc >= r:
            indices[kk] = 1.0
            break

    return indices

def argmin(a):
    return min(list(range(len(a))), key=lambda x: a[x])

def normalize(probs):
    s = sum(probs)
    for p in probs:
        assert p >= 0
    if s == 0:
        s = 1;
    return list(map(lambda x: x/s, probs))

def power(nalloc,a,sigma,N,K,T,p,incr):
    """ Function to evaluate power for given p_0 and p_1 in [0,30]"""
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
        ncp = pow(pc,2)/f*N
        x = chi2.ppf(0.95,1)
        pwr[i] = 1-ncx2.cdf(x, 1, ncp)
        pos = pos+incr

    return pwr

def objfuncDopt(nalloc,a,sigma,priorSigma,ncur,K,T):
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

# BLUE estimator
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
            #print('(%i,%i)=%.5f',i,j,str(rho[i,j]))
    #rho[:] = [x / N for x in n]

    g = np.zeros(T)
    for i in range(T):
        g[i] = 1.0 + a[i]*rho[i,0]/sigma[0] + a[i]*rho[i,1]/sigma[1]

    #print(g)
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
    #print(psest)
    return (psest[1]-psest[0])
