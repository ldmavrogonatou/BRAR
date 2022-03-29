""" These parameters are used for both simulations """
import numpy as np

Nsim = 500

Nall = 24
Ntreats = 2
Nblocks = 1
Nperb = int(Nall/Nblocks)
Nsubblocks = 6
Npersubb = int(Nperb/Nsubblocks)

allp = [[5],[15]]
allsigma = [[16,16,16,5.29,5.29,3.43],[1.61,3.43,5.29,1.61,3.43,1.61]]

# finds internal length of nested lists
len_allsigma = len(allsigma[0])
len_allp = len(allp[0])

alla = [1]
a = [alla[0]]
aallsubb = np.ones(Nsubblocks)*alla[0]

priormu = [0,0]
priorSigma = np.diag(np.ones(Ntreats)*10)
