import random
import time
import math
import numpy as np
import numpy.matlib
from TS_optimisation import (posterior_update, objfunc, sample_from_probs, argmin,
                            normalize, power, objfuncDopt, est)
                            
from parameters import (Nsim, Nall, Ntreats, Nblocks, Nperb, Nsubblocks, Npersubb,
                        allp, allsigma, len_allsigma, len_allp,
                        alla, a, aallsubb,priormu, priorSigma)

###################

# allocate kth treatement arm
def alloKthArm(k,arms_prob,sigma):
    u = np.random.normal(0,math.sqrt(alla[0]))
    return np.random.normal(arms_prob[k]+u,math.sqrt(sigma[k]))

# generate simulated data
def gensumy(nalloc,arms_prob,sigma,Ntreats,Nsubblocks):
    sumy = np.zeros(np.shape(nalloc))
    for i in range(Nsubblocks):
        for j in range(Ntreats):
            for k in range(int(nalloc[i][j])):
                sumy[i,j] += alloKthArm(j,arms_prob,sigma)
    return sumy

# repeat for every scenario of response variances (sigma_0,sigma_1)
for ss in range(len_allsigma):

    sigmatrue = [math.pow(allsigma[0][ss],2.0),math.pow(allsigma[1][ss],2.0)]

# repeat for every scenario of treatment effects (p_0,p_1)
    for ppp in range(len_allp):

        arms_prob = [allp[0][ppp],allp[1][ppp]]
        f = open("TS_N%i_b%i_p%i%i.txt" % (Nall,Nsubblocks,arms_prob[0],arms_prob[1]), "w")

        allvar = [[[0 for ___ in range(Ntreats)] for __ in range(Nsubblocks)] for _ in range(Nsim)]
        S = [[[0 for ___ in range(Ntreats)] for __ in range(Nsubblocks)] for _ in range(Nsim)]
        allS = [[0 for __ in range(Ntreats)] for _ in range(Nsim)]

        allcount = [[[0 for ___ in range(Ntreats)] for __ in range(Nsubblocks)] for _ in range(Nsim)]
        allsumcount=[0 for _ in range(Ntreats)]

        allest = np.zeros(Nsim)
        allsdest = np.zeros(Nsim)
        mse = 0

        # repeat for Nsim simulated datasets
        for n in range(Nsim):

            # asign prior distributions
            mustar = priormu
            Sigmastar = priorSigma

            noptall = np.zeros((Nsubblocks,Ntreats))
            sigma = [0,0] # store the estimates of sigma

            sumcount = [0 for _ in range(Ntreats)]
            for k in range(Ntreats):
                S[n][0][k] = 0

            # initial cohort (0) is equally randomised to the Ntreats treatments
            count = [0 for _ in range(Ntreats)]
            tempS = [0.0 for _ in range(Ntreats)]
            storeS = np.zeros((Nperb,2))
            while ( count[0]<=1 or count[1]<=1 ):
                count = [0.0 for _ in range(Ntreats)]
                tempS = [0.0 for _ in range(Ntreats)]
                for tt in range(Npersubb):
                    arm_choice = np.random.randint(2) # allocate patient tt randomly to one of the treatments with equal probability
                    count[arm_choice] += 1                    # count total number of patients per treatment within cohort 0
                    res = alloKthArm(arm_choice,arms_prob,sigmatrue) # generate observation for patient tt allocated to treatment arm_choice
                    tempS[arm_choice] += res                          # sum up all patient outcomes for treatment arm_choice in order to obtain the mean
                    storeS[tt][0] = arm_choice
                    storeS[tt][1] = res

            # store information for summary statistics
            for k in range(Ntreats):
                allcount[n][0][k] = count[k]
                noptall[0,k] = count[k]
                sumcount[k] += count[k]
                allsumcount[k] += count[k]
                S[n][0][k] = tempS[k]/count[k] # store mean outcome of patient within cohort 0 for each treatment
                allS[n][k] += S[n][0][k]
                sigma[k] = np.var(storeS[np.argwhere(storeS[:Npersubb,0]==k),1]) # estimate the outcome variance based on oberations
                allvar[n][0][k] = sigma[k]

            indices_value = [0 for _ in range(Ntreats)]
            # repeat process for the remaining cohorts within the current block
            for t in range(1, Nsubblocks):

                    # update posterior to incorporate observations from previous cohort
                    ncurp = t*Npersubb
                    mustar,Sigmastar = posterior_update(noptall,allS[n][:],alla[0],sigma,priormu,priorSigma,ncurp,Ntreats,t,mustar,Sigmastar)

                    nopt = np.zeros((1,Ntreats),dtype=np.int) # 1 is because we observe 1 subblock per time
                    for i in range(Npersubb):

                        indices_value = objfunc(t,Nall,Ntreats,Nsubblocks,mustar,Sigmastar)
                        best_indices_value = max(indices_value)
                        candidates = list(filter(lambda x: indices_value[x] == best_indices_value, list(range(Ntreats))))
                        random.shuffle(candidates)
                        nopt[0,candidates[0]] += 1

                    # ensure there are no 0 allocations
                    if( nopt[0,0]==0 ):
                        nopt[0,0]=1
                        nopt[0,1]=nopt[0,1]-1

                    if( nopt[0,1]==0 ):
                        nopt[0,1]=1
                        nopt[0,0]=nopt[0,0]-1

                    noptall[t,] = nopt[0,]

                    # simulate observations for each patient in each treatment
                    count = 0
                    for i in range(Ntreats):
                        for j in range(nopt[0,i]):
                            res = alloKthArm(i,arms_prob,sigmatrue)
                            tempS[i] += res
                            storeS[t*Npersubb+count][0] = i
                            storeS[t*Npersubb+count][1] = res
                            count = count+1

                    # store information for summary statistics
                    for kkk in range(Ntreats):
                        sumcount[kkk] += nopt[0,kkk]
                        allsumcount[kkk] += nopt[0,kkk]
                        allcount[n][t][kkk] = nopt[0,kkk]
                        sigma[kkk] = np.var(storeS[np.argwhere(storeS[:(t+1)*Npersubb,0]==kkk),1])
                        S[n][t][kkk] = tempS[kkk]/sumcount[kkk]
                        allS[n][kkk] += S[n][t][kkk]
                        allvar[n][t][kkk] = sigma[kkk]

            # Power calculation
            # for outcome-oriented allocation that was obtained above
            incr = 1.0
            end = int(30.0/incr)
            pwr = power(noptall,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks,15,incr)
            #print(str(t)+' --- sigma = ('+str(sigma[0])+','+str(sigma[1])+')')

            # for non-adaptive D_A-optimal (Neyman) allocation
            NNeyman = int((Nall/Nsubblocks)*math.sqrt(sigmatrue[0])/(math.sqrt(sigmatrue[0])+math.sqrt(sigmatrue[1])))
            nNeyman = [[NNeyman,Nall/Nsubblocks-NNeyman],[NNeyman,Nall/Nsubblocks-NNeyman],[NNeyman,Nall/Nsubblocks-NNeyman],[NNeyman,Nall/Nsubblocks-NNeyman],[NNeyman,Nall/Nsubblocks-NNeyman],[NNeyman,Nall/Nsubblocks-NNeyman]]
            pwrNeyman = power(nNeyman,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks,15,incr)

            # for balanced allocation
            Nbal = Nall/Ntreats/Nsubblocks
            nbal = [[Nbal,Nbal],[Nbal,Nbal],[Nbal,Nbal],[Nbal,Nbal],[Nbal,Nbal],[Nbal,Nbal]]
            pwrbal = power(nbal,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks,15,incr)

            # Estimation of treatment effect and calculation of mean squared error for TS
            Nsimest = 10000
            tempest = np.zeros(Nsimest)
            for i in range(Nsimest):

                sy = np.zeros(np.shape(noptall))
                sy = gensumy(noptall,arms_prob,sigmatrue,Ntreats,Nsubblocks)
                tempest[i] = est(noptall,sy,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks)
                mse += math.pow(tempest[i]-(allp[1][0]-allp[0][0]),2.0)/(Nsimest*Nsim)

            allest[n] = sum(tempest)/Nsimest
            allsdest[n] = np.std(tempest)

            #if( t==(Nsubblocks-1) ):
                #print(str(t)+' --- sigma = ('+str(sigma[0])+','+str(sigma[1])+')')
            #for tt in range(Nsubblocks):
            #     f.write("%.3f %.3f " % (S[n][tt][0],S[n][tt][1]))
            # for tt in range(Nsubblocks):
            #     f.write("%.3f %.3f " % (allvar[n][tt][0],allvar[n][tt][1]))
            # for tt in range(Nsubblocks):
            #     f.write("%.1f %.1f " % (allcount[n][tt][0],allcount[n][tt][1]))
            # for tt in range(end):
            #     f.write("%.3f " % (pwr[tt]))
            # for tt in range(end):
            #     f.write("%.3f " % (pwrNeyman[tt]))
            # for tt in range(end):
            #     f.write("%.3f " % (pwrbal[tt]))
            # f.write("%.3f" % (fTS))
            # #f.write("%.1f %.1f" % (sumcount[0],sumcount[1]))
            # f.write("\n")
        #print("n = ("+str(allsumcount[0]/Nsim)+" , "+str(allsumcount[1]/Nsim)+")")
        #f.write("%.3f,%.3f,%.1f %.1f" % (sigmatrue[0],sigmatrue[1],allsumcount[0]/Nsim,allsumcount[1]/Nsim))
        #f.write("\n")

        mest = sum(allest)/Nsim
        stdest = sum(allsdest)/Nsim

        #f.write("%.3f %.3f %.3f \n" % (allsigma0[ss],allsigma1[ss],mse))
        #print('('+str(allsigma0[ss])+','+str(allsigma1[ss])+') & '+str(round(mest,2))+' ('+str(round(stdest,2))+')')

        #f.close()
f.close()
