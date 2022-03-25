import random
import time
import math
import numpy as np
import numpy.matlib
from BDopt_optimisation import *

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
        f = open("Dopt_N%i_b%i_p%i%i.txt_sigma%i%i" % (Nall,Nsubblocks,arms_prob[0],arms_prob[1],100*math.sqrt(sigmatrue[0]),100*math.sqrt(sigmatrue[1])), "w")

        allvar = [[[0 for ___ in range(Ntreats)] for __ in range(Nsubblocks)] for _ in range(Nsim)]
        S = [[[0 for ___ in range(Ntreats)] for __ in range(Nsubblocks)] for _ in range(Nsim)]
        allS = [[0 for __ in range(Ntreats)] for _ in range(Nsim)]

        allcount = [[[0 for ___ in range(Ntreats)] for __ in range(Nsubblocks)] for _ in range(Nsim)]
        allsumcount=[0 for _ in range(Ntreats)]

        sumy = [[[0 for ___ in range(Ntreats)] for __ in range(Nsubblocks)] for _ in range(Nsim)]

        allest = np.zeros(Nsim)
        allestN = np.zeros(Nsim)
        allestb = np.zeros(Nsim)
        allsdest = np.zeros(Nsim)

        mse = 0
        mseN = 0
        mseb = 0

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
                S[n][0][k] = tempS[k]/count[k]
                sumy[n][0][k] = tempS[k]
                sigma[k] = np.var(storeS[np.argwhere(storeS[:Npersubb,0]==k),1]) # calculate the outcome variance based on observations
                allvar[n][0][k] = sigma[k]

            indices_value = [0 for _ in range(Ntreats)]
            # repeat process for the remaining cohorts within the current block
            for t in range(1, Nsubblocks):

                    ncurp = t*Npersubb
                    nopt = np.zeros((1,Ntreats)) ## 1 is because we observe 1 subblock per time

                    # optimisation to find the D_A-optimal allocation
                    nopt = opt(Npersubb,Ntreats,1,a,sigma,priorSigma)
                    noptall[t,] = nopt[0,]

                    count = [0.0 for _ in range(Ntreats)]
                    tempS = [0.0 for _ in range(Ntreats)]
                    count = 0
                    # simulate observations for each patient in each treatment
                    for i in range(Ntreats):
                        for j in range(nopt[0,i]):
                            res = alloKthArm(i,arms_prob,sigmatrue)
                            tempS[i] += res
                            storeS[t*Npersubb+count][0] = i
                            storeS[t*Npersubb+count][1] = res
                            count = count+1

                    for kkk in range(Ntreats):
                        sumcount[kkk] += nopt[0,kkk]
                        allsumcount[kkk] += nopt[0,kkk]
                        allcount[n][t][kkk] = nopt[0,kkk]
                        sigma[kkk] = np.var(storeS[np.argwhere(storeS[:(t+1)*Npersubb,0]==kkk),1]) # calculate the outcome variance based on observations
                        S[n][t][kkk] = tempS[kkk]/nopt[0,kkk]
                        sumy[n][t][kkk] = tempS[kkk]
                        allvar[n][t][kkk] = sigma[kkk]

            # Calculate power
            incr = 1.0
            end = int(30.0/incr)
            # for response-adaptive D_A-optimal allocation
            pwr = power(noptall,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks,15,incr)

            # for non-adaptive D_A-optimal (Neyman) allocation
            NNeyman = int((Nall/Nsubblocks)*math.sqrt(sigmatrue[0])/(math.sqrt(sigmatrue[0])+math.sqrt(sigmatrue[1])))
            nNeyman = [[NNeyman,Nall/Nsubblocks-NNeyman],[NNeyman,Nall/Nsubblocks-NNeyman],[NNeyman,Nall/Nsubblocks-NNeyman],[NNeyman,Nall/Nsubblocks-NNeyman],[NNeyman,Nall/Nsubblocks-NNeyman],[NNeyman,Nall/Nsubblocks-NNeyman]]
            pwrNeyman = power(nNeyman,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks,15,incr)

            # for balanced allocation
            Nbal = Nall/Ntreats/Nsubblocks
            nbal = [[Nbal,Nbal],[Nbal,Nbal],[Nbal,Nbal],[Nbal,Nbal],[Nbal,Nbal],[Nbal,Nbal]]
            pwrbal = power(nbal,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks,15,incr)

            # Estimation of treatment effect and calculation of mean squared error
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
            # for tt in range(Nsubblocks):
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
            # f.write("%.3f %.3f %.3f" % (fBDopt,fNeyman,fbal))
            # #f.write("%.1f %.1f" % (sumcount[0],sumcount[1]))
            # f.write("\n")

        tempestN = np.zeros(Nsimest)
        tempestb = np.zeros(Nsimest)
        for i in range(Nsimest):

            sy = gensumy(nNeyman,arms_prob,sigmatrue,Ntreats,Nsubblocks)
            tempestN[i] = est(nNeyman,sy,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks)
            mseN += math.pow(tempestN[i]-(allp2[0]-allp[0]),2.0)/Nsimest

            sy = gensumy(nbal,arms_prob,sigmatrue,Ntreats,Nsubblocks)
            tempestb[i] = est(nbal,sy,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks)
            mseb += math.pow(tempestb[i]-(allp2[0]-allp[0]),2.0)/Nsimest

        mest = sum(allest)/Nsim
        stdest = sum(allsdest)/Nsim
        mestN = sum(tempestN)/Nsimest
        stdestN = np.std(tempestN)
        mestb = sum(tempestb)/Nsimest
        stdestb = np.std(tempestb)

        # for individual power calculations
        #pwr = powerind(noptall,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks,5,15)
        #pwrN = powerind(nNeyman,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks,5,15)
        #pwrb = powerind(nbal,aallsubb,sigmatrue,Nall,Ntreats,Nsubblocks,5,15)

        #f.write("%.3f %.3f %.3f %.3f %.3f %.3f" % (mest,stdest,mestN,stdestN,mestb,stdestb))
        f.write("%.3f %.3f %.3f %.3f %.3f \n" % (allsigma0[ss],allsigma1[ss],mse,mseN,mseb))

        print('('+str(allsigma0[ss])+','+str(allsigma1[ss])+') & '+str(round(mest,2))+' ('+str(round(stdest,2))+')'+'& '+str(round(mestN,2))+' ('+str(round(stdestN,2))+')'+'& '+str(round(mestb,2))+' ('+str(round(stdestb,2))+')'+'& '+str(pwr)+'& '+str(pwrN)+'& '+str(pwrb))
        #print("n = ("+str(allsumcount[0]/Nsim)+" , "+str(allsumcount[1]/Nsim)+")")
        #f.write("%.3f,%.3f,%.1f %.1f" % (sigmatrue[0],sigmatrue[1],allsumcount[0]/Nsim,allsumcount[1]/Nsim))
        #f.write("\n")

        #f.close()
f.close()
