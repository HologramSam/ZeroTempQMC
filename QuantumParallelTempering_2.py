from re import A
import numpy as np
import random as r
import matplotlib as mpl
import matplotlib.pyplot as plt

class ZeroTempQMCPT_Ising:
    
    #N int is number of spins
    #L int is number of replicas
    #Gamma float is the transverse field strength
    #Jij NxN numpy array is the coupling matrix
    def __init__(self,N,L,replicas,Gamma_max,Gamma_min,Jij):
        
        self.L = L
        self.Gamma_max = Gamma_max
        self.Gamma_min = Gamma_min
        self.Gammas = [Gamma_min + i*(Gamma_max-Gamma_min)/replicas for i in range(replicas)]
        self.N = N
        self.replicas = replicas
        self.Jij = Jij
        self.Jmax = np.max(np.absolute(Jij)) 
        self.Wmax = [N*self.Gammas[i] + (N**2)*self.Jmax for i in range(replicas)] 
        #self.Spins = [[np.array([1 for i in range(N)]) for i in range(L)] for j in replicas]

        self.Spins = []
        for i in range(replicas):
            randConf = np.array([(-1)**r.randint(0,1) for k in range(N)])
            self.Spins.append([randConf for j in range(L)])
        
    def MCupdateLocal(self,replica):

        #Within replica pick a copy to update
        copyToUpdate = r.randint(0,self.L-1)

        currentConfig = self.Spins[replica][copyToUpdate]

        adjacentConfigP = self.Spins[replica][(copyToUpdate+1)%self.L]
        adjacentConfigM = self.Spins[replica][(copyToUpdate-1)%self.L]



        if (adjacentConfigM == currentConfig).all() and  (currentConfig == adjacentConfigP).all() and (adjacentConfigM == currentConfig).all():

            #choose spin to update:
            spintoUpdate = r.randint(0,N-1)

            #calculate weight of the config
            Wconfig = self.calcWdiag(currentConfig,replica) 
            
            metWeight = self.N*(self.Gammas[replica]**2)/(Wconfig**2)

            monteCarlo = r.uniform(0,1)

            if monteCarlo < metWeight:
                currentConfig[spintoUpdate] = currentConfig[spintoUpdate]*(-1)

        elif ((adjacentConfigM == currentConfig).all() and (currentConfig != adjacentConfigP).all()) or ((adjacentConfigM != currentConfig).all() and (currentConfig.all() == adjacentConfigP)):

            if (adjacentConfigM != currentConfig).all():
                diffIndex = findIndexofDiff1(adjacentConfigM,currentConfig)

                WadjacentM = self.calcWdiag(adjacentConfigM)
                Wconfig = self.calcWdiag(currentConfig)

                metWeight = WadjacentM/Wconfig

                monteCarlo = r.uniform(0,1)

                if monteCarlo < metWeight:
                    currentConfig[diffIndex] = currentConfig[diffIndex]*(-1)

            elif (adjacentConfigP != currentConfig).all():
                diffIndex = findIndexofDiff1(adjacentConfigP,currentConfig)

                WadjacentP = self.calcWdiag(adjacentConfigP,replica)
                Wconfig = self.calcWdiag(currentConfig,replica)

                metWeight = WadjacentP/Wconfig

                monteCarlo = r.uniform(0,1)

                if monteCarlo < metWeight:
                    currentConfig[diffIndex] = currentConfig[diffIndex]*(-1)

        elif (adjacentConfigM == adjacentConfigP).all() and (currentConfig != adjacentConfigM).all():

            WadjConf = self.calcWdiag(adjacentConfigM,replica)

            metWeight = (WadjConf**2)/(self.N*(self.Gammas[replica]**2))

            monteCarlo = r.uniform(0,1)

            if monteCarlo < metWeight:
                currentConfig = adjacentConfigM.copy()

        elif ((adjacentConfigM != currentConfig).all() and (currentConfig != adjacentConfigP).all()) and (adjacentConfigM != adjacentConfigP).all():

            diffM = findIndexofDiff1(adjacentConfigM)
            diffP = findIndexofDiff1(adjacentConfigP)

            currentConfig[diffM] = currentConfig[diffM]*(-1)
            currentConfig[diffP] = currentConfig[diffP]*(-1)

    def MCUpdateSwap(self,replica1,replica2):

        #calculate weights of initial configs
        Pi1R1 = self.CalcTotalWeightofReplica(self.Spins[replica1],replica1)
        Pi2R2 = self.CalcTotalWeightofReplica(self.Spins[replica2],replica2)

        #calculate weight of swapped configs
        Pi1R2 = self.CalcTotalWeightofReplica(self.Spins[replica1],replica2)
        Pi2R1 = self.CalcTotalWeightofReplica(self.Spins[replica2],replica1)

        metWeight = (Pi1R2*Pi2R1)/(Pi1R1*Pi2R2)

        monteCarlo = r.uniform(0,1)

        if monteCarlo < metWeight:
            rep1Copy = self.Spins[replica1].copy()
            rep2Copy = self.Spins[replica2].copy()


            self.Spins[replica1] = rep2Copy
            self.Spins[replica2] = rep1Copy

    def LocalSweep(self):
        for rep in range(self.replicas):
            self.MCupdateLocal(rep)

    def SwapSweep(self):
        for rep in range(self.replicas):
            self.MCUpdateSwap(rep,(rep+1)%self.replicas)

    def fullMonteCarloStep(self):
        for i in range(self.L):
            self.LocalSweep()
        
        self.SwapSweep()



        
    def CalcTotalWeightofReplica(self,copyOfReplica,replicaNum):
        W = 1
        for i in range(self.L):
            if (copyOfReplica[i] == copyOfReplica[(i+1)%self.L]).all():
                W = W*self.calcWdiag(copyOfReplica[i],replicaNum)
            else:
                W = W*self.Gammas[replicaNum]
        
        return W
        

    #spinConfig -1,1**N array
    def calcWdiag(self,spinConfig,replica):
        
        energy = spinConfig@ (self.Jij@spinConfig)
        
        return self.Wmax[replica] - energy

    



def findIndexofDiff1(array1,array2):
    """
    assumes lists only differ in one index and are of same length
    """
    for i in range(len(array1)):
        if array1[i]!=array2[i]:
            return i



#number of spins
N = 2

J = np.zeros((N,N))

J[0][1] = 1
J[1][0] = 1

#create grid ferromagnetic interactions 

# for i in range(N):
#     for j in range(N):
#         if i == j+1  and i>j:
#             J[i][j] = -1
#             J[j][i] = J[i][j]
#         elif j == i%10 and i != j:
#             J[i][j] = -1
#             J[j][i] = J[i][j]


def computeM(config):
    return sum(config)/N

max_gamma = 4
min_gamma = 0
replicas = 25

QPT = ZeroTempQMCPT_Ising(N,10,replicas,max_gamma,min_gamma,J)
print(QPT.Spins[0][0])
print(abs(computeM(QPT.Spins[0][0])))
print(QPT.Gammas[0])
m_vals = [0]
m_av = 0

probs = 0

iterate = 10000
for i in range(iterate):
    #QPT.MCupdateLocal(0)
    QPT.fullMonteCarloStep()
    m_vals.append(abs(computeM(QPT.Spins[0][0])))
    m_av += abs(computeM(QPT.Spins[0][0]))
    a =''.join([str(int((1+j)/2)) for j in QPT.Spins[0][0]]) 
    if a == '00' or a == '11':
        probs+=1

probs =probs/iterate
print(probs)
plt.plot(m_vals)
plt.show()