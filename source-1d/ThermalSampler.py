import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d
#execfile('atomic.unit')
from units import AtomicUnit
AU = AtomicUnit()

class BoltzmannLight_1D(object):
    """
    Sample Thermal Light
    """
    def __init__(self,beta,num_k=1000):

        # Set up k grid according to beta
        max_k = 10.0/beta
        self.beta = beta
        self.Ks = np.linspace(-max_k,max_k,num=num_k)
        self.Ws = np.abs(self.Ks) #*c
        self.Planck = self.Ws**3/(np.pi**2)/(np.exp(self.beta*self.Ws)-1)
        if len(self.Ws)==1:
            self.dW = 1.0
        else:
            self.dW = np.abs(self.Ws[1]-self.Ws[0])
        # print Ks

        # Boltzmann Distribution
        # probabilities = np.exp(-beta*np.abs(Ks))
        # probabilities = probabilities/np.sum(probabilities)

    def sample(self):
        # Sample X and P
        mean = np.zeros(len(self.Ks))
        cov = np.diag(1.0/(self.beta*np.abs(self.Ks)))
        XPs = np.random.multivariate_normal(mean, cov, 2)

        Xk = XPs[0]
        Pk = XPs[1]

        # Calculate Ek from Xk Pk
        # Eks = np.sqrt(np.abs(self.Ks)/4)*1j*((Xk-Xk[::-1]) + 1j*(Pk+Pk[::-1]))
        Eks = np.sqrt(np.abs(self.Ks)/4)*1j*((Xk-Xk[::-1]) + 1j*(Pk+Pk[::-1])) * self.Planck*self.dW

        self.Eks = Eks
        self.phase = np.random.random()*2*np.pi

        return Xk,Pk


    def useEks(self,Eks):
        self.Eks = Eks

    def getEr(self,Rgrid,time):
        dK = self.Ks[1]-self.Ks[0]
        Egrid = np.zeros(len(Rgrid))
        for K,Ek in zip(self.Ks,self.Eks):
            Egrid =  Egrid+ Ek*np.exp(1j*K*Rgrid) *np.exp(-1j*K*time+1j*self.phase)*dK

        return np.real(Egrid)


class BoltzmannLight_1mode(object):
    """
    Sample Thermal Light
    """
    def __init__(self,beta,KCW):

        # Set up k grid according to beta
        self.KCW = KCW
        self.beta = beta

    def sample_ACW(self):
        mean = 0.0
        std = 1.0/np.sqrt(self.beta)
        self.ACW = np.random.normal(mean, std)

        return self.ACW


    def calculate_ECW(self,Rgrid,time):
        self.ECWx = self.ACW*np.cos(self.KCW*(Rgrid-time))
        self.ECWy = np.zeros(len(Rgrid))
        self.BCWx = np.zeros(len(Rgrid))
        self.BCWy = self.ACW*np.sin(self.KCW*(Rgrid-time))


class BoltzmannLight_Nmode(object):
    """
    Sample Thermal Light
    """
    def __init__(self,beta,KCW,N,Kmax):
        # Set up k grid according to beta
        self.KCW = KCW
        self.beta = beta
        self.N = N
        self.KCWs = np.linspace(-Kmax,Kmax,num=N) + KCW

    def sample_ACW(self):
        mean = 0.0
        std = 1.0/np.sqrt(self.beta)
        self.ACWs = np.random.normal(mean, std, self.N )

        return self.ACWs


    def calculate_ECW(self,Rgrid,time):
        self.ECWx = np.zeros(len(Rgrid))
        # # equally weighted
        # for i in range(self.N):
        #     W = 1.0/self.N
        #     self.ECWx += W*self.ACWs[i]*np.cos(self.KCWs[i]*(Rgrid-time))

        # Planck formula
        Wall = 0.0
        for i in range(self.N):
            W = self.KCWs[i]**3*(np.exp(self.beta*self.KCWs[i])+1.0)
            Wall += W
            self.ECWx += W*self.ACWs[i]*np.cos(self.KCWs[i]*(Rgrid-time))
        self.ECWx = self.ECWx/Wall


class PlanckLight_Nmode(object):
    """
    Sample Thermal Light
    """
    def __init__(self,beta,dK,Kmax):
        # Set up k grid according to beta
        self.beta = beta
        self.dK = dK
        self.KCWs = np.arange(dK,Kmax,dK)
        self.N = len(self.KCWs)


    def sample_ACW(self):
        self.ACWs = np.sqrt(self.KCWs)*(1.0/(np.exp(self.KCWs*self.beta)-1.0)+0.0) *self.dK
        self.phases = np.random.random(self.N)*2*np.pi
        return self.ACWs


    def calculate_ECW(self,Rgrid,time):
        self.ECWx = np.zeros(len(Rgrid))
        for i in range(self.N):
            self.ECWx += self.ACWs[i]*np.cos(self.KCWs[i]*(Rgrid-time)+self.phases[i])
