import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d
#execfile('atomic.unit')
from units import AtomicUnit
AU = AtomicUnit()

class ThermalLight_1D(object):
    """
    Sample Thermal Light
    """
    def __init__(self,beta,num_k=1000):

        # Set up k grid according to beta
        max_k = 5.0/beta
        self.beta = beta
        self.Ks = np.linspace(-max_k,max_k,num=num_k)
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
        Eks = np.sqrt(np.abs(self.Ks)/4)*1j*((Xk-Xk[::-1]) + 1j*(Pk+Pk[::-1]))

        self.Eks = Eks

        return Xk,Pk

    def getEr(self,Rgrid,time):
        dK = self.Ks[1]-self.Ks[0]
        Egrid = np.zeros(len(Rgrid))
        for K,Ek in zip(self.Ks,self.Eks):
            Egrid =  Egrid+ Ek*np.exp(1j*K*Rgrid) *np.exp(-1j*K*time)*dK

        return np.real(Egrid)
