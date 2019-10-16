import numpy as np
import math
from scipy.integrate import ode
from scipy.interpolate import interp1d
#execfile('atomic.unit')
from units import AtomicUnit
AU = AtomicUnit()

class ZeroPointEnergy_1D(object):
    """
    Sample Zero Point Energy
    """
    def __init__(self,param,boxsize=2*np.pi,Wmax=200.0,num_mode=400):
        self.param = param
        self.boxsize = boxsize
        self.NZgrid = self.param.NZgrid
        self.Zgrid = self.param.Zgrid
        self.Ezpe = np.zeros(self.NZgrid)
        self.Bzpe = np.zeros(self.NZgrid)

        # Sample the photon modes
        dW = Wmax/num_mode
        self.Ws = np.linspace(0.0, Wmax, num=num_mode, endpoint=False) + dW
        self.X0s = np.zeros(num_mode)
        self.P0s = np.zeros(num_mode)
        self.Amps = np.zeros(num_mode)
        self.Phis = np.zeros(num_mode)
        self.phase = np.zeros(num_mode)

        for j in range(num_mode):
            self.X0s[j] = np.random.normal(0.0, np.sqrt(0.5/self.Ws[j]))
            self.P0s[j] = np.random.normal(0.0, np.sqrt(0.5*self.Ws[j]))
            self.Amps[j] = np.sqrt(self.X0s[j]**2+self.P0s[j]**2)
            self.Phis[j] = math.atan2(self.X0s[j],self.P0s[j])

    def getFields(self,t,r):
        Xt = self.Amps*np.sin(self.Ws*t + self.Phis)
        Pt = self.Amps*np.cos(self.Ws*t + self.Phis)
        Etr = np.sqrt(2.0/self.boxsize) * np.sum(self.Ws*Xt*np.sin(self.Ws*r))# / len(self.Ws)
        Btr = np.sqrt(2.0/self.boxsize) * np.sum(Pt*np.cos(self.Ws*r))# / len(self.Ws)

        return Etr,Btr

    def getFields_range(self,t,rs):
        Etrs = np.zeros_like(rs)
        Btrs = np.zeros_like(rs)
        for ir in range(len(rs)):
            Etrs[ir], Btrs[ir] = self.getFields(t,rs[ir])

        return Etrs,Btrs


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
    def __init__(self,beta,K0,dK,Kmax):
        # Set up k grid according to beta
        self.beta = beta
        self.dK = dK
        #self.KCWs = np.arange(dK,Kmax,dK)
        self.KCWs = np.arange(K0-Kmax*dK,K0+Kmax*dK,dK)
        if Kmax==0:
            self.KCWs = np.array([K0])
        self.N = len(self.KCWs)


    def sample_ACW(self):
        self.ACWs = np.sqrt(self.KCWs*(1.0/(np.exp(self.KCWs*self.beta)-1.0)+0.0)*self.dK)*self.KCWs
        #self.ACWs = np.sqrt(self.KCWs*(1.0/(np.exp(self.KCWs*self.beta)-1.0)+0.0)*0.0001)*self.KCWs
        self.ACWs = self.ACWs*np.sqrt(2.0/3.0)/np.pi
        #self.ACWs = self.ACWs/4.0
        self.phases = np.random.random(self.N)*2*np.pi
        #self.phases2 = np.random.random(self.N)*2*np.pi
        return self.ACWs


    def calculate_ECW(self,Rgrid,time):
        self.ECWx = np.zeros(len(Rgrid))
        for i in range(self.N):
            self.ECWx += self.ACWs[i]*np.cos(self.KCWs[i]*(Rgrid-time)+self.phases[i])
            #self.ECWx += self.ACWs[i]*np.cos(self.KCWs[i]*(Rgrid-time)+self.phases[i])*np.cos(self.phases2[i])
