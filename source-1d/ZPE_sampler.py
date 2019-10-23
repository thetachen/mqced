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
    def __init__(self,param):
        self.param = param

        # Sample the photon modes
        self.dW = np.pi/self.param.boxsize
        if self.param.Wmin == 0.0: self.param.Wmin=self.dW
        self.Ws = np.arange(self.param.Wmin,self.param.Wmax,self.dW)
        self.X0s = np.zeros(len(self.Ws))
        self.P0s = np.zeros(len(self.Ws))
        self.Amps = np.zeros(len(self.Ws))
        self.Phis = np.zeros(len(self.Ws))

        for j in range(len(self.Ws)):
            self.X0s[j] = np.random.normal(0.0, np.sqrt(0.5/self.Ws[j]))
            self.P0s[j] = np.random.normal(0.0, np.sqrt(0.5*self.Ws[j]))
            self.Amps[j] = np.sqrt(self.X0s[j]**2+self.P0s[j]**2)
            self.Phis[j] = math.atan2(self.X0s[j],self.P0s[j])

    def getFields(self,t,r):
        Xt = self.Amps*np.sin(self.Ws*t + self.Phis)
        Pt = self.Amps*np.cos(self.Ws*t + self.Phis)
        Etr = np.sqrt(2.0/self.param.boxsize) * np.sum(self.Ws*Xt*np.sin(self.Ws*r))# / len(self.Ws)
        Btr = np.sqrt(2.0/self.param.boxsize) * np.sum(Pt*np.cos(self.Ws*r))# / len(self.Ws)

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
    def __init__(self,param):
        self.param = param

        # Sample the photon modes
        self.dW = np.pi/self.param.boxsize
        if self.param.Wmin == 0.0: self.param.Wmin=self.dW
        self.Ws = np.arange(self.param.Wmin,self.param.Wmax,self.dW)
        self.num_modes = len(self.Ws)


        self.As = np.sqrt(self.Ws*0.5*self.dW)*self.Ws
        #self.ACWs = np.sqrt(self.KCWs*(1.0/(np.exp(self.KCWs*self.beta)-1.0)+0.0)*0.0001)*self.KCWs
        self.As = self.As*np.sqrt(2.0/3.0)/np.pi
        self.phases = np.random.random(self.num_modes)*2*np.pi

    def getFields(self,t,r):
        Etr = 0.0
        for i in range(self.num_modes):
            Etr += self.As[i]*np.cos(self.Ws[i]*(r-t)+self.phases[i])
        Btr = Etr
        return Etr,Btr
