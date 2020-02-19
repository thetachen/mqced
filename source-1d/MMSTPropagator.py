import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d
#execfile('atomic.unit')
from units import AtomicUnit
AU = AtomicUnit()

class MMSTlight_Nmode(object):
    """
    MMST Light with N mode in 1D
    """
    def __init__(self,W0,dW,NWmax):
        # Set up omega grid
        self.W0 = W0
        self.dW = dW
        self.Ws = np.arange(W0-NWmax*dW,W0+NWmax*dW,dW)
        if NWmax==0:
            self.Ws = np.array([W0])
        self.NW = len(self.Ws)

        # Set up a long vector [X,P]
        self.XP = np.zeros(2*self.NW)

        # Set up current
        self.Current = np.zeros(self.NW)

        # Set up dammping
        self.Damp = np.zeros(self.NW)

        # Set up auxiliary bath
        self.Action = 0.0
        self.ActionPerMode = np.zeros(self.NW)
        self.Temperature = 0.0
        self.TemperaturePerMode = np.zeros(self.NW)
        self.RandomForce = np.zeros(self.NW)

        # Set up energy per mode
        self.EnergyPerMode = ( (self.Ws**2*self.XP[:self.NW]**2)+(self.XP[:self.NW]**2) )/2
        self.EnergyPerMode0 = self.EnergyPerMode

    def updateCurrent(self,Current):
        self.Current = Current

    def updateDamp(self, KFGR, rho11):
        width = KFGR #*(1.0-rho11)
        self.Damp = width*self.W0*self.W0/((self.Ws-self.W0)**2+width**2) * np.abs(rho11) * self.dW /5

    def resetDamp(self):
        self.Damp = np.zeros(self.NW)

    def updateRandomForce(self,EnergyChange,dt):

        # Option #1: Use the same action for all temperature per mode
        self.Action = self.Action +  EnergyChange*dt
        self.TemperaturePerMode = self.Ws*np.exp(-self.Action)
        self.RandomForce = np.random.normal(0.0, np.sqrt(np.abs(2*self.Damp*self.TemperaturePerMode/dt)))

        # #Option #2
        # self.EnergyPerMode = ( (self.Ws**2*self.XP[:self.NW]**2)+(self.XP[:self.NW]**2) )/2
        #
        # # Integrate the Action function
        # """
        # # NOTE: I scale the action down to make the effect longer ...
        # """
        # self.ActionPerMode = self.ActionPerMode + (self.EnergyPerMode - self.EnergyPerMode0)*dt /100
        # # self.ActionPerMode = self.ActionPerMode + (self.EnergyPerMode - self.EnergyPerMode0)*dt
        # # self.ActionPerMode = (self.EnergyPerMode)
        # # The temperatures
        # self.TemperaturePerMode = self.Ws *np.exp(-self.ActionPerMode)
        #
        # # Choose the random force from the temperature
        # """
        # # NOTE: I scale the STD up to push the system more
        # """
        # # self.RandomForce = np.random.normal(0.0, np.sqrt(np.abs(2*self.Damp*self.TemperaturePerMode/dt*50)))
        # self.RandomForce = np.random.normal(0.0, np.sqrt(np.abs(2*self.Damp*self.TemperaturePerMode/dt)))

    def resetRandomForce(self):
        self.RandomForce = np.zeros(self.NW)

    def initializeODEsolver(self,XP0,T0):
        # XP ode solver
        self.solver = ode(self.f)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(XP0, T0)
        self.XP = self.solver.y


    def f(self,t,vec):
        """
        dvdt function with variable media velocity
        """
        def ifzero(x):
            for i in range(len(x)):
                if x[i]==0.0:  x[i]=1.0
            return x

        dvdt = np.zeros(2*self.NW)
        # dxdt
        """
        # NOTE: The additional Damp and RandomForce to the dxdt term is added to ensure the phase of (x,p) "steady".
        This is suggested by Joe's email on 1/28/2020 ...

        # NOTE: I put abs to the random force to make it more stable??
        """
        dvdt[:self.NW] = vec[self.NW:] + self.Current - vec[:self.NW] * self.Damp + (self.RandomForce) #* ifzero(vec[:self.NW])
        # dPdt
        dvdt[self.NW:] = -(self.Ws**2)*vec[:self.NW] - vec[self.NW:] * self.Damp + (self.RandomForce) #* ifzero(vec[self.NW:])

        return dvdt


    def propagate(self,dt):
        self.t_backstep = self.solver.t
        self.XP_backstep = self.solver.y
        self.solver.integrate(self.solver.t+dt)
        self.XP = self.solver.y

    def resetSolver(self):
        self.solver.set_initial_value(self.XP_backstep, self.t_backstep)

    def getEnergy(self):
        return 0.5*( np.sum( self.Ws**2*self.XP[:self.NW]**2 )+np.sum(self.XP[self.NW:]**2) )* self.dW


    def getEnergyDistribution(self):
        return 0.5*( self.Ws**2*self.XP[:self.NW]**2 + self.XP[self.NW:]**2 )
