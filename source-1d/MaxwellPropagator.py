import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d
#execfile('atomic.unit')
from units import AtomicUnit
AU = AtomicUnit()

class MaxwellPropagator_1D(object):
    """
    EM field propagator for 1D grid
    """
    def __init__(self, param):
        self.param = param
        self.NZgrid = self.param.NZgrid
        self.Zgrid = self.param.Zgrid
        self.dZ = self.param.dZ
        self._Ex = self.param._Ex
        self._Ey = self.param._Ey
        self._Bx = self.param._Bx
        self._By = self.param._By

        # current flux
        self.Jx = np.zeros(self.NZgrid)
        self.Jy = np.zeros(self.NZgrid)

        # a long vector [Ex,Ey,Bx,By]
        self.EB = np.zeros(4*self.NZgrid)

        # vector potential
        self.Ax = np.zeros(self.NZgrid)
        self.Ay = np.zeros(self.NZgrid)

    def initializeODEsolver(self,EB,T0):
        # EM ode solver
        self.solver = ode(self.f)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(EB, T0)
        self.EB = self.solver.y

    def f(self,t,vec):
        dvdt = np.zeros(4*self.NZgrid)
        dvdt[self._Ex+1:self._Ex+self.NZgrid-1] = -( vec[self._By+2:self._By+self.NZgrid] - vec[self._By:self._By+self.NZgrid-2] )/self.dZ/2 * self.unit.C**2 \
                                                  - self.Jx[1:self.NZgrid-1]/self.unit.E0
        dvdt[self._Ey+1:self._Ey+self.NZgrid-1] =  ( vec[self._Bx+2:self._Bx+self.NZgrid] - vec[self._Bx:self._Bx+self.NZgrid-2] )/self.dZ/2 * self.unit.C**2 \
                                                  - self.Jy[1:self.NZgrid-1]/self.unit.E0
        dvdt[self._Bx+1:self._Bx+self.NZgrid-1] =  ( vec[self._Ey+2:self._Ey+self.NZgrid] - vec[self._Ey:self._Ey+self.NZgrid-2] )/self.dZ/2
        dvdt[self._By+1:self._By+self.NZgrid-1] = -( vec[self._Ex+2:self._Ex+self.NZgrid] - vec[self._Ex:self._Ex+self.NZgrid-2] )/self.dZ/2

        dvdt[self._Ex] = -( vec[self._By+1] - vec[self._By+0] )/self.dZ * self.unit.C**2  - self.Jx[0]/self.unit.E0
        dvdt[self._Ey] =  ( vec[self._Bx+1] - vec[self._Bx+0] )/self.dZ * self.unit.C**2  - self.Jx[0]/self.unit.E0
        dvdt[self._Bx] =  ( vec[self._Ey+1] - vec[self._Ey+0] )/self.dZ
        dvdt[self._By] = -( vec[self._Ex+1] - vec[self._Ex+0] )/self.dZ

        dvdt[self._Ex+self.NZgrid-1] = -( vec[self._By+self.NZgrid-1] - vec[self._By+self.NZgrid-2] )/self.dZ * self.unit.C**2 - self.Jx[self.NZgrid-1]/self.unit.E0
        dvdt[self._Ey+self.NZgrid-1] =  ( vec[self._Bx+self.NZgrid-1] - vec[self._Bx+self.NZgrid-2] )/self.dZ * self.unit.C**2 - self.Jy[self.NZgrid-1]/self.unit.E0
        dvdt[self._Bx+self.NZgrid-1] =  ( vec[self._Ey+self.NZgrid-1] - vec[self._Ey+self.NZgrid-2] )/self.dZ
        dvdt[self._By+self.NZgrid-1] = -( vec[self._Ex+self.NZgrid-1] - vec[self._Ex+self.NZgrid-2] )/self.dZ

        return dvdt

    def update_JxJy(self,Jx,Jy):
        self.Jx = Jx
        self.Jy = Jy

    def update_AxAy(self,dt):
        self.Ax = self.Ax - dt*np.array(self.EB[self._Ex:self._Ex+self.NZgrid])
        self.Ay = self.Ay - dt*np.array(self.EB[self._Ey:self._Ey+self.NZgrid])

        # Just for checking if B = curl A
        self.dAxdZ = np.zeros(self.NZgrid)
        self.dAydZ = np.zeros(self.NZgrid)
        for n in range(1,self.NZgrid-1):
            self.dAxdZ[n] = (self.Ax[n+1]-self.Ax[n-1])/self.dZ/2
            self.dAydZ[n] = (self.Ay[n+1]-self.Ay[n-1])/self.dZ/2
        self.dAxdZ[0] = (self.Ax[1]-self.Ax[0])/self.dZ
        self.dAydZ[0] = (self.Ay[1]-self.Ay[0])/self.dZ
        self.dAxdZ[self.NZgrid-1] = (self.Ax[self.NZgrid-1]-self.Ax[self.NZgrid-2])/self.dZ
        self.dAydZ[self.NZgrid-1] = (self.Ay[self.NZgrid-1]-self.Ay[self.NZgrid-2])/self.dZ


    def propagate(self,dt):
        self.solver.integrate(self.solver.t+dt)
        self.EB = self.solver.y

    def getEnergyDensity(self):
        U = np.zeros(self.NZgrid)
        for n in range(self.NZgrid):
            U[n] = 0.5*AU.E0*( self.EB[self._Ex + n]**2 + self.EB[self._Ey + n]**2 ) \
                 + 0.5/AU.M0*( self.EB[self._Bx + n]**2 + self.EB[self._By + n]**2 )
        return U

    def applyAbsorptionBoundaryCondition(self):
        for iz in range(self.NZgrid):
            Z = np.abs(self.Zgrid[iz])
            S = 1*(Z<self.param.Z0) + \
                (Z<=self.param.Z1 and Z>=self.param.Z0)/(1.0+np.exp(-(self.param.Z0-self.param.Z1)/(self.param.Z0-Z)-(self.param.Z1-self.param.Z0)/(Z-self.param.Z1))) + \
                0*(Z>self.param.Z1)

            self.EB[self._Ex+iz] = self.EB[self._Ex+iz]*S
            self.EB[self._Ey+iz] = self.EB[self._Ey+iz]*S
            self.EB[self._Bx+iz] = self.EB[self._Bx+iz]*S
            self.EB[self._By+iz] = self.EB[self._By+iz]*S
            self.Ax[iz] = self.Ax[iz]*S
            self.Ay[iz] = self.Ay[iz]*S

class SurfaceHoppingMaxwellPropagator_1D(object):
    """
    EM field propagator using surface hopping in 1-D grid (z)
    """
    def __init__(self, param):
        self.param = param
        self.NZgrid = self.param.NZgrid
        self.Zgrid = self.param.Zgrid
        self.dZ = self.param.dZ
        self._Ex = self.param._Ex
        self._Ey = self.param._Ey
        self._Bx = self.param._Bx
        self._By = self.param._By

        # current flux
        self.Jx = np.zeros(self.NZgrid)
        self.Jy = np.zeros(self.NZgrid)

        # a long vector [Ex,Ey,Bx,By]
        self.EB = np.zeros(4*self.NZgrid)

        # vector potential
        self.Ax = np.zeros(self.NZgrid)
        self.Ay = np.zeros(self.NZgrid)

        # unit constats:
        self.unit = AU

        self.dAxdZ = np.zeros(self.NZgrid)
        self.dAydZ = np.zeros(self.NZgrid)

    def initializeODEsolver(self,EB,T0):
        # EM ode solver
        self.solver = ode(self.f)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(EB, T0)
        self.EB = self.solver.y

    def f(self,t,vec):
        dvdt = np.zeros(4*self.NZgrid)
        dvdt[self._Ex+1:self._Ex+self.NZgrid-1] = -( vec[self._By+2:self._By+self.NZgrid] - vec[self._By:self._By+self.NZgrid-2] )/self.dZ/2 * self.unit.C**2 \
                                                  - self.Jx[1:self.NZgrid-1]/self.unit.E0
        dvdt[self._Ey+1:self._Ey+self.NZgrid-1] =  ( vec[self._Bx+2:self._Bx+self.NZgrid] - vec[self._Bx:self._Bx+self.NZgrid-2] )/self.dZ/2 * self.unit.C**2 \
                                                  - self.Jy[1:self.NZgrid-1]/self.unit.E0
        dvdt[self._Bx+1:self._Bx+self.NZgrid-1] =  ( vec[self._Ey+2:self._Ey+self.NZgrid] - vec[self._Ey:self._Ey+self.NZgrid-2] )/self.dZ/2
        dvdt[self._By+1:self._By+self.NZgrid-1] = -( vec[self._Ex+2:self._Ex+self.NZgrid] - vec[self._Ex:self._Ex+self.NZgrid-2] )/self.dZ/2

        dvdt[self._Ex] = -( vec[self._By+1] - vec[self._By+0] )/self.dZ * self.unit.C**2  - self.Jx[0]/self.unit.E0
        dvdt[self._Ey] =  ( vec[self._Bx+1] - vec[self._Bx+0] )/self.dZ * self.unit.C**2  - self.Jx[0]/self.unit.E0
        dvdt[self._Bx] =  ( vec[self._Ey+1] - vec[self._Ey+0] )/self.dZ
        dvdt[self._By] = -( vec[self._Ex+1] - vec[self._Ex+0] )/self.dZ

        dvdt[self._Ex+self.NZgrid-1] = -( vec[self._By+self.NZgrid-1] - vec[self._By+self.NZgrid-2] )/self.dZ * self.unit.C**2 - self.Jx[self.NZgrid-1]/self.unit.E0
        dvdt[self._Ey+self.NZgrid-1] =  ( vec[self._Bx+self.NZgrid-1] - vec[self._Bx+self.NZgrid-2] )/self.dZ * self.unit.C**2 - self.Jy[self.NZgrid-1]/self.unit.E0
        dvdt[self._Bx+self.NZgrid-1] =  ( vec[self._Ey+self.NZgrid-1] - vec[self._Ey+self.NZgrid-2] )/self.dZ
        dvdt[self._By+self.NZgrid-1] = -( vec[self._Ex+self.NZgrid-1] - vec[self._Ex+self.NZgrid-2] )/self.dZ

        return dvdt

    def update_JxJy(self,Jx,Jy):
        self.Jx = Jx
        self.Jy = Jy

    def update_AxAy(self,dt):
		# dAdt = -E
        self.Ax = self.Ax - dt*np.array(self.EB[self._Ex:self._Ex+self.NZgrid])
        self.Ay = self.Ay - dt*np.array(self.EB[self._Ey:self._Ey+self.NZgrid])


        # Just for checking if B = curl A
        for n in range(1,self.NZgrid-1):
            self.dAxdZ[n] = (self.Ax[n+1]-self.Ax[n-1])/self.dZ/2
            self.dAydZ[n] = (self.Ay[n+1]-self.Ay[n-1])/self.dZ/2
        self.dAxdZ[0] = (self.Ax[1]-self.Ax[0])/self.dZ
        self.dAydZ[0] = (self.Ay[1]-self.Ay[0])/self.dZ
        self.dAxdZ[self.NZgrid-1] = (self.Ax[self.NZgrid-1]-self.Ax[self.NZgrid-2])/self.dZ
        self.dAydZ[self.NZgrid-1] = (self.Ay[self.NZgrid-1]-self.Ay[self.NZgrid-2])/self.dZ

    def propagate_free(self,dt):
        fEx = interp1d(self.Zgrid, self.EB[self._Ex:self._Ex + self.NZgrid])
        fEy = interp1d(self.Zgrid, self.EB[self._Ey:self._Ey + self.NZgrid])
        fBx = interp1d(self.Zgrid, self.EB[self._Bx:self._Bx + self.NZgrid])
        fBy = interp1d(self.Zgrid, self.EB[self._By:self._By + self.NZgrid])
        fAx = interp1d(self.Zgrid, self.Ax)
        fAy = interp1d(self.Zgrid, self.Ay)
        for iz in range(self.NZgrid):
            Z = self.Zgrid[iz] - AU.C*dt
            if Z < max(self.Zgrid) and Z>min(self.Zgrid):
                self.EB[self._Ex + iz] = fEx(Z)
                self.EB[self._Ey + iz] = fEy(Z)
                self.EB[self._Bx + iz] = fBx(Z)
                self.EB[self._By + iz] = fBy(Z)
                self.Ax[iz] = fAx(Z)
                self.Ay[iz] = fAy(Z)
            else:
                self.EB[self._Ex + iz] = 0.0
                self.EB[self._Ey + iz] = 0.0
                self.EB[self._Bx + iz] = 0.0
                self.EB[self._By + iz] = 0.0
                self.Ax[iz] = 0.0
                self.Ay[iz] = 0.0

    def propagate(self,dt):
        self.solver.integrate(self.solver.t+dt)
        self.EB = self.solver.y


    def getEnergyDensity(self):
        U = np.zeros(self.NZgrid)
        for n in range(self.NZgrid):
            U[n] = 0.5*AU.E0*( self.EB[self._Ex + n]**2 + self.EB[self._Ey + n]**2 ) \
                 + 0.5/AU.M0*( self.EB[self._Bx + n]**2 + self.EB[self._By + n]**2 )
        return U

    def applyAbsorptionBoundaryCondition(self):
        for iz in range(self.NZgrid):
            Z = np.abs(self.Zgrid[iz])
            S = 1*(Z<self.param.Z0) + \
                (Z<=self.param.Z1 and Z>=self.param.Z0)/(1.0+np.exp(-(self.param.Z0-self.param.Z1)/(self.param.Z0-Z)-(self.param.Z1-self.param.Z0)/(Z-self.param.Z1))) + \
                0*(Z>self.param.Z1)

            self.EB[self._Ex+iz] = self.EB[self._Ex+iz]*S
            self.EB[self._Ey+iz] = self.EB[self._Ey+iz]*S
            self.EB[self._Bx+iz] = self.EB[self._Bx+iz]*S
            self.EB[self._By+iz] = self.EB[self._By+iz]*S
            if True:
            ##try:
                self.Ax[iz] = self.Ax[iz]*S
                self.Ay[iz] = self.Ay[iz]*S
                self.dAxdZ[iz] = self.dAxdZ[iz]*S
                self.dAydZ[iz] = self.dAydZ[iz]*S
            ##except:
                ##pass

class EhrenfestPlusREB_MaxwellPropagator_1D(object):
    """
    EM field propagator using Ehrenfest+REB in 1-D grid (z)
    """
    def __init__(self, param):
        self.param = param
        self.NZgrid = self.param.NZgrid
        self.Zgrid = self.param.Zgrid
        self.dZ = self.param.dZ
        self._Ex = self.param._Ex
        self._Ey = self.param._Ey
        self._Bx = self.param._Bx
        self._By = self.param._By

        # current flux
        self.Jx = np.zeros(self.NZgrid)
        self.Jy = np.zeros(self.NZgrid)

        # a long vector [Ex,Ey,Bx,By]
        self.EB = np.zeros(4*self.NZgrid)

        # vector potential
        self.Ax = np.zeros(self.NZgrid)
        self.Ay = np.zeros(self.NZgrid)

        # unit constats:
        self.unit = AU

        self.dAxdZ = np.zeros(self.NZgrid)
        self.dAydZ = np.zeros(self.NZgrid)

        # total alpha and beta
        self.ALPHA = 0.0
        self.BETA = 0.0

        # save Scattering field out of the box
        self.Es = []
        self.Bs = []
        self.Xs = []

        # absoprtion boundary condition
        self.ABC = np.zeros(4*self.NZgrid)
        self.S = np.zeros(self.NZgrid)
        i_min,i_max = np.argmin(np.abs(self.Zgrid+self.param.Z0)),np.argmin(np.abs(self.Zgrid-self.param.Z0))
        self.i_min,self.i_max = i_min,i_max
        for iz in range(self.NZgrid):
            Z = np.abs(self.Zgrid[iz])
            S = 1*(Z<self.param.Z0) + \
                (Z<=self.param.Z1 and Z>=self.param.Z0)/(1.0+np.exp(-(self.param.Z0-self.param.Z1)/(self.param.Z0-Z)-(self.param.Z1-self.param.Z0)/(Z-self.param.Z1))) + \
                0*(Z>self.param.Z1)
            self.ABC[self._Ex+iz] = S
            self.ABC[self._Ey+iz] = S
            self.ABC[self._Bx+iz] = S
            self.ABC[self._By+iz] = S
            self.S[iz] = S
        #for i in range(i_min+1):
            #print self.S[i],self.S[-1-i]
        #print self.S[:i_min+1]
        #print self.S[i_max:]
        #print self.S[i_min-1],self.S[i_min]
        #print self.S[i_max],self.S[i_max+1]

    def initializeODEsolver(self,EB,T0):
        # EM ode solver
        self.solver = ode(self.f)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(EB, T0)
        self.EB = self.solver.y

    def f(self,t,vec):
        dvdt = np.zeros(4*self.NZgrid)
        dvdt[self._Ex+1:self._Ex+self.NZgrid-1] = -( vec[self._By+2:self._By+self.NZgrid] - vec[self._By:self._By+self.NZgrid-2] )/self.dZ/2 * self.unit.C**2 \
                                                  - self.Jx[1:self.NZgrid-1]/self.unit.E0
        dvdt[self._Ey+1:self._Ey+self.NZgrid-1] =  ( vec[self._Bx+2:self._Bx+self.NZgrid] - vec[self._Bx:self._Bx+self.NZgrid-2] )/self.dZ/2 * self.unit.C**2 \
                                                  - self.Jy[1:self.NZgrid-1]/self.unit.E0
        dvdt[self._Bx+1:self._Bx+self.NZgrid-1] =  ( vec[self._Ey+2:self._Ey+self.NZgrid] - vec[self._Ey:self._Ey+self.NZgrid-2] )/self.dZ/2
        dvdt[self._By+1:self._By+self.NZgrid-1] = -( vec[self._Ex+2:self._Ex+self.NZgrid] - vec[self._Ex:self._Ex+self.NZgrid-2] )/self.dZ/2

        dvdt[self._Ex] = -( vec[self._By+1] - vec[self._By+0] )/self.dZ * self.unit.C**2  - self.Jx[0]/self.unit.E0
        dvdt[self._Ey] =  ( vec[self._Bx+1] - vec[self._Bx+0] )/self.dZ * self.unit.C**2  - self.Jy[0]/self.unit.E0
        dvdt[self._Bx] =  ( vec[self._Ey+1] - vec[self._Ey+0] )/self.dZ
        dvdt[self._By] = -( vec[self._Ex+1] - vec[self._Ex+0] )/self.dZ

        dvdt[self._Ex+self.NZgrid-1] = -( vec[self._By+self.NZgrid-1] - vec[self._By+self.NZgrid-2] )/self.dZ * self.unit.C**2 - self.Jx[self.NZgrid-1]/self.unit.E0
        dvdt[self._Ey+self.NZgrid-1] =  ( vec[self._Bx+self.NZgrid-1] - vec[self._Bx+self.NZgrid-2] )/self.dZ * self.unit.C**2 - self.Jy[self.NZgrid-1]/self.unit.E0
        dvdt[self._Bx+self.NZgrid-1] =  ( vec[self._Ey+self.NZgrid-1] - vec[self._Ey+self.NZgrid-2] )/self.dZ
        dvdt[self._By+self.NZgrid-1] = -( vec[self._Ex+self.NZgrid-1] - vec[self._Ex+self.NZgrid-2] )/self.dZ

        return dvdt

    def update_JxJy(self,Jx,Jy):
        self.Jx = Jx
        self.Jy = Jy

    def update_AxAy(self,dt):
		# dAdt = -E
        self.Ax = self.Ax - dt*np.array(self.EB[self._Ex:self._Ex+self.NZgrid])
        self.Ay = self.Ay - dt*np.array(self.EB[self._Ey:self._Ey+self.NZgrid])


        # Just for checking if B = curl A
        for n in range(1,self.NZgrid-1):
            self.dAxdZ[n] = (self.Ax[n+1]-self.Ax[n-1])/self.dZ/2
            self.dAydZ[n] = (self.Ay[n+1]-self.Ay[n-1])/self.dZ/2
        self.dAxdZ[0] = (self.Ax[1]-self.Ax[0])/self.dZ
        self.dAydZ[0] = (self.Ay[1]-self.Ay[0])/self.dZ
        self.dAxdZ[self.NZgrid-1] = (self.Ax[self.NZgrid-1]-self.Ax[self.NZgrid-2])/self.dZ
        self.dAydZ[self.NZgrid-1] = (self.Ay[self.NZgrid-1]-self.Ay[self.NZgrid-2])/self.dZ

    def propagate_free(self,dt):
        fEx = interp1d(self.Zgrid, self.EB[self._Ex:self._Ex + self.NZgrid])
        fEy = interp1d(self.Zgrid, self.EB[self._Ey:self._Ey + self.NZgrid])
        fBx = interp1d(self.Zgrid, self.EB[self._Bx:self._Bx + self.NZgrid])
        fBy = interp1d(self.Zgrid, self.EB[self._By:self._By + self.NZgrid])
        fAx = interp1d(self.Zgrid, self.Ax)
        fAy = interp1d(self.Zgrid, self.Ay)
        for iz in range(self.NZgrid):
            Z = self.Zgrid[iz] - AU.C*dt
            if Z < max(self.Zgrid) and Z>min(self.Zgrid):
                self.EB[self._Ex + iz] = fEx(Z)
                self.EB[self._Ey + iz] = fEy(Z)
                self.EB[self._Bx + iz] = fBx(Z)
                self.EB[self._By + iz] = fBy(Z)
                self.Ax[iz] = fAx(Z)
                self.Ay[iz] = fAy(Z)
            else:
                self.EB[self._Ex + iz] = 0.0
                self.EB[self._Ey + iz] = 0.0
                self.EB[self._Bx + iz] = 0.0
                self.EB[self._By + iz] = 0.0
                self.Ax[iz] = 0.0
                self.Ay[iz] = 0.0

    def propagate(self,dt):
        self.solver.integrate(self.solver.t+dt)
        self.EB = self.solver.y


    def getEnergyDensity(self):
        U = np.zeros(self.NZgrid)
        for n in range(self.NZgrid):
            U[n] = 0.5*AU.E0*( self.EB[self._Ex + n]**2 + self.EB[self._Ey + n]**2 ) \
                 + 0.5/AU.M0*( self.EB[self._Bx + n]**2 + self.EB[self._By + n]**2 )
        return U

    def applyAbsorptionBoundaryCondition(self):
        #self.EB =  self.EB * self.ABC
        #for iz in range(self.NZgrid):
            #Z = np.abs(self.Zgrid[iz])
            #S = 1*(Z<self.param.Z0) + \
                #(Z<=self.param.Z1 and Z>=self.param.Z0)/(1.0+np.exp(-(self.param.Z0-self.param.Z1)/(self.param.Z0-Z)-(self.param.Z1-self.param.Z0)/(Z-self.param.Z1))) + \
                #0*(Z>self.param.Z1)

            #self.EB[self._Ex+iz] = self.EB[self._Ex+iz]*self.ABC[self._Ex+iz]
            #self.EB[self._Ey+iz] = self.EB[self._Ey+iz]*self.ABC[self._Ey+iz]
            #self.EB[self._Bx+iz] = self.EB[self._Bx+iz]*self.ABC[self._Bx+iz]
            #self.EB[self._By+iz] = self.EB[self._By+iz]*self.ABC[self._By+iz]
        for iz in range(self.i_min+1):
            S = self.S[iz]
            self.EB[self._Ex+iz] *= S
            self.EB[self._Ey+iz] *= S #self.EB[self._Ey+iz]*self.S[iz]
            self.EB[self._Bx+iz] *= S #self.EB[self._Bx+iz]*self.S[iz]
            self.EB[self._By+iz] *= S #self.EB[self._By+iz]*self.S[iz]

            self.EB[self._Ex+self.NZgrid-1-iz] *= S #= self.EB[self._Ex+self.NZgrid-1-iz]*self.S[-1-iz]
            self.EB[self._Ey+self.NZgrid-1-iz] *= S #= self.EB[self._Ey+self.NZgrid-1-iz]*self.S[-1-iz]
            self.EB[self._Bx+self.NZgrid-1-iz] *= S #= self.EB[self._Bx+self.NZgrid-1-iz]*self.S[-1-iz]
            self.EB[self._By+self.NZgrid-1-iz] *= S #= self.EB[self._By+self.NZgrid-1-iz]*self.S[-1-iz]
            #print self.S[i],self.S[-1-i]

    def saveFarField(self,time,Tmax):
        self.Es.append(self.EB[self._Ex+self.i_max])
        self.Bs.append(self.EB[self._By+self.i_max])
        Xs = self.param.Z0 + (Tmax-time)
        self.Xs.append(Xs)

    def getTotalEnergy(self,dt):
        U =( np.sum(np.array(self.Es)**2)+np.sum(np.array(self.Bs)**2) )*dt/2 *2 #*2 is because we only save one side
        for iz in range(self.i_min,self.i_max+1):
            U += (self.EB[self._Ex+iz]**2 + self.EB[self._By+iz]**2)*self.dZ/2
        return U

    def randomEmit_byE(self,deltaE, intDD, Dx):
        absEx = np.sum(self.EB[self._Ex:self._Ex+self.NZgrid]**2)*self.dZ
        if absEx==0.0:
            self.EB[self._Ex:self._Ex+self.NZgrid] = Dx * np.sqrt(deltaE/intDD)
        else:
            self.EB[self._Ex:self._Ex+self.NZgrid] = self.EB[self._Ex:self._Ex+self.NZgrid] * np.sqrt(1+deltaE/absEx)

    def randomEmit_byD(self,deltaE,intDE,intDD,Dx):
        absEx = np.sum(self.EB[self._Ex:self._Ex+self.NZgrid]**2)*self.dZ
        if absEx==0.0:
            self.EB[self._Ex:self._Ex+self.NZgrid] = Dx * np.sqrt(deltaE/intDD)
        else:
            K = ( -2*intDE+np.sqrt(4*intDE**2+4*deltaE*intDD) )/2/intDD
            self.EB[self._Ex:self._Ex+self.NZgrid] = self.EB[self._Ex:self._Ex+self.NZgrid] + K * Dx[:]


    def ContinuousEmission_E(self,deltaE,intDE,intDD,Dx):
        if intDE==0.0:
            self.EB[self._Ex:self._Ex+self.NZgrid] = np.random.choice([1, -1])*Dx * np.sqrt(deltaE/intDD)
        else:
            alphas = [(-intDE + np.sqrt(intDE**2+2*intDD*deltaE) )/intDD, \
                     (-intDE - np.sqrt(intDE**2+2*intDD*deltaE) )/intDD]
            #if np.abs(alphas[0])<np.abs(alphas[1]):
            if np.abs(alphas[0]+self.ALPHA) <np.abs(alphas[1]+self.ALPHA):
                alpha = alphas[0]
            else:
                alpha = alphas[1]
            self.ALPHA += alpha
            #alpha = alphas[0]
            self.EB[self._Ex:self._Ex+self.NZgrid] = self.EB[self._Ex:self._Ex+self.NZgrid] + alpha* Dx[:]


    def ContinuousEmission_B(self,deltaE,intdDdzB,intdDdzdDdz,dDxdz):
        if intdDdzB==0.0:
            self.EB[self._By:self._By+self.NZgrid] = np.random.choice([1, -1])*dDxdz * np.sqrt(deltaE/intdDdzdDdz)
        else:
            betas = [(-intdDdzB + np.sqrt(intdDdzB**2+2*intdDdzdDdz*deltaE) )/intdDdzdDdz, \
                      (-intdDdzB - np.sqrt(intdDdzB**2+2*intdDdzdDdz*deltaE) )/intdDdzdDdz]
            #if np.abs(betas[0])<np.abs(betas[1]):
            if np.abs(betas[0]+self.BETA) <np.abs(betas[1]+self.BETA):
                beta = betas[0]
            else:
                beta = betas[1]
            self.BETA += beta
            #beta = betas[0]
            self.EB[self._By:self._By+self.NZgrid] = self.EB[self._By:self._By+self.NZgrid] + beta* dDxdz[:]

    def update_TETB(self,TEx,TEy,TBx,TBy):
		self.TEx = TEx
		self.TEy = TEy
		self.TBx = TBx
		self.TBy = TBy
		self.TE2 = self.dZ*(np.dot(TEx, TEx) + np.dot(TEy, TEy))
		self.TB2 = self.dZ*(np.dot(TBx, TBx) + np.dot(TBy, TBy))

    def MakeTransition(self,deltaE,UseRandomEB=True):
        """
        Make a quantum transition of the EM field
        """
        if UseRandomEB:
            dU_E = np.random.rand()*deltaE
            dU_B = deltaE - dU_E
        else:
            dU_E = 0.5 * deltaE
            dU_B = 0.5 * deltaE

        # calculate int(E*TE) and int(B*TB)
        intETE = self.dZ*np.dot(self.TEx, np.array(self.EB[self._Ex:self._Ex+self.NZgrid])) \
	           + self.dZ*np.dot(self.TEy, np.array(self.EB[self._Ey:self._Ey+self.NZgrid]))
        intBTB = self.dZ*np.dot(self.TBx, np.array(self.EB[self._Bx:self._Bx+self.NZgrid])) \
        	   + self.dZ*np.dot(self.TBy, np.array(self.EB[self._By:self._By+self.NZgrid]))
        if intETE==0.0:
            alpha = 1.0
            # alpha = np.random.choice([1, -1])
            self.EB[self._Ex:self._Ex+self.NZgrid] = alpha*self.TEx * np.sqrt(2*dU_E/self.TE2)
            # self.EB[self._Ex:self._Ex+self.NZgrid] = 1*self.TEx * np.sqrt(2*dU_E/self.TE2)
        else:
            alphas = [(-intETE - np.sqrt(intETE**2+2*self.TE2*dU_E) )/self.TE2, \
                 	  (-intETE + np.sqrt(intETE**2+2*self.TE2*dU_E) )/self.TE2]
            if np.abs(alphas[0]) <np.abs(alphas[1]):
            # if np.abs(alphas[0]+self.ALPHA) <np.abs(alphas[1]+self.ALPHA):
            # if alphas[0] < 0.0:
            # if np.abs(alphas[0]) <np.abs(alphas[1]):
                alpha = alphas[0]
            else:
                alpha = alphas[1]
            self.ALPHA += alpha
            #alpha = alphas[0]
            self.EB[self._Ex:self._Ex+self.NZgrid] = self.EB[self._Ex:self._Ex+self.NZgrid] + alpha* self.TEx[:]
        if intBTB==0.0:
            beta = 1.0
            # beta = np.random.choice([1, -1])
            self.EB[self._By:self._By+self.NZgrid] = beta*self.TBy * np.sqrt(2*dU_B/self.TB2)
            # self.EB[self._By:self._By+self.NZgrid] = 1*self.TBy * np.sqrt(2*dU_B/self.TB2)
        else:
            betas = [(-intBTB - np.sqrt(intBTB**2+2*self.TB2*dU_B) )/self.TB2, \
                     (-intBTB + np.sqrt(intBTB**2+2*self.TB2*dU_B) )/self.TB2]
            if np.abs(betas[0])<np.abs(betas[1]):
            # if np.abs(betas[0]+self.BETA)<np.abs(betas[1]+self.BETA):
            # if alpha*betas[0] > 0.0: # alpha and beta have the same sign
            # if betas[0] < 0.0:
                beta = betas[0]
            else:
                beta = betas[1]
            self.BETA += beta
            #alpha = alphas[0]
            self.EB[self._By:self._By+self.NZgrid] = self.EB[self._By:self._By+self.NZgrid] + beta* self.TBy[:]

    def MakeTransition_sign(self,deltaE,sign,UseRandomEB=True):
        if UseRandomEB:
            dU_E = np.random.rand()*deltaE
            dU_B = deltaE - dU_E
        else:
            dU_E = 0.5 * deltaE
            dU_B = 0.5 * deltaE

        # # calculate int(E*TE) and int(B*TB)
        # intETE = self.dZ*np.dot(self.TEx, np.array(self.EB[self._Ex:self._Ex+self.NZgrid])) \
	    #        + self.dZ*np.dot(self.TEy, np.array(self.EB[self._Ey:self._Ey+self.NZgrid]))
        # intBTB = self.dZ*np.dot(self.TBx, np.array(self.EB[self._Bx:self._Bx+self.NZgrid])) \
        # 	   + self.dZ*np.dot(self.TBy, np.array(self.EB[self._By:self._By+self.NZgrid]))

        def choose_small(list):
            if np.abs(list[0]) <np.abs(list[1]):
            # if np.abs(alphas[0]+self.ALPHA) <np.abs(alphas[1]+self.ALPHA):
            # if alphas[0] < 0.0:
            # if np.abs(alphas[0]) <np.abs(alphas[1]):
                return list[0]
            else:
                return list[1]
        def choose_sign(list,sign):
            new_list = filter(lambda x: x*sign >0, list)
            if len(new_list)==1:
                return new_list[0]
            else:
                print "something wrong"
                exit()
        # if intETE==0.0 or sign==0.0:
        if sign==0.0:
            # alpha = choose_sign([1, -1],sign)
            alpha = np.random.choice([1, -1])* np.sqrt(2*dU_E/self.TE2)
            self.EB[self._Ex:self._Ex+self.NZgrid] = alpha*self.TEx[:]
        else:
            # maintain energy conservation
            # alphas = [(-intETE - np.sqrt(intETE**2+2*self.TE2*dU_E) )/self.TE2, \
            #      	  (-intETE + np.sqrt(intETE**2+2*self.TE2*dU_E) )/self.TE2]
            # drop the cross term, i.e. dump the energy into a separated EM field
            alphas = [np.sqrt(2*dU_E/self.TE2),-np.sqrt(2*dU_E/self.TE2)]
            alpha = choose_sign(alphas,sign)
            self.EB[self._Ex:self._Ex+self.NZgrid] = self.EB[self._Ex:self._Ex+self.NZgrid] + alpha* self.TEx[:]
            # print 'alpha=',alpha, alphas
        # if intBTB==0.0 or sign==0.0:
        if sign==0.0:
            # beta = choose_sign([1, -1],sign)
            beta = np.random.choice([1, -1])* np.sqrt(2*dU_B/self.TB2)
            self.EB[self._By:self._By+self.NZgrid] = beta*self.TBy[:]
        else:
            # maintain energy conservation
            # betas = [(-intBTB - np.sqrt(intBTB**2+2*self.TB2*dU_B) )/self.TB2, \
            #          (-intBTB + np.sqrt(intBTB**2+2*self.TB2*dU_B) )/self.TB2]
            # drop the cross term, i.e. dump the energy into a separated EM field
            betas = [np.sqrt(2*dU_B/self.TB2),-np.sqrt(2*dU_B/self.TB2)]
            beta = choose_sign(betas,sign)
            self.EB[self._By:self._By+self.NZgrid] = self.EB[self._By:self._By+self.NZgrid] + beta* self.TBy[:]
            # print 'beta=',beta, betas
        return alpha,beta

class EhrenfestPlusRDB_MaxwellPropagator_1D(object):
    """
    EM field propagator using Ehrenfest+RDB in 1-D grid (z)
    """
    def __init__(self, param):
        self.param = param
        self.NZgrid = self.param.NZgrid
        self.Zgrid = self.param.Zgrid
        self.dZ = self.param.dZ
        self._Dx = self.param._Ex
        self._Dy = self.param._Ey
        self._Bx = self.param._Bx
        self._By = self.param._By

        # current flux
        self.curlPx = np.zeros(self.NZgrid)
        self.curlPy = np.zeros(self.NZgrid)

        # a long vector [Dx,Dy,Bx,By]
        self.DB = np.zeros(4*self.NZgrid)

        # vector potential
        self.Ax = np.zeros(self.NZgrid)
        self.Ay = np.zeros(self.NZgrid)

        # unit constats:
        self.unit = AU

        self.dAxdZ = np.zeros(self.NZgrid)
        self.dAydZ = np.zeros(self.NZgrid)


    def initializeODEsolver(self,DB,T0):
        # EM ode solver
        self.solver = ode(self.f)
        self.solver.set_integrator('dopri5')
        self.solver.set_initial_value(DB, T0)
        self.DB = self.solver.y

    def f(self,t,vec):
        dvdt = np.zeros(4*self.NZgrid)
        dvdt[self._Dx+1:self._Dx+self.NZgrid-1] = -( vec[self._By+2:self._By+self.NZgrid] - vec[self._By:self._By+self.NZgrid-2] )/self.dZ/2 / self.unit.M0
        dvdt[self._Dy+1:self._Dy+self.NZgrid-1] =  ( vec[self._Bx+2:self._Bx+self.NZgrid] - vec[self._Bx:self._Bx+self.NZgrid-2] )/self.dZ/2 / self.unit.M0
        dvdt[self._Bx+1:self._Bx+self.NZgrid-1] =  ( vec[self._Dy+2:self._Dy+self.NZgrid] - vec[self._Dy:self._Dy+self.NZgrid-2] )/self.dZ/2 / self.unit.E0\
                                                  + self.curlPx[1:self.NZgrid-1] / self.unit.E0
        dvdt[self._By+1:self._By+self.NZgrid-1] = -( vec[self._Dx+2:self._Dx+self.NZgrid] - vec[self._Dx:self._Dx+self.NZgrid-2] )/self.dZ/2 / self.unit.E0\
                                                  + self.curlPy[1:self.NZgrid-1] / self.unit.E0

        dvdt[self._Dx] = -( vec[self._By+1] - vec[self._By+0] )/self.dZ / self.unit.M0
        dvdt[self._Dy] =  ( vec[self._Bx+1] - vec[self._Bx+0] )/self.dZ / self.unit.M0
        dvdt[self._Bx] =  ( vec[self._Dy+1] - vec[self._Dy+0] )/self.dZ / self.unit.E0 + self.curlPx[0] / self.unit.E0
        dvdt[self._By] = -( vec[self._Dx+1] - vec[self._Dx+0] )/self.dZ / self.unit.E0 + self.curlPy[0] / self.unit.E0

        dvdt[self._Dx+self.NZgrid-1] = -( vec[self._By+self.NZgrid-1] - vec[self._By+self.NZgrid-2] )/self.dZ / self.unit.M0
        dvdt[self._Dy+self.NZgrid-1] =  ( vec[self._Bx+self.NZgrid-1] - vec[self._Bx+self.NZgrid-2] )/self.dZ / self.unit.M0
        dvdt[self._Bx+self.NZgrid-1] =  ( vec[self._Dy+self.NZgrid-1] - vec[self._Dy+self.NZgrid-2] )/self.dZ / self.unit.E0 + self.curlPx[self.NZgrid-1] / self.unit.E0
        dvdt[self._By+self.NZgrid-1] = -( vec[self._Dx+self.NZgrid-1] - vec[self._Dx+self.NZgrid-2] )/self.dZ / self.unit.E0 + self.curlPy[self.NZgrid-1] / self.unit.E0

        return dvdt

    def update_curlP(self,curlPx,curlPy):
        self.curlPx = curlPx
        self.curlPy = curlPy

    def propagate(self,dt):
        self.solver.integrate(self.solver.t+dt)
        self.DB = self.solver.y


    def getEnergyDensity(self,Pxt,Pyt):
        U = np.zeros(self.NZgrid)
        for n in range(self.NZgrid):
            U[n] = 0.5/AU.E0*( (self.DB[self._Dx + n]-Pxt[n])**2 + (self.DB[self._Dy + n]-Pyt[n])**2 ) \
                 + 0.5/AU.M0*( self.DB[self._Bx + n]**2 + self.DB[self._By + n]**2 )
        return U

    def applyAbsorptionBoundaryCondition(self):
        for iz in range(self.NZgrid):
            Z = np.abs(self.Zgrid[iz])
            S = 1*(Z<self.param.Z0) + \
                (Z<=self.param.Z1 and Z>=self.param.Z0)/(1.0+np.exp(-(self.param.Z0-self.param.Z1)/(self.param.Z0-Z)-(self.param.Z1-self.param.Z0)/(Z-self.param.Z1))) + \
                0*(Z>self.param.Z1)

            self.DB[self._Dx+iz] = self.DB[self._Dx+iz]*S
            self.DB[self._Dy+iz] = self.DB[self._Dy+iz]*S
            self.DB[self._Bx+iz] = self.DB[self._Bx+iz]*S
            self.DB[self._By+iz] = self.DB[self._By+iz]*S

    def ContinuousEmission_D(self,deltaE,intPD,intPP,Px):
        if intPD==0.0:
            self.DB[self._Dx:self._Dx+self.NZgrid] = np.random.choice([1, -1])*Px * np.sqrt(deltaE/intPP)
        else:
            Kappa = [(-intPD + np.sqrt(intPD**2+2*intPP*deltaE) )/intPP, \
                     (-intPD - np.sqrt(intPD**2+2*intPP*deltaE) )/intPP]
            if np.abs(Kappa[0])<np.abs(Kappa[1]):
                Kappa = Kappa[0]
            else:
                Kappa = Kappa[1]
            self.DB[self._Dx:self._Dx+self.NZgrid] = self.DB[self._Dx:self._Dx+self.NZgrid] + Kappa* Px[:]


    def ContinuousEmission_B(self,deltaE,intdPdzB,intdPdzdPdz,dPxdz):
        if intdPdzB==0.0:
            self.DB[self._By:self._By+self.NZgrid] = np.random.choice([1, -1])*dPxdz * np.sqrt(deltaE/intdPdzdPdz)
            return [0,0]
        else:
            alphas = [(-intdPdzB + np.sqrt(intdPdzB**2+2*intdPdzdPdz*deltaE) )/intdPdzdPdz, \
                      (-intdPdzB - np.sqrt(intdPdzB**2+2*intdPdzdPdz*deltaE) )/intdPdzdPdz]
            if np.abs(alphas[0])<np.abs(alphas[1]):
                alpha_min = alphas[0]
            else:
                alpha_min = alphas[1]
            alpha = alphas[1]
            self.DB[self._By:self._By+self.NZgrid] = self.DB[self._By:self._By+self.NZgrid] + alpha* dPxdz[:]
            return alphas
