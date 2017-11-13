import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d
#execfile('atomic.unit')
from units import AtomicUnit
AU = AtomicUnit()

#import numba as nb
#@nb.jit(nopython=True)
#def rhs(t,vec, arg, dz):
    #dvdt = np.zeros(4*arg[0])
    #dvdt[arg[1]+1:arg[1]+arg[0]-1] = -( vec[arg[4]+2:arg[4]+arg[0]] - vec[arg[4]:arg[4]+arg[0]-2] )/dz/2# * self.unit.C**2 \
                                                  #- self.Jx[1:self.NZgrid-1]/self.unit.E0
	#dvdt[_Ey+1:self._Ey+self.NZgrid-1] =  ( vec[self._Bx+2:self._Bx+self.NZgrid] - vec[self._Bx:self._Bx+self.NZgrid-2] )/self.dZ/2 * self.unit.C**2 \
                                                  #- self.Jy[1:self.NZgrid-1]/self.unit.E0
	#dvdt[self._Bx+1:self._Bx+self.NZgrid-1] =  ( vec[self._Ey+2:self._Ey+self.NZgrid] - vec[self._Ey:self._Ey+self.NZgrid-2] )/self.dZ/2
	#dvdt[self._By+1:self._By+self.NZgrid-1] = -( vec[self._Ex+2:self._Ex+self.NZgrid] - vec[self._Ex:self._Ex+self.NZgrid-2] )/self.dZ/2
#
	#dvdt[self._Ex] = -( vec[self._By+1] - vec[self._By+0] )/self.dZ * self.unit.C**2  - self.Jx[0]/self.unit.E0
 	#dvdt[self._Ey] =  ( vec[self._Bx+1] - vec[self._Bx+0] )/self.dZ * self.unit.C**2  - self.Jx[0]/self.unit.E0
	#dvdt[self._Bx] =  ( vec[self._Ey+1] - vec[self._Ey+0] )/self.dZ
	#dvdt[self._By] = -( vec[self._Ex+1] - vec[self._Ex+0] )/self.dZ
#
	#dvdt[self._Ex+self.NZgrid-1] = -( vec[self._By+self.NZgrid-1] - vec[self._By+self.NZgrid-2] )/self.dZ * self.unit.C**2 - self.Jx[self.NZgrid-1]/self.unit.E0
	#dvdt[self._Ey+self.NZgrid-1] =  ( vec[self._Bx+self.NZgrid-1] - vec[self._Bx+self.NZgrid-2] )/self.dZ * self.unit.C**2 - self.Jy[self.NZgrid-1]/self.unit.E0
	#dvdt[self._Bx+self.NZgrid-1] =  ( vec[self._Ey+self.NZgrid-1] - vec[self._Ey+self.NZgrid-2] )/self.dZ
	#dvdt[self._By+self.NZgrid-1] = -( vec[self._Ex+self.NZgrid-1] - vec[self._Ex+self.NZgrid-2] )/self.dZ

    #return dvdt


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

class AugmentedEhrenfestMaxwellPropagator_1D(object):
    """
    EM field propagator using Augmented Ehrenfest in 1-D grid (z)
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
        absEx = np.sum(self.EB[self._Ex:self._Ex+self.NZgrid]**2)*self.dZ
        if intDE==0.0:
            self.EB[self._Ex:self._Ex+self.NZgrid] = np.random.choice([1, -1])*Dx * np.sqrt(deltaE/intDD)
        else:
            Kappa = [(-intDE + np.sqrt(intDE**2+2*intDD*deltaE) )/intDD, (-intDE - np.sqrt(intDE**2+2*intDD*deltaE) )/intDD]
            if np.abs(Kappa[0])<np.abs(Kappa[1]):
                Kappa = Kappa[0]
            else:
                Kappa = Kappa[1]
            self.EB[self._Ex:self._Ex+self.NZgrid] = self.EB[self._Ex:self._Ex+self.NZgrid] + Kappa* Dx[:]


    def ContinuousEmission_B(self,deltaE,intDE,intDD,Dx):
        absEx = np.sum(self.EB[self._Ex:self._Ex+self.NZgrid]**2)*self.dZ
        #if absEx==0.0:
            #pass
        if intDE==0.0:
            #self.EB[self._Ex:self._Ex+self.NZgrid] = Dx * np.sqrt(deltaE/intDD)
            self.EB[self._By:self._By+self.NZgrid] = np.random.choice([1, -1])*Dx * np.sqrt(1*deltaE/intDD)
			#pass
        else:
            Kappa = [(-intDE + np.sqrt(intDE**2+2*intDD*deltaE) )/intDD, (-intDE - np.sqrt(intDE**2+2*intDD*deltaE) )/intDD]
            if np.abs(Kappa[0])<np.abs(Kappa[1]):
                Kappa = Kappa[0]
            else:
                Kappa = Kappa[1]
            self.EB[self._By:self._By+self.NZgrid] = self.EB[self._By:self._By+self.NZgrid] + Kappa* Dx[:]
