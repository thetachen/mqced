import numpy as np
from scipy.integrate import ode
#execfile('atomic.unit')
from units import AtomicUnit
AU = AtomicUnit()

class ThreeLevelSystemPropagator(object):
    """
    simple three level system propagator
    """
    def __init__(self, param):
        self.param = param
        self.nstates = param.nstates
        self.Ht = np.zeros((self.nstates,self.nstates),complex)
        self.C = np.zeros((self.nstates,1),complex)

        #Set up wave vector
        self.C = param.C0

        #Set up Hamiltonian
        #self.H0 = np.diag(param.levels)
        self.H0 = param.H0
        for n in range(self.nstates):
            self.Ht[n,n] = self.H0[n,n]

        #Set up Polarization Operator
        self.VP = param.VP 

    def update_coupling(self,intPE):
        self.Ht = self.H0 - self.VP*intPE

    def update_coupling_byA(self,intAD):
        for n in range(self.nstates):
            self.Ht[n,n] = self.H0[n,n]
        for i in range(self.nstates):
            for j in range(i+1,self.nstates):
                self.Ht[i,j] = self.VP[i,j] * 1j*(self.H0[j,j]-self.H0[i,i])*intAD
                #self.Ht[i,j] = self.VP[i,j]*(self.H0[j,j]-self.H0[i,i])*intAD
                self.Ht[j,i] = np.conj(self.Ht[i,j])

    def propagate(self,dt):
        """
        propagate wave vector C by Ht   
        """
        W, U = np.linalg.eig(self.Ht)
        expiHt = np.dot(U,np.dot(np.diag(np.exp(-1j*W*dt)),np.conj(U).T))
        self.C = np.dot(expiHt,self.C)

    def getEnergy(self):
        Esys = 0.0
        for n in range(self.nstates):
            Esys += self.H0[n,n]*np.abs(self.C[n,0])**2
        return Esys

    def rescaleState(self,dE):
        if self.C[0,0] == 0.0:
            self.C[0,0] = np.sqrt(dE)
            self.C[1,0] = np.sqrt(1.0-dE)
        elif self.C[1,0] == 0.0:
            pass
        #elif np.abs(self.C[1,0])**2 > dE:
        else:
            #self.C[0,0] = self.C[0,0]/np.abs(self.C[0,0]) * np.sqrt(np.abs(self.C[0,0])**2 + dE)
            #self.C[1,0] = self.C[1,0]/np.abs(self.C[1,0]) * np.sqrt(np.abs(self.C[1,0])**2 - dE)

            self.C[0,0] = np.sqrt(np.abs(self.C[0,0])**2 + dE)
            self.C[1,0] = np.sqrt(np.abs(self.C[1,0])**2 - dE)

    def rescale(self,ii,jj,drho):
		"""
		rescale the state vector from ii to jj by drho
		"""
		if self.C[ii,0] == 0.0:
			pass
		elif self.C[jj,0] == 0.0:
			self.C[jj,0] = np.sqrt(drho)
			self.C[ii,0] = self.C[ii,0]/np.abs(self.C[ii,0])*np.sqrt(np.abs(self.C[ii,0])**2-drho)
			#TLSP.C[1,0] = TLSP.C[1,0]*np.exp(-dt*(gamma/2))
			#TLSP.C[0,0] = np.sqrt(1.0-np.abs(TLSP.C[1,0])**2)
		else:
			self.C[jj,0] = self.C[jj,0]/np.abs(self.C[jj,0]) * np.sqrt(np.abs(self.C[jj,0])**2 + drho)
			self.C[ii,0] = self.C[ii,0]/np.abs(self.C[ii,0]) * np.sqrt(np.abs(self.C[ii,0])**2 - drho)
        	#TLSP.C[1,0] = TLSP.C[1,0]*np.exp(-dt*(gamma/2))
        	#C0 = np.sqrt(1.0-np.abs(TLSP.C[1,0])**2)
        	#TLSP.C[0,0] = TLSP.C[0,0]/np.abs(TLSP.C[0,0]) * C0

class TwoLevelDensityMatrixPropagator(object):
    """
    two level system propagator using density matrix
	can involve population relaxation and dephasing
	Gamma_r and Gamma_d
    """
    def __init__(self, param):
        self.param = param
        self.nstates = 2
        self.Ht = np.zeros((self.nstates,self.nstates))
        self.Gamma_r = param.Gamma_r
        self.Gamma_d = param.Gamma_d

        #Set up density matrix
        self.rhomatrix = param.rho0
        self.rhovec = np.zeros(3)
        self.rhovec[0] = np.real(self.rhomatrix[1,1] - self.rhomatrix[0,0])
        self.rhovec[1] = np.real(self.rhomatrix[0,1])
        self.rhovec[2] = np.imag(self.rhomatrix[0,1])
        #self.rhovec = np.zeros(4,complex)
        #self.rhovec[0] = self.rhomatrix[0,0]
        #self.rhovec[1] = self.rhomatrix[0,1]
        #self.rhovec[2] = self.rhomatrix[1,0]
        #self.rhovec[3] = self.rhomatrix[1,1]

        #Set up Hamiltonian
        self.H0 = param.H0
        self.Ht = self.H0

        #Set up Polarization Operator
        self.VP = param.VP

    #def rho(self,i,j):
        #self.rhomatrix[0,0] = (1.0+self.rhovec[0])/2
        #self.rhomatrix[1,1] = (1.0-self.rhovec[0])/2
        #self.rhomatrix[0,1] = self.rhovec[1] + 1j*self.rhovec[2]
        #self.rhomatrix[1,0] = self.rhovec[1] - 1j*self.rhovec[2]
        ##self.rhomatrix[0,0] = self.rhovec[0]
        ##self.rhomatrix[0,1] = self.rhovec[1]
        ##self.rhomatrix[1,0] = self.rhovec[2]
        ##self.rhomatrix[1,1] = self.rhovec[3]
#
        #return self.rhomatrix[i,j]

    def update_coupling(self,intPE):
        self.Ht = self.H0 - self.VP*intPE

    def initializeODEsolver(self,T0):
        # Density matrix ode solver
        self.solver = ode(self.f)
        self.solver.set_integrator('dop853')
        self.solver.set_initial_value(self.rhovec, T0)
        self.rhovec = self.solver.y

    def f(self,t,vec):
        V = self.Ht[0,1]
        W = self.Ht[0,0]-self.Ht[1,1]
        dvdt = np.zeros(3)
        dvdt[0] = 4*V*self.rhovec[2] - self.Gamma_r*(self.rhovec[0]-(-1))
        dvdt[1] = W*self.rhovec[2] - self.Gamma_d/2*self.rhovec[1]
        dvdt[2] =-W*self.rhovec[1] - V*self.rhovec[0] - self.Gamma_d/2*self.rhovec[2]
        #dvdt = np.zeros(4,complex)
        #dvdt[0] = (self.Ht[0,1]*self.rhovec[2] - self.Ht[1,0]*self.rhovec[1])/1j
        #dvdt[1] = ((self.Ht[0,0]-self.Ht[1,1])*self.rhovec[1] + self.Ht[0,1]*(self.rhovec[3]-self.rhovec[1]))/1j
        #dvdt[2] = np.conj(dvdt[1])
        #dvdt[3] =-dvdt[0]
        return dvdt


    def propagate(self,dt):
        """
        propagate density matrix by Ht for dt  
        """
        self.solver.integrate(self.solver.t+dt)
        self.rhovec = self.solver.y
        self.rhomatrix[0,0] = (1.0-self.rhovec[0])/2
        self.rhomatrix[1,1] = (1.0+self.rhovec[0])/2
        self.rhomatrix[0,1] = self.rhovec[1] + 1j*self.rhovec[2]
        self.rhomatrix[1,0] = self.rhovec[1] - 1j*self.rhovec[2]

    def makePure(self):
        c = self.rhomatrix[0,0]*self.rhomatrix[1,1]
        d = self.rhovec[1]**2 + self.rhovec[2]**2
        self.rhomatrix[0,1] = self.rhomatrix[0,1] *c/d
        self.rhomatrix[1,0] = self.rhomatrix[1,0] *c/d

    #def propagate(self,dt): BUGGY!
        #"""
        #propagate wave vector C by Ht   
        #"""
        #V = self.Ht[0,1]*np.sqrt(2.0)
        #W = self.Ht[0,0]-self.Ht[1,1]  
        #Hmat = np.array([[ 0.0,   V,  -V],\
                         #[   V,  -W, 0.0],\
                         #[  -V, 0.0,   W]])
#
        #W, U = np.linalg.eig(Hmat)
        #expiHt = np.dot(U,np.dot(np.diag(np.exp(1j*W*dt)),np.conj(U).T))
        #self.rhovec = np.dot(expiHt,self.rhovec)
#
        #self.rhomatrix[0,0] = (1.0+self.rhovec[0])/2
        #self.rhomatrix[1,1] = (1.0-self.rhovec[0])/2
        #self.rhomatrix[0,1] = self.rhovec[1] + 1j*self.rhovec[2]
        #self.rhomatrix[1,0] = self.rhovec[1] - 1j*self.rhovec[2]


    def getEnergy(self):
        Esys = 0.0
        for n in range(self.nstates):
            Esys += self.H0[n,n]*np.real(self.rhomatrix[n,n])
        return Esys

class TwoLevelSurfaceHoppingPropagator(object):
    """
    two level system propagator using adiabatic surface hopping
    """
    def __init__(self, param):
        self.param = param
        self.nstates = 2

        #Set up wave vector
        self.Cd = param.C0
        self.Ca = np.zeros((self.nstates,1),complex)

        #Set up Hamiltonian
        self.H0 = param.H0
        self.Hd = np.zeros((self.nstates,self.nstates),complex)
        self.Ha = np.zeros((self.nstates,self.nstates),complex)

        #Set up Polarization Operator
        self.VP = param.VP

    def update_coupling(self,intDA, intDE):
        #self.Ht = self.H0 - self.VP*1j*Chi
        self.Hd[0,0] = self.H0[0,0]
        self.Hd[1,1] = self.H0[1,1]
        self.Hd[0,1] = 1j*(self.H0[1,1]-self.H0[0,0])*intDA
        self.Hd[1,0] = 1j*(self.H0[0,0]-self.H0[1,1])*intDA

        _chi = 1.0 + 4.0*intDA**2
        _norm = np.sqrt(2.0*_chi+2.0*np.sqrt(_chi))

        # Eigenvalues
        _lambda = 0.5 * (self.Hd[0,0]+self.Hd[1,1])
        self.Ha[0,0] = _lambda - 0.5*(self.Hd[1,1]-self.Hd[0,0]) * np.sqrt(_chi)
        self.Ha[1,1] = _lambda + 0.5*(self.Hd[1,1]-self.Hd[0,0]) * np.sqrt(_chi)

        # Derivative Coupling
        self.Ha[0,1] = 1j*intDE/_chi
        self.Ha[1,0] = np.conj(self.Ha[0,1])

        # Unitary Transformaion
        _cos = (1.0+np.sqrt(_chi))/_norm
        _sin = 2.0*intDA/_norm
        self.U = np.array([[    _cos,   _sin],\
                           [-1j*_sin,1j*_cos]],complex)
        
    def propagate(self,dt):
        """
        propagate adiabatic wave vector Ca by Ha   
        """
		# Ca = exp(-iHat)*Ca
        W, U = np.linalg.eig(self.Ha)
        expiHt = np.dot(U,np.dot(np.diag(np.exp(-1j*W*dt)),np.conj(U).T))
        self.Ca = np.dot(expiHt,self.Ca)
        #self.Ca = self.Ca + dt*np.dot(self.Ha,self.Ca)

		# Convert to Cd by unitary transform
        self.Cd = np.dot(self.U,self.Ca)

    def getProbability(self,i,j,dt):
        """
        compute the hopping probability from i to j
        """
        prob = np.imag(2*dt* self.Ha[i,j] * self.Ca[j,0]*np.conj(self.Ca[i,0]))\
              /(self.Ca[i,0]*np.conj(self.Ca[i,0]))

        return min(max(prob,0.0),1.0)

    def getEnergy(self):
        Esys = 0.0
        for n in range(self.nstates):
            Esys += self.H0[n,n]*np.abs(self.C[n,0])**2
        return Esys


class HarmonicOscillationPropagator(object):
    """
    simple harmonic oscillation propagator
    """
    def __init__(self, param):
        self.param = param

        self.N = param.NRgrid
        self.Rgrid = param.Rgrid
        self.dR = param.dR

        self.C = np.zeros((2*self.N,1),complex)
        self.VP = np.array([[0.0,1.0],\
                            [1.0,0.0]])
        self.Vt = np.zeros((2,2))

    def H0(self,R):
        H0 = np.zeros((2,2), complex)
        #H0[0,0] = 0.5*self.param.mass*(self.param.Wc**2)* ( R+self.param.Rc )**2
        #H0[1,1] = 0.5*self.param.mass*(self.param.Wc**2)* ( R-self.param.Rc )**2 + self.param.epsilon
        H0[0,0] = 0.5*self.param.mass*(self.param.Wc**2)* R**2
        H0[1,1] = 0.5*self.param.mass*(self.param.Wc**2)* R**2 - self.param.Gc*R + self.param.epsilon
        H0[0,1] = self.param.coupling
        H0[1,0] = self.param.coupling
		
        return H0+self.Vt

    def initializeGaussianWavePacket(self,sigma,R0,P0):
        # gaussian
        self.C = np.zeros((2*self.N,1),complex)
        for ir in range(len(self.Rgrid)):
            R = self.Rgrid[ir]
            self.C[ir,0] = np.exp(-0.5*(R-R0)**2/sigma**2)*np.exp(1j*P0*R)

            self.C[ir+self.N,0] = 0.0
        self.C = self.C/np.sqrt(np.sum(self.C*np.conj(self.C)))

    def generateKEPropagator(self,dt):
        #T = np.diag(-2.0*np.ones(2*self.N))
        #for i in range(self.N-1):
        #    T[i,i+1] = 1.0
        #    T[i+1,i] = 1.0
        #    T[i+self.N,i+1+self.N] = 1.0
        #    T[i+1+self.N,i+self.N] = 1.0
        # Central Finite difference coefficient:
        # http://en.wikipedia.org/wiki/Finite_difference_coefficients 
        T = np.diag((-49.0/18.0)*np.ones(self.N))
        for i in range(self.N):
            T[i,(i+1)%self.N] =  3.0/2.0  if i+1 in range(self.N) else T[i,(i+1)%self.N]
            T[i,(i+2)%self.N] = -3.0/20.0 if i+2 in range(self.N) else T[i,(i+2)%self.N]
            T[i,(i+3)%self.N] =  1.0/90.0 if i+3 in range(self.N) else T[i,(i+3)%self.N]
            T[i,(i-1)%self.N] =  3.0/2.0  if i-1 in range(self.N) else T[i,(i-1)%self.N]
            T[i,(i-2)%self.N] = -3.0/20.0 if i-2 in range(self.N) else T[i,(i-2)%self.N]
            T[i,(i-3)%self.N] =  1.0/90.0 if i-3 in range(self.N) else T[i,(i-3)%self.N]
        z = np.zeros((self.N,self.N))
        T = np.bmat([[T,z],[z,T]])

        W, U = np.linalg.eigh(-T/(self.dR**2)/self.param.mass/2)
        self.expT = np.dot(U,np.dot(np.diag(np.exp(-1j*W*dt)),np.conj(U).T))

    def generatePEPropagator(self,dt):
        self.expV = np.zeros((2*self.N,2*self.N),complex)
        for i in range(self.N):
            R = self.Rgrid[i]
            e = (self.H0(R)[0,0]-self.H0(R)[1,1])/2
            m = (self.H0(R)[0,0]+self.H0(R)[1,1])/2
            d = np.abs(self.H0(R)[0,1])
            if d ==0.0:
                ExpiPhi = 1.0
            else:
                ExpiPhi = self.H0(R)[0,1]/d

            l = np.sqrt(e**2+d**2)
            expm = np.exp(-1j*m*dt/2)
            expl = np.exp( 1j*l*dt/2)
            rl = np.real(expl)
            il = np.imag(expl)
            cos = e/l
            sin = d/l

            self.expV[i,i]= (rl - 1j*il*cos)*expm
            self.expV[i+self.N,i+self.N]= (rl + 1j*il*cos)*expm
            self.expV[i, i+self.N] = -1j*il*sin*ExpiPhi
            self.expV[i+self.N, i] = -1j*il*sin/ExpiPhi

    def update_Vt(self,intPE):
        self.Vt = self.VP*intPE

    def propagate(self,dt):
        """
        propagate wave vector C by Ht   
        """
        self.C = np.dot(self.expV,np.dot(self.expT,np.dot(self.expV,self.C)))



    def getEnergySurfaces(self):
        PES = np.zeros((2,self.N))
        for i in range(self.N):
            R = self.Rgrid[i]
            PES[0,i] = self.H0(R)[0,0]
            PES[1,i] = self.H0(R)[1,1]
        return PES

    def getWaveFunction(self):
        wf = np.zeros((2,self.N))
        for i in range(self.N):
            wf[0,i]=np.abs(self.C[i,0])**2
            wf[1,i]=np.abs(self.C[i+self.N,0])**2

        return wf

