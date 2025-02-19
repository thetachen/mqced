import numpy as np
import copy
from random import random
from scipy.integrate import ode
#execfile('atomic.unit')
from units import AtomicUnit
AU = AtomicUnit()

class PureStatePropagator(object):
    """
    system propagator in terms of pure state wavefunction
    """
    def __init__(self, param,KFGR_dimension="1D"):
        self.param = param
        self.nstates = param.nstates
        self.Ht = np.zeros((self.nstates,self.nstates),complex)
        self.C = np.zeros((self.nstates,1),complex)
        self.rho = np.zeros((self.nstates,self.nstates),complex)

        #Set up wave vector
        self.C = copy.copy(param.C0)

        #Set up Hamiltonian
        #self.H0 = np.diag(param.levels)
        self.H0 = param.H0
        for n in range(self.nstates):
            self.Ht[n,n] = self.H0[n,n]

        #Set up Polarization Operator
        self.VP = param.VP

        # generate FGR rate
        self.FGR = np.zeros((self.nstates,self.nstates))
        for i in range(self.nstates):
            for j in range(self.nstates):
                if KFGR_dimension=="3D":
                    self.FGR[i,j] = ((self.H0[i,i]-self.H0[j,j])**3)*(param.Pmax**2)/3/np.pi
                else:
                    self.FGR[i,j] = (self.H0[i,i]-self.H0[j,j])*param.Pmax**2 #/AU.C/AU.E0 / AU.fs

        self.getrho()

    def update_coupling(self,intPE):
        self.Ht = self.H0 - self.VP*intPE

    def update_static_dipole(self,intSE):
        self.Ht = self.Ht - self.VS*intSE

    def propagate(self,dt):
        """
        propagate wave vector C by Ht
        """
        W, U = np.linalg.eig(self.Ht)
        expiHt = np.dot(U,np.dot(np.diag(np.exp(-1j*W*dt)),np.conj(U).T))
        self.C = np.dot(expiHt,self.C)
        self.getrho()

    def getEnergy(self):
        # Esys = 0.0
        # for n in range(self.nstates):
            # Esys += self.H0[n,n]*np.abs(self.C[n,0])**2
        # return Esys
        Esys = np.dot(np.conj(self.C).T,np.dot(self.H0,self.C))
        return np.real(Esys[0,0])

    def getrho(self):
        for i in range(self.nstates):
            for j in range(self.nstates):
                self.rho[i,j] = self.C[i,0]*np.conj(self.C[j,0])
        return self.rho

    def rescale(self,ii,jj,drho):
        """
        rescale the state vector from ii to jj by drho
        """
        if self.C[ii,0] == 0.0:
            pass
        elif self.C[jj,0] == 0.0:
            self.C[jj,0] = np.sqrt(drho)
			#TODO: there should be a random phase here!!!
            self.C[ii,0] = self.C[ii,0]/np.abs(self.C[ii,0])*np.sqrt(np.abs(self.C[ii,0])**2-drho)
            self.C[ii,0] *= np.exp(1j*2*np.pi*random())
			#TLSP.C[1,0] = TLSP.C[1,0]*np.exp(-dt*(gamma/2))
			#TLSP.C[0,0] = np.sqrt(1.0-np.abs(TLSP.C[1,0])**2)
        # else:
        elif np.abs(self.C[ii,0])**2 - drho > 0.0:
            self.C[jj,0] = self.C[jj,0]/np.abs(self.C[jj,0]) * np.sqrt(np.abs(self.C[jj,0])**2 + drho)
            self.C[ii,0] = self.C[ii,0]/np.abs(self.C[ii,0]) * np.sqrt(np.abs(self.C[ii,0])**2 - drho)
        	#TLSP.C[1,0] = TLSP.C[1,0]*np.exp(-dt*(gamma/2))
        	#C0 = np.sqrt(1.0-np.abs(TLSP.C[1,0])**2)
        	#TLSP.C[0,0] = TLSP.C[0,0]/np.abs(TLSP.C[0,0]) * C0
        self.getrho()

    def relaxation(self,ii,jj,kRdt):
        """
        rescale the state vector from ii to jj by exp(-kRdt)
        """
        if self.C[ii,0] == 0.0:
            pass
        elif self.C[jj,0] == 0.0:
            self.C[jj,0] = np.sqrt(1.0-np.exp(-kRdt))
            self.C[ii,0] = self.C[ii,0]*np.exp(-kRdt/2)
            self.C[ii,0] *= np.exp(1j*2*np.pi*random())
        else:
            self.C[jj,0] = self.C[jj,0]*np.sqrt(1.0+(np.abs(self.C[ii,0])**2/np.abs(self.C[jj,0])**2)*(1.0-np.exp(-kRdt)))
            self.C[ii,0] = self.C[ii,0]*np.exp(-kRdt/2)
        self.getrho()

    def dephasing(self,ii,jj,kDdt):
        """
        make a dephasing with probailtity kDdt
        """
        test = random()
        if test < kDdt:
            # collapse
            #self.C[ii,0] = np.abs(self.C[ii,0])*np.exp(1j*2*np.pi*random())
            self.C[jj,0] = np.abs(self.C[jj,0])*np.exp(1j*2*np.pi*random())
        self.getrho()

    def resetting(self,ii,jj,kDdt):
        """
        dummy function doing nothing
        """
        return

    def equilibrate(self,ii,jj,dt,transition=[1,1]):
        """
        equilibrate state ii and jj according to Boltzmann distribution
        transition = [up, down]
        """
        drho = self.param.gamma_vib*dt *( np.abs(self.C[ii,0])**2 * np.exp(-self.param.beta*self.H0[jj,jj]) \
                              -np.abs(self.C[jj,0])**2 * np.exp(-self.param.beta*self.H0[ii,ii]) )
        if drho > 0.0:
            # ii->jj
            self.rescale(ii,jj,np.abs(drho)*transition[0])
        elif drho < 0.0:
            # jj->ii
            self.rescale(jj,ii,np.abs(drho)*transition[1])
        else:
            pass

    def getComplement(self,ii,ff,dt):
        """
        calcualte complementary from ii --> ff state
        """
        gamma = self.FGR[ii,ff] * (1.0 - np.abs(self.rho[ff,ff]))
        drho = gamma*dt * np.abs(self.rho[ii,ii])
        if np.abs(self.rho[ff,ii])!=0.0:
            drho *= 2*(np.imag(self.rho[ff,ii])/np.abs(self.rho[ff,ii]))**2

        dE = (self.H0[ii,ii]-self.H0[ff,ff])*drho
        return drho, dE

    def getComplement_angle(self,ii,ff,dt,angle):
        """
        calcualte complementary from ii --> ff state
        """
        kR = self.FGR[ii,ff] * (1.0 - np.abs(self.rho[ff,ff]))
        if np.abs(self.rho[ff,ii])!=0.0:
            kR *= 2*(np.sin(angle))**2

        kRdt = kR*dt
        drho = np.abs(self.rho[ii,ii])*(1.0-np.exp(-kRdt))
        dE = (self.H0[ii,ii]-self.H0[ff,ff])*drho

        kD = self.FGR[ii,ff]/2 * (  1.0 - ( np.abs(self.rho[ff,ff]) - np.abs(self.rho[ii,ii]) ) )
        #kD = self.FGR[ii,ff] * np.abs(self.rho[ii,ii])
        kDdt = np.abs(kD)*dt

        return kRdt,kDdt,drho,dE

        # gamma = self.FGR[ii,ff] * (1.0 - np.abs(self.rho[ff,ff]))
        # drho = gamma*dt * np.abs(self.rho[ii,ii])
        # if np.abs(self.rho[ff,ii])!=0.0:
        #     drho *= 2*(np.sin(angle))**2
        #
        # dE = (self.H0[ii,ii]-self.H0[ff,ff])*drho
        # return drho, dE

class DensityMatrixPropagator(object):
    """
    two level system propagator using density matrix
	can involve population relaxation and dephasing
	Gamma_r and Gamma_d
    """
    def __init__(self, param,KFGR_dimension="1D"):
        self.param = param
        self.nstates = self.param.nstates
        self.Ht = np.zeros((self.nstates,self.nstates))
        self.rho = np.zeros((self.nstates,self.nstates),complex)

        #Set up density matrix
        try:
            self.rho = param.rho0
        except:
			for i in range(self.nstates):
				for j in range(self.nstates):
					self.rho[i,j] = param.C0[i,0]*np.conj(param.C0[j,0])


        #Set up Hamiltonian
        self.H0 = param.H0
        self.Ht = self.H0

        #Set up Polarization Operator
        self.VP = param.VP

        # generate FGR rate
        self.FGR = np.zeros((self.nstates,self.nstates))
        for i in range(self.nstates):
            for j in range(self.nstates):
                if KFGR_dimension=="3D":
                    self.FGR[i,j] = ((self.H0[i,i]-self.H0[j,j])**3)*(param.Pmax**2)/3/np.pi
                else:
                    self.FGR[i,j] = (self.H0[i,i]-self.H0[j,j])*param.Pmax**2 #/AU.C/AU.E0 / AU.fs

    def update_coupling(self,intPE):
        self.Ht = self.H0 - self.VP*intPE

    def update_static_dipole(self,intSE):
        self.Ht = self.Ht - self.VS*intSE

    def propagate(self,dt):
        """
        propagate density matrix by Ht for dt
        """
        # if np.abs(self.rho[0,1])==0.0: kill=True
        # else:   kill = False
        W, U = np.linalg.eig(self.Ht)
        expiHt = np.dot(U,np.dot(np.diag(np.exp(1j*W*dt)),np.conj(U).T))
        self.rho = np.dot(np.conj(expiHt).T,np.dot(self.rho,expiHt))
        # if kill:
            # self.rho[0,1]=0.0*1j
            # self.rho[1,0]=0.0*1j

    def relaxation(self,ii,jj,kRdt):
        """
        make a population relaxation of the density matrix from ii to jj by drho
        """
        drho = self.rho[ii,ii]*(1.0-np.exp(-kRdt))
        self.rho[ii,ii] = self.rho[ii,ii]-drho
        self.rho[jj,jj] = self.rho[jj,jj]+drho


    def dephasing(self,ii,jj,kDdt):
        """
        make a dephasing of the density matrix of ii,jj element by e^-kDdt
        """
        self.rho[ii,jj] = self.rho[ii,jj]*(np.exp(-kDdt))
        self.rho[jj,ii] = self.rho[jj,ii]*(np.exp(-kDdt))

        if np.abs(self.rho[ii,ii])==0.0 or np.abs(self.rho[jj,jj])==0.0:
            random_phase = 2*np.pi*random()
            self.rho[ii,jj] = self.rho[ii,jj]*np.exp( 1j*random_phase )
            self.rho[jj,ii] = self.rho[jj,ii]*np.exp(-1j*random_phase )


    def resetting(self,ii,jj,kDdt):
        """
        make a dephasing of the density matrix of ii,jj element by e^-kDdt
        """
        test = random()
        if test < kDdt:
            # reset (with a random phase )
            random_phase = 2*np.pi*random()
            self.rho[ii,jj] = self.rho[ii,jj]*np.exp( 1j*random_phase )
            self.rho[jj,ii] = self.rho[jj,ii]*np.exp(-1j*random_phase )
            print "reset"
            # # rescale (but keep phase )
            # new_coherence = np.sqrt( self.rho[ii,ii]*self.rho[jj,jj] )
            # self.rho[ii,jj] = self.rho[ii,jj]/np.abs(self.rho[ii,jj]) * new_coherence
            # self.rho[jj,ii] = self.rho[jj,ii]/np.abs(self.rho[jj,ii]) * new_coherence
            # print "reset"

    def getComplement_angle(self,ii,ff,dt,angle):
        """
        calcualte complementary from ii to ff state
        """

        if np.abs(self.rho[ii,ii])!=0.0:
            kR = self.FGR[ii,ff] * np.real( 1.0 - (np.abs(self.rho[ii,ff])**2)/np.abs(self.rho[ii,ii]) )* 2*(np.sin(angle))**2
            # kR = self.FGR[ii,ff] * ( 1.0 - (np.abs(self.rho[ii,ff])**2)/np.abs(self.rho[ii,ii]) )
        else:
            kR = 0.0

        # if np.abs(self.rho[ff,ii])!=0.0:
        #     kRdt = kR *dt * 2*(np.sin(angle))**2
        # else:
        #     kRdt = kR *dt
        kRdt = kR *dt
        drho = np.real(self.rho[ii,ii])*(1.0-np.exp(-kRdt))
        # print drho
        if np.abs(self.rho[ff,ii])!=0.0:
            # kD = self.FGR[ii,ff] * (  1.0 - 2.0*(np.imag(self.rho[ii,ff])**2)/(np.abs(self.rho[ii,ff])**2) \
                                           # *( np.abs(self.rho[ff,ff]) - np.abs(self.rho[ii,ii]) )  )
            kD = self.FGR[ii,ff]/2 * np.real(  1.0 - ( np.abs(self.rho[ff,ff]) - np.abs(self.rho[ii,ii]) ) )
        else:
            kD = 0.0

        kDdt = np.abs(kD)*dt
        # print kDdt

        dE = np.real(self.H0[ii,ii]-self.H0[ff,ff])*drho

        return kRdt,kDdt,drho,dE

    def getEnergy(self):
        # Esys = 0.0
        # for n in range(self.nstates):
        #     Esys += self.H0[n,n]*np.real(self.rho[n,n])
        # return Esys
        Esys = np.trace(np.dot(self.H0,self.rho))
        return Esys

    def LindbladDecay(self,ii,jj,drho):
        """
        rescale the density matrix by a Lindblad operator
        """
        drho12 = 0.5*drho*self.rho[0,1]/self.rho[1,1]
        self.rho[0,0] = self.rho[0,0]+drho
        self.rho[1,1] = self.rho[1,1]-drho
        self.rho[0,1] = self.rho[0,1]-drho12
        self.rho[1,0] = self.rho[1,0]-np.conj(drho12)

    def Lindblad(self,dt):
        """
        propagate the density matrix by a Lindblad operator
        """
        self.rho[1,1] = self.rho[1,1]*np.exp(-self.FGR[1,0]*dt)
        self.rho[0,0] = 1.0 - self.rho[1,1]
        self.rho[0,1] = self.rho[0,1]*np.exp(-self.FGR[1,0]*dt/2)
        self.rho[1,0] = self.rho[1,0]*np.exp(-self.FGR[1,0]*dt/2)

    def _rescale(self,ii,jj,drho):
        """
        rescale the density matrix from ii to jj by drho
        """
        if self.rho[ii,ii] == 0.0:
            pass
        elif self.rho[jj,jj] == 0.0:
            self.rho[jj,jj] = drho
            #TODO: there should be a random phase here!!!
            self.rho[ii,ii] = 1.0-drho
            self.rho[ii,jj] = np.sqrt(drho*(1.0-drho))
            self.rho[jj,ii] = np.sqrt(drho*(1.0-drho))
        #else:
        elif np.abs(self.rho[ii,ii]) - drho > 0.0:
            self.rho[jj,jj] += drho
            self.rho[ii,ii] -= drho
            self.rho[jj,ii] = self.rho[jj,ii]/np.abs(self.rho[jj,ii]) * np.sqrt(self.rho[0,0]*self.rho[1,1])
            self.rho[ii,jj] = self.rho[ii,jj]/np.abs(self.rho[ii,jj]) * np.sqrt(self.rho[0,0]*self.rho[1,1])
            #TLSP.C[1,0] = TLSP.C[1,0]*np.exp(-dt*(gamma/2))
            #C0 = np.sqrt(1.0-np.abs(TLSP.C[1,0])**2)
            #TLSP.C[0,0] = TLSP.C[0,0]/np.abs(TLSP.C[0,0]) * C0

            #rho(0,0) += additional_population_loss;
            #rho(1,1) -= additional_population_loss;
            #rho(0,1) = rho(0,1) / abs(rho(0,1)) * sqrt(rho(0,0) * rho(1,1));
            #rho(1,0) = rho(1,0) / abs(rho(1,0)) * sqrt(rho(0,0) * rho(1,1));

class PureStatePropagator_Transform(object):
    """
    system propagator in terms of pure state wavefunction
    This object is for electron transfer study.
    """
    def __init__(self, param,KFGR_dimension="1D",Transform=True):
        self.param = param
        self.nstates = param.nstates
        self.Ht = np.zeros((self.nstates,self.nstates),complex)
        self.C = np.zeros((self.nstates,1),complex)
        self.rho = np.zeros((self.nstates,self.nstates),complex)

        #Set up wave vector
        self.C = copy.copy(param.C0)

        #Set up Hamiltonian
        #self.H0 = np.diag(param.levels)
        self.H0 = param.H0
        for n in range(self.nstates):
            self.Ht[n,n] = self.H0[n,n]

        #Set up Polarization Operator
        self.VP = param.VP
        self.VS = param.VS
        self.Pmax = param.Pmax
        self.Smax = param.Smax
        self.coupling = self.VP*self.Pmax + self.VS*self.Smax

        # generate FGR rate
        self.FGR = np.zeros((self.nstates,self.nstates))
        for i in range(self.nstates):
            for j in range(self.nstates):
                if KFGR_dimension=="3D":
                    self.FGR[i,j] = ((self.H0[i,i]-self.H0[j,j])**3)*(param.Pmax**2)/3/np.pi
                else:
                    self.FGR[i,j] = (self.H0[i,i]-self.H0[j,j])*param.Pmax**2 #/AU.C/AU.E0 / AU.fs

        if Transform:
            # generate unitary transformation
            sin2 = self.H0[0,1]/np.sqrt( 0.25*(self.H0[1,1]-self.H0[0,0])**2+(self.H0[0,1])**2 )
            cos2 = 0.5*(self.H0[1,1]-self.H0[0,0])/np.sqrt( 0.25*(self.H0[1,1]-self.H0[0,0])**2+(self.H0[0,1])**2 )
            cos = np.sqrt((1+cos2)/2)
            sin = np.sqrt((1-cos2)/2)
            # self.trans = np.array([[cos,sin],[-sin,cos]])
            self.H0_trans, self.trans = np.linalg.eig(self.H0)
            self.H0 = np.diag(self.H0_trans)
            self.VP = np.dot(self.trans.T,np.dot(self.VP,self.trans))
            self.VS = np.dot(self.trans.T,np.dot(self.VS,self.trans))
            self.coupling = np.dot(self.trans.T,np.dot(self.coupling,self.trans))
            # self.C = np.dot(self.trans.T, self.C)  #initial wavefunction

            print self.H0
            print self.coupling
            print self.Smax*cos2 - self.Pmax*sin2
            print self.Smax*sin2 + self.Pmax*cos2
            # generate new FGR rate
            self.FGR = np.zeros((self.nstates,self.nstates))
            for i in range(self.nstates):
                for j in range(self.nstates):
                    if KFGR_dimension=="3D":
                        self.FGR[i,j] = ((self.H0[i,i]-self.H0[j,j])**3)*(param.Pmax**2)/3/np.pi
                    else:
                        self.FGR[i,j] = (self.H0[i,i]-self.H0[j,j])*self.coupling[i,j]**2 #/AU.C/AU.E0 / AU.fs
        else:
            self.trans = np.diag([1.0,1.0])
        self.getrho()

    def update_coupling(self,intPE,intSE):
        self.coupling = self.VP*intPE + self.VS*intSE
        self.Ht = self.H0 - self.coupling

    def propagate(self,dt):
        """
        propagate wave vector C by Ht
        """
        W, U = np.linalg.eig(self.Ht)
        expiHt = np.dot(U,np.dot(np.diag(np.exp(-1j*W*dt)),np.conj(U).T))
        self.C = np.dot(expiHt,self.C)
        self.getrho()

    def getEnergy(self):
        # Esys = 0.0
        # for n in range(self.nstates):
            # Esys += self.H0[n,n]*np.abs(self.C[n,0])**2
        # return Esys
        Esys = np.dot(np.conj(self.C).T,np.dot(self.H0,self.C))
        return np.real(Esys[0,0])

    def getrho(self):
        for i in range(self.nstates):
            for j in range(self.nstates):
                self.rho[i,j] = self.C[i,0]*np.conj(self.C[j,0])
        return self.rho

    def rescale(self,ii,jj,drho):
        """
        rescale the state vector from ii to jj by drho
        """
        if self.C[ii,0] == 0.0:
            pass
        elif self.C[jj,0] == 0.0:
            self.C[jj,0] = np.sqrt(drho)
			#TODO: there should be a random phase here!!!
            self.C[ii,0] = self.C[ii,0]/np.abs(self.C[ii,0])*np.sqrt(np.abs(self.C[ii,0])**2-drho)
            self.C[ii,0] *= np.exp(1j*2*np.pi*random())
			#TLSP.C[1,0] = TLSP.C[1,0]*np.exp(-dt*(gamma/2))
			#TLSP.C[0,0] = np.sqrt(1.0-np.abs(TLSP.C[1,0])**2)
        # else:
        elif np.abs(self.C[ii,0])**2 - drho > 0.0:
            self.C[jj,0] = self.C[jj,0]/np.abs(self.C[jj,0]) * np.sqrt(np.abs(self.C[jj,0])**2 + drho)
            self.C[ii,0] = self.C[ii,0]/np.abs(self.C[ii,0]) * np.sqrt(np.abs(self.C[ii,0])**2 - drho)
        	#TLSP.C[1,0] = TLSP.C[1,0]*np.exp(-dt*(gamma/2))
        	#C0 = np.sqrt(1.0-np.abs(TLSP.C[1,0])**2)
        	#TLSP.C[0,0] = TLSP.C[0,0]/np.abs(TLSP.C[0,0]) * C0
        self.getrho()

    def relaxation(self,ii,jj,kRdt):
        """
        rescale the state vector from ii to jj by exp(-kRdt)
        """
        if self.C[ii,0] == 0.0:
            pass
        elif self.C[jj,0] == 0.0:
            self.C[jj,0] = np.sqrt(1.0-np.exp(-kRdt))
            self.C[ii,0] = self.C[ii,0]*np.exp(-kRdt/2)
            self.C[ii,0] *= np.exp(1j*2*np.pi*random())
        else:
            self.C[jj,0] = self.C[jj,0]*np.sqrt(1.0+(np.abs(self.C[ii,0])**2/np.abs(self.C[jj,0])**2)*(1.0-np.exp(-kRdt)))
            self.C[ii,0] = self.C[ii,0]*np.exp(-kRdt/2)
        self.getrho()

    def dephasing(self,ii,jj,kDdt):
        """
        make a dephasing with probailtity kDdt
        """
        test = random()
        if test < kDdt:
            # collapse
            #self.C[ii,0] = np.abs(self.C[ii,0])*np.exp(1j*2*np.pi*random())
            self.C[jj,0] = np.abs(self.C[jj,0])*np.exp(1j*2*np.pi*random())
        self.getrho()

    def resetting(self,ii,jj,kDdt):
        """
        dummy function doing nothing
        """
        return

    def equilibrate(self,ii,jj,dt,transition=[1,1]):
        """
        equilibrate state ii and jj according to Boltzmann distribution
        transition = [up, down]
        """
        drho = self.param.gamma_vib*dt *( np.abs(self.C[ii,0])**2 * np.exp(-self.param.beta*self.H0[jj,jj]) \
                              -np.abs(self.C[jj,0])**2 * np.exp(-self.param.beta*self.H0[ii,ii]) )
        if drho > 0.0:
            # ii->jj
            self.rescale(ii,jj,np.abs(drho)*transition[0])
        elif drho < 0.0:
            # jj->ii
            self.rescale(jj,ii,np.abs(drho)*transition[1])
        else:
            pass

    def getComplement(self,ii,ff,dt):
        """
        calcualte complementary from ii --> ff state
        """
        gamma = self.FGR[ii,ff] * (1.0 - np.abs(self.rho[ff,ff]))
        drho = gamma*dt * np.abs(self.rho[ii,ii])
        if np.abs(self.rho[ff,ii])!=0.0:
            drho *= 2*(np.imag(self.rho[ff,ii])/np.abs(self.rho[ff,ii]))**2

        dE = (self.H0[ii,ii]-self.H0[ff,ff])*drho
        return drho, dE

    def getComplement_angle(self,ii,ff,dt,angle):
        """
        calcualte complementary from ii --> ff state
        """
        kR = self.FGR[ii,ff] * (1.0 - np.abs(self.rho[ff,ff]))
        if np.abs(self.rho[ff,ii])!=0.0:
            kR *= 2*(np.sin(angle))**2

        kRdt = kR*dt
        drho = np.abs(self.rho[ii,ii])*(1.0-np.exp(-kRdt))
        dE = (self.H0[ii,ii]-self.H0[ff,ff])*drho

        kD = self.FGR[ii,ff]/2 * (  1.0 - ( np.abs(self.rho[ff,ff]) - np.abs(self.rho[ii,ii]) ) )
        #kD = self.FGR[ii,ff] * np.abs(self.rho[ii,ii])
        kDdt = np.abs(kD)*dt

        return kRdt,kDdt,drho,dE

        # gamma = self.FGR[ii,ff] * (1.0 - np.abs(self.rho[ff,ff]))
        # drho = gamma*dt * np.abs(self.rho[ii,ii])
        # if np.abs(self.rho[ff,ii])!=0.0:
        #     drho *= 2*(np.sin(angle))**2
        #
        # dE = (self.H0[ii,ii]-self.H0[ff,ff])*drho
        # return drho, dE

class FloquetStatePropagator(object):
    """
    system propagator using Floquet state wavefunction
    """
    def __init__(self, param_TLS, param_EM,dt,NPMAX=100):
        self.NPMAX = NPMAX
        self.nstates = param_TLS.nstates
        self.C = np.zeros((self.nstates,1),complex)
        self.rho = np.zeros((self.nstates,self.nstates),complex)

        #Set up wave vector
        self.C = copy.copy(param_TLS.C0)

        #Set up Hamiltonian
        #self.H0 = np.diag(param.levels)
        self.H0 = param_TLS.H0
        #for n in range(self.nstates):
            #self.Ht[n,n] = self.H0[n,n]

        #Set up Polarization Operator
        self.VP = param_TLS.VP
        self.Pmax = param_TLS.Pmax

        # external field
        self.A_CW = param_EM.A_CW
        self.K_CW = param_EM.K_CW

        # construct Floquet Hamiltonian
        NFLOQUET = (2*self.NPMAX+1)*self.nstates
        self.HF = np.zeros((NFLOQUET,NFLOQUET))
        self.CF = np.zeros((NFLOQUET,1),complex)
        for n in range(-self.NPMAX,self.NPMAX+1):
            nb = self.nstates*(n+self.NPMAX)
            for i in range(self.nstates):
                self.HF[nb+i,nb+i]= self.H0[i,i] - n*self.K_CW

            if n!=self.NPMAX:
                self.HF[nb+1,nb+self.nstates]=-self.A_CW*self.Pmax/2
                self.HF[nb,nb+self.nstates+1]=-self.A_CW*self.Pmax/2
                self.HF[nb+self.nstates,nb+1]=-self.A_CW*self.Pmax/2
                self.HF[nb+self.nstates+1,nb]=-self.A_CW*self.Pmax/2

        # initialize CF
        for i in range(self.nstates):
            self.CF[self.nstates*(0+self.NPMAX)+i] = self.C[i,0]

    #def MakePropagator(self,dt):
        WF, UF = np.linalg.eig(self.HF)
        self.expiHFdt = np.dot(UF,np.dot(np.diag(np.exp(-1j*WF*dt)),np.conj(UF).T))

    def CFtoCt(self,time):
        for i in range(self.nstates):
            self.C[i,0] = 0.0
            for n in range(-self.NPMAX,self.NPMAX+1):
                nb = self.nstates*(n+self.NPMAX)
                self.C[i,0] += self.CF[nb+i,0]*np.exp(-1j*n*self.K_CW*time)

    def propagate(self,time):
        self.CF = np.dot(self.expiHFdt,self.CF)
        self.CFtoCt(time)
        self.getrho()

    def decay(self,gamma,dt,time):
        if np.abs(self.C[0,0])!=0.0 and gamma>1E-8:
            for n in range(-self.NPMAX,self.NPMAX+1):
                nb = self.nstates*(n+self.NPMAX)
                self.CF[nb+0,0] = self.CF[nb+0,0]*np.sqrt((1.0-np.exp(-gamma*dt)*np.abs(self.C[1,0])**2)/np.abs(self.C[0,0])**2)
                self.CF[nb+1,0] = self.CF[nb+1,0]*np.exp(-0.5*gamma*dt)
        self.CFtoCt(time)
        self.getrho()

    def getrho(self):
        for i in range(self.nstates):
            for j in range(self.nstates):
                self.rho[i,j] = self.C[i,0]*np.conj(self.C[j,0])
        return self.rho

    def update_coupling(self,intPE):
        pass

    def rescale(self,ii,jj,drho):
        pass

    def getEnergy(self):
        Esys = 0.0
        for n in range(self.nstates):
            Esys += self.H0[n,n]*np.real(self.rho[n,n])
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
