#!/usr/bin/python
import unittest
import numpy as np
import sys
from utility import *
from units import AtomicUnit
AU = AtomicUnit()

from SystemPropagator import *


import matplotlib.pyplot as plt
from matplotlib import cm,rc
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman', size='12')


class TestPureStatePropagator(unittest.TestCase):
    """
    a simple TLS propagator should give the same dynamics
    """
    def setUp(self):
        param_TLS={
		    'nstates':  2,
		    # Hamiltoniam
		    'H0':       np.array([[-0.5,0.0],\
        		                  [0.0,0.5]]),
    		# Coupling
    		'VP':       np.array([[0.0,1.0],\
            		              [1.0,0.0]]),
    		# initial density matrix
            # 'rho0':     np.sqrt(np.array([[0.0,0.0],\
             		                      # [0.0,1.0]],complex)),
    		# relaxation and dephasing rate
    		'Gamma_r':  0.0,  # relaxation rate
    		'Gamma_d':  0.0,  # dephasing rate
            'Pmax':     0.025*np.sqrt(2.0),
		}
        param_TLS['Omega0']=(param_TLS['H0'][1,1]-param_TLS['H0'][0,0])
        param_TLS['levels']=[param_TLS['H0'][0,0],param_TLS['H0'][1,1]]
        param_TLS['C0']=np.sqrt(np.array([[1.0],[0.0]],complex))
        # param_TLS['C0']=np.sqrt(np.array([[0.5],[0.5]],complex))
        param_TLS=Struct(**param_TLS)

        self.param = param_TLS

    def test_propagate(self):
        """
        test the agreement of C and rho propagator
        """
        TLSP = PureStatePropagator(self.param)
        TLDMP = DensityMatrixPropagator(self.param)
        # TLDMP.initializeODEsolver(0.0)

        coupling  = 0.1
        Rabi = np.sqrt(coupling**2+(0.5*(self.param.H0[1,1]-self.param.H0[0,0]))**2)
        sintheta2 = (coupling/Rabi)**2
        TLSP.update_coupling(coupling)
        TLDMP.update_coupling(coupling)
        dt = 0.01
        Tmax = int(6*np.pi/dt)
        Ct = np.zeros(Tmax)
        Rt = np.zeros(Tmax)
        Et = np.zeros(Tmax)
        for it in range(Tmax):
            TLSP.propagate(dt)
            TLDMP.propagate(dt)
            # for i in range(10): TLDMP.propagate(dt/10)
            #for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
            for (i,j) in [(1,1)]:
                Ct[it] = np.real(TLSP.C[i,0]*np.conj(TLSP.C[j,0]))
                Rt[it] = np.real(TLDMP.rho[i,j])
                Et[it] = (sintheta2/4)*(2.0-2.0*np.cos(2*Rabi*(it+1)*dt))
                #print i,j,CCC,RHO,1.0-np.cos(it*dt)**2
                self.assertAlmostEqual(Ct[it],Et[it])
                self.assertAlmostEqual(Rt[it],Et[it])
        fig, ax= plt.subplots(figsize=(6.0,4.0))

        ax.plot(Ct,label='Ct',lw=2,alpha=0.5)
        ax.plot(Rt,label='Rt',lw=2,alpha=0.5)
        ax.plot(Et,'-k',label='Et',alpha=0.5)
        ax.legend()
        # plt.show()

    # def test_getComplement_angle(self):
    # def test_getEnergy(self):
    # def test_getrho(self):
    def test_rescale(self):
        param = self.param
        param.C0=np.sqrt(np.array([[0.5],[0.5]],complex))
        ii = 1
        jj = 0
        kRdt = 0.01

        drho = np.abs(param.C0[ii,0])**2*(1.0-np.exp(-kRdt))
        Pii_Direct = np.abs(param.C0[ii,0])**2 - drho
        Pjj_Direct = np.abs(param.C0[jj,0])**2 + drho


        TLSP = PureStatePropagator(param)
        TLSP.rescale(ii,jj,drho)
        self.assertAlmostEqual(np.abs(TLSP.C[ii,0])**2, Pii_Direct)
        self.assertAlmostEqual(np.abs(TLSP.C[jj,0])**2, Pjj_Direct)

        TLSP = PureStatePropagator(param)
        TLSP.rescale_kRdt(ii,jj,kRdt)
        self.assertAlmostEqual(np.abs(TLSP.C[ii,0])**2, Pii_Direct)
        self.assertAlmostEqual(np.abs(TLSP.C[jj,0])**2, Pjj_Direct)

        TLDMP = DensityMatrixPropagator(param)
        TLDMP.relaxation(ii,jj,kRdt)
        self.assertAlmostEqual(np.abs(TLDMP.rho[ii,ii]), Pii_Direct)
        self.assertAlmostEqual(np.abs(TLDMP.rho[jj,jj]), Pjj_Direct)




    def test_dephasing(self):
        param = self.param
        param.C0=np.sqrt(np.array([[0.5],[0.5*np.exp(1j*np.pi)]],complex))
        TLDMP = DensityMatrixPropagator(param)

        ii = 1
        jj = 0
        kDdt = 0.01

        TLDMP.dephasing(ii,jj,kDdt)

        rho01_Direct = param.C0[0,0]*np.conj(param.C0[1,0])*np.exp(-kDdt)

        self.assertAlmostEqual(np.abs(TLDMP.rho[0,1]), np.abs(rho01_Direct))
        self.assertAlmostEqual(np.angle(TLDMP.rho[0,1]), np.angle(rho01_Direct))

    # def test_equilibrate(self):

if __name__ == '__main__':
    unittest.main()
