#!/usr/bin/python
import unittest
import numpy as np
import sys
sys.path.append("/home/theta/sync/tdsh/EM/Tao")
sys.path.append("/home/theta/sync/tdsh/EM/Tao/peakdetect")
from utility import *
from units import AtomicUnit
AU = AtomicUnit()

from SystemPropagator import *


import matplotlib.pyplot as plt
from matplotlib import cm,rc
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman', size='12')


class TestPropagator(unittest.TestCase):
    """
    a simple TLS propagator should give the same dynamics
    """
    def setUp(self):
        param_TLS={
		    'nstates':  2,
		    # Hamiltoniam
		    'H0':       np.array([[0.0,0.0],\
        		                  [0.0,5.0]]),
    		# Coupling 
    		'VP':       np.array([[0.0,1.0],\
            		              [1.0,0.0]]),
    		# initial density matrix
    		'rho0':     np.sqrt(np.array([[0.0,0.0],\
             		                      [0.0,1.0]],complex)),
    		# relaxation and dephasing rate
    		'Gamma_r':  0.0,  # relaxation rate
    		'Gamma_d':  0.0,  # dephasing rate
		}
        param_TLS['Omega0']=(param_TLS['H0'][1,1]-param_TLS['H0'][0,0])
        param_TLS['levels']=[param_TLS['H0'][0,0],param_TLS['H0'][1,1]]
        param_TLS['C0']=np.sqrt(np.array([[0.0],[1.0]],complex))
        param_TLS=Struct(**param_TLS)

        self.param = param_TLS

    def test_propagate(self):
        
        TLSP = ThreeLevelSystemPropagator(self.param)
        TLDMP = TwoLevelDensityMatrixPropagator(self.param)
        TLDMP.initializeODEsolver(0.0)

        TLSP.update_coupling(1.0)
        TLDMP.update_coupling(1.0)
        dt = 0.001
        Tmax = int(6*np.pi/dt)
        Ct = np.zeros(Tmax)
        Rt = np.zeros(Tmax)
        Et = np.zeros(Tmax)
        for it in range(Tmax):
            TLSP.propagate(dt)
            for i in range(10): TLDMP.propagate(dt/10)
            #for (i,j) in [(0,0),(0,1),(1,0),(1,1)]:
            for (i,j) in [(0,0)]:
                Ct[it] = np.real(TLSP.C[i,0]*np.conj(TLSP.C[j,0]))
                Rt[it] = np.real(TLDMP.rhomatrix[0,0])
                Et[it] = 1.0-np.cos((it+1)*dt)**2
                #print i,j,CCC,RHO,1.0-np.cos(it*dt)**2
        fig, ax= plt.subplots(figsize=(6.0,4.0))

        ax.plot(Ct,label='Ct',lw=2,alpha=0.5)
        ax.plot(Rt,label='Rt',lw=2,alpha=0.5)
        ax.plot(Et,'-k',label='Et',alpha=0.5)
        ax.legend()
        plt.show()
if __name__ == '__main__':
    unittest.main()


