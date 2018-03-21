#!/usr/bin/python
import unittest
import numpy as np
import sys
sys.path.append("/home/theta/sync/tdsh/EM/Tao")
sys.path.append("/home/theta/sync/tdsh/EM/Tao/peakdetect")
from utility import *
from units import AtomicUnit
from scipy.special import erf
AU = AtomicUnit()

from MaxwellPropagator import *


import matplotlib.pyplot as plt
from matplotlib import cm,rc
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman', size='12')


class TestPropagator(unittest.TestCase):
    """
    a simple TLS propagator should give the same dynamics
    """
    def setUp(self):
        param_EM={
		    # Z coordinate grid parameters
    		'Zlim':     [-1700,1700],
    		'NZgrid':    200+1,
    		# ABC pararmeters
    		'Zo':       0.0,
    		'Z0':       2701.1,
    		'Z1':       2988.1,
    		# Initial pulse parameters
    		'B':        0.00015569585/60,
    		'K0':       2*16.46/AU.eV/AU.C * 4,
    		'U0':       6580/AU.eV * 0.225,
        }
        param_EM['Zgrid'] = np.linspace(param_EM['Zlim'][0],param_EM['Zlim'][1], num=param_EM['NZgrid'])
        param_EM['dZ'] = param_EM['Zgrid'][1]-param_EM['Zgrid'][0]
		# start index for [Ex,Ey,Bx,By]
        param_EM['_Ex'] = 0*param_EM['NZgrid']
        param_EM['_Ey'] = 1*param_EM['NZgrid']
        param_EM['_Bx'] = 2*param_EM['NZgrid']  
        param_EM['_By'] = 3*param_EM['NZgrid']
        param_EM=Struct(**param_EM)

        self.param = param_EM

    def test_propagate(self):
        param_EM=self.param
        # a long vector [Ex,Ey,Bx,By]
        EB = np.zeros(4*param_EM.NZgrid)
        Ax = np.zeros(param_EM.NZgrid)
        Ay = np.zeros(param_EM.NZgrid)

		# initialize EB and Dx,Dy function
        A = np.sqrt(param_EM.U0/np.sqrt(np.pi/2/param_EM.B)/(1+np.exp(-param_EM.K0**2/2/param_EM.B)))
        for iz in range(param_EM.NZgrid):
            Z = param_EM.Zgrid[iz]
            EB[param_EM._Ex + iz] = A*np.exp(-(Z)**2*param_EM.B)*np.cos(param_EM.K0*Z) / np.sqrt(AU.E0)
            EB[param_EM._Ey + iz] = EB[param_EM._Ex + iz]
            EB[param_EM._Bx + iz] =-A*np.exp(-(Z)**2*param_EM.B)*np.cos(param_EM.K0*Z) * np.sqrt(AU.M0)
            EB[param_EM._By + iz] =-EB[param_EM._Bx + iz]
            #EB[param_EM._By + iz] = A*np.exp(-(Z)**2*param_EM.B)*np.cos(param_EM.K0*Z) * np.sqrt(AU.M0)
            Ax[iz] = A* np.sqrt(AU.M0) * np.sqrt(np.pi/param_EM.B)/4 * np.exp(-param_EM.K0**2/param_EM.B/4) * \
            		 2*np.real( erf((2*param_EM.B*Z+1j*param_EM.K0)/2/np.sqrt(param_EM.B)) )
            Ay[iz] = Ax[iz]

		# create EM object
        EMP = SurfaceHoppingMaxwellPropagator_1D(param_EM)
        EMP.initializeODEsolver(EB,0.0)
        EMP.Ax = Ax
        EMP.Ay = Ay
        EMP.applyAbsorptionBoundaryCondition()

        T0 = 0.0
        Tmax = 40.0/AU.fs
        dt = 2E-3/AU.fs
        times = np.arange(T0, Tmax, dt)
        plt.ion()
        fig, ax= plt.subplots(2,figsize=(10.0,6.0))
        for it in range(len(times)):
            EMP.propagate(dt)
            EMP.update_AxAy(dt)
            EMP.applyAbsorptionBoundaryCondition()
			
            plt.sca(ax[0])
            plt.cla()
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.EB[param_EM._Bx:param_EM._Bx+EMP.NZgrid],alpha=0.5,color='blue',label='$B_x$')
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.EB[param_EM._By:param_EM._By+EMP.NZgrid],alpha=0.5,color='green',label='$B_y$')
            ax[0].plot(EMP.Zgrid,(EMP.dAxdZ),alpha=1,color='green',label='$\partial A_x/\partial z$',lw=2)
            ax[0].plot(EMP.Zgrid,-(EMP.dAydZ),alpha=1,color='blue',label='$-\partial A_y/\partial z$',lw=2)
            ax[0].legend()
#
            plt.sca(ax[1])
            plt.cla()
            ax[1].fill_between(EMP.Zgrid,0.0,(EMP.EB[param_EM._Ex:param_EM._Ex+EMP.NZgrid]),alpha=0.5,color='red',label='$E_x$')
            ax[1].fill_between(EMP.Zgrid,0.0,(EMP.EB[param_EM._Ey:param_EM._Ey+EMP.NZgrid]),alpha=0.5,color='orange',label='$E_y$')
            ax[1].plot(EMP.Zgrid,EMP.Ax,alpha=0.5,color='red',label='$A_x$')
            ax[1].plot(EMP.Zgrid,EMP.Ay,alpha=0.5,color='orange',label='$A_y$')
#
            ax[1].legend()
            fig.canvas.draw()        
        
if __name__ == '__main__':
    unittest.main()


