#!/usr/bin/python
import unittest
import numpy as np
import sys
from utility import *
from units import AtomicUnit
from scipy.special import erf
AU = AtomicUnit()

from ThermalSampler import *

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, MaxNLocator
import matplotlib.gridspec as gridspec

class TestThermalLight_1D(unittest.TestCase):
    def setUp(self):
        param_EM={
		    # Z coordinate grid parameters
    		'Zlim':     [-8.0,8.0],
    		'NZgrid':   80*2*8,
    		# ABC pararmeters
    		'Zo':       0.0,
    		'Z0':       13.0/2,
    		'Z1':       15.0/2,
            # cavity boundary
            'Zb0':      4.0,
            'Zb1':      4.1,
            'refract':  10.0, # the n
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
        self.beta = 1.0

    def test_sample(self):
        TLbeta = ThermalLight_1D(self.beta,num_k=1)

        num_sample = 100000
        X2, P2 = 0, 0
        for i in range(num_sample):
            Xk,Pk = TLbeta.sample()
            X2 += Xk**2
            P2 += Pk**2
        energy = (X2+P2)/num_sample/2

        energy_Direct = 1.0/(np.abs(TLbeta.Ks[0])*TLbeta.beta)
        self.assertAlmostEqual(energy, energy_Direct, places=3)
    def test_getEr(self):
        TLbeta = ThermalLight_1D(self.beta)
        TLbeta.sample()

    	fig, ax= plt.subplots(1,figsize=(5.0,5.0))

        t = 0.0
        ECW = TLbeta.getEr(self.param.Zgrid,t)
        ax.plot(self.param.Zgrid,ECW)
        t = 1.0
        ECW = TLbeta.getEr(self.param.Zgrid,t)
        ax.plot(self.param.Zgrid,ECW)
        t = 2.0
        ECW = TLbeta.getEr(self.param.Zgrid,t)
        ax.plot(self.param.Zgrid,ECW)

        plt.show(block=False)


if __name__ == '__main__':
    unittest.main()
