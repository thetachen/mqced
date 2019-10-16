#!/usr/bin/python
import unittest
import numpy as np
import sys
from utility import *
from units import AtomicUnit
from scipy.special import erf
AU = AtomicUnit()

from ZPE_sampler import *

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, MaxNLocator
import matplotlib.gridspec as gridspec

class TestZeroPointEnergy_1D(unittest.TestCase):
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
        self.beta = 1E0
        self.KCW = 0.25

        self.dK = 0.025
        self.Kmax = 1.0

    def test_sample(self):
        ZPE = ZeroPointEnergy_1D(self.param)

        T0 = 0
        Xt = ZPE.Amps*np.sin(ZPE.Ws*T0 + ZPE.Phis)
        Pt = ZPE.Amps*np.cos(ZPE.Ws*T0 + ZPE.Phis)

        # self.assertListEqual(list(ZPE.X0s), list(Xt))
        np.testing.assert_array_almost_equal(ZPE.X0s,Xt,decimal=6)
        np.testing.assert_array_almost_equal(ZPE.P0s,Pt,decimal=6)

    # def test_stdconverge(self):
    #     ZPE = ZeroPointEnergy_1D(self.param)
    #
    #     T0 = 0
    #     Xt = ZPE.Amps*np.sin(ZPE.Ws*T0 + ZPE.Phis)
    #     Pt = ZPE.Amps*np.cos(ZPE.Ws*T0 + ZPE.Phis)
    #
    #     # self.assertListEqual(list(ZPE.X0s), list(Xt))
    #     np.testing.assert_array_almost_equal(ZPE.X0s,Xt,decimal=6)
    #     np.testing.assert_array_almost_equal(ZPE.P0s,Pt,decimal=6)

    def test_getFields(self):
        boxsize = 2*np.pi
        ZPE = ZeroPointEnergy_1D(self.param,boxsize=boxsize)
        t = 0
        r = np.pi
        Etr,Btr = ZPE.getFields(t,r)

        Xt = ZPE.Amps*np.sin(ZPE.Ws*t + ZPE.Phis)
        Pt = ZPE.Amps*np.cos(ZPE.Ws*t + ZPE.Phis)

        Etr_Direct = np.sqrt(2.0/boxsize) * np.sum(ZPE.Ws*Xt*np.sin(ZPE.Ws*r))
        Btr_Direct = np.sqrt(2.0/boxsize) * np.sum(Pt*np.cos(ZPE.Ws*r))

        self.assertAlmostEqual(Etr,Etr_Direct, places=3)
        self.assertAlmostEqual(Btr,Btr_Direct, places=3)

        t = 0
        r = 0.0
        Etr,Btr = ZPE.getFields(t,r)
        self.assertAlmostEqual(Etr,0.0, places=3)
        
        t = 0
        r = 2*np.pi
        Etr,Btr = ZPE.getFields(t,r)
        self.assertAlmostEqual(Etr,0.0, places=3)

    #
    # def test_PlanckLight_Nmode(self):
    #     BZL = PlanckLight_Nmode(self.beta,self.dK,self.Kmax)
    #     ACWs = BZL.sample_ACW()
    #
    #     fig, ax= plt.subplots(1,figsize=(5.0,5.0))
    #     ax.plot(BZL.KCWs,BZL.ACWs,'-o')
    #     plt.show()

    #
    # def test_BoltzmannLight_1mode_calculate_ECW(self):
    #     TLbeta = BoltzmannLight_1mode(self.beta,self.KCW)
    #     TLbeta.sample_ACW()
    #
    # 	fig, ax= plt.subplots(1,figsize=(5.0,5.0))
    #
    #     t = 0.0
    #     TLbeta.calculate_ECW(self.param.Zgrid,t)
    #     ax.plot(self.param.Zgrid,TLbeta.ECWx)
    #     t = 10.0
    #     TLbeta.calculate_ECW(self.param.Zgrid,t)
    #     ax.plot(self.param.Zgrid,TLbeta.ECWx)
    #     t = 20.0
    #     TLbeta.calculate_ECW(self.param.Zgrid,t)
    #     ax.plot(self.param.Zgrid,TLbeta.ECWx)
    #
    #     plt.show()
if __name__ == '__main__':
    unittest.main()
