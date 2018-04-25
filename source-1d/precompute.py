#!/usr/bin/python
from sys import argv
import numpy as np
from scipy.integrate import ode
from utility import *
from random import random
import cPickle as pickle

from units import AtomicUnit
AU = AtomicUnit()

from MaxwellPropagator import *
from SystemPropagator import *




if (len(argv) == 1):
    execfile('param.in')
    outfile = 'data.pkl'
elif (len(argv) == 2):
    execfile(argv[1])
    outfile = 'data.pkl'
elif (len(argv) == 3):
    execfile(argv[1])
    outfile = argv[2]

#Default Options:
ShowAnimation = True
AveragePeriod = 2
UseInitialRandomPhase = True
NumberTrajectories = 1
UsePlusEmission = True
UseRandomEB = False

def precompute(param_EM,param_TLS,ShowAnimation=False):
    """
    Initialize
    """
    # dipole moment (Xi): Px, Py
    Px = np.zeros(param_EM.NZgrid)
    Py = np.zeros(param_EM.NZgrid)

    # curl P (for rescale B and keep div B =0)
    dPxdz = np.zeros(param_EM.NZgrid)
    dPydz = np.zeros(param_EM.NZgrid)

    # curl (curl P) (for rescale E)
    d2Pxdz = np.zeros(param_EM.NZgrid)
    d2Pydz = np.zeros(param_EM.NZgrid)

    d3Pxdz = np.zeros(param_EM.NZgrid)
    d3Pydz = np.zeros(param_EM.NZgrid)
    d4Pxdz = np.zeros(param_EM.NZgrid)
    d4Pydz = np.zeros(param_EM.NZgrid)

    # TE and TB
    TEx = np.zeros(param_EM.NZgrid)
    TEy = np.zeros(param_EM.NZgrid)
    TBx = np.zeros(param_EM.NZgrid)
    TBy = np.zeros(param_EM.NZgrid)

    # current: Jx, Jy
    Jx = np.zeros(param_EM.NZgrid)
    Jy = np.zeros(param_EM.NZgrid)

    # a long vector [Ex,Ey,Bx,By]
    EB = np.zeros(4*param_EM.NZgrid)

    # density matrix
    rhot = np.zeros((param_TLS.nstates,param_TLS.nstates,len(times)),complex)

    # total energy
    Ut = np.zeros((2,len(times)))

    # energy change
    Overlap = np.zeros((2,len(times)))

    # initialize Px,Py,dPxdz, dPydz, d2Pxdz, d2Pydz
    for iz in range(param_EM.NZgrid):
        Z = param_EM.Zgrid[iz]
        Px[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 )
        Py[iz] = 0.0
        dPxdz[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * -param_TLS.Sigma * (Z-param_TLS.Mu)*2
        dPydz[iz] = 0.0
        d2Pxdz[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * \
					  (-param_TLS.Sigma *2) \
					+ param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * \
					  (-param_TLS.Sigma * (Z-param_TLS.Mu)*2) * \
					  (-param_TLS.Sigma * (Z-param_TLS.Mu)*2)
        d2Pydz[iz] = 0.0
        # d3Pxdz[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * \
		# 			  (-param_TLS.Sigma *2) * \
        #               (-param_TLS.Sigma * (Z-param_TLS.Mu)*2)\
		# 			+ param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * \
        #               (-param_TLS.Sigma * (Z-param_TLS.Mu)*2) * \
		# 			  (-param_TLS.Sigma * (Z-param_TLS.Mu)*2) * \
		# 			  (-param_TLS.Sigma * (Z-param_TLS.Mu)*2)\
		# 			+ param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * \
		# 			  (-param_TLS.Sigma *2) * \
		# 			  (-param_TLS.Sigma * (Z-param_TLS.Mu)*2) *2
        d3Pxdz[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * \
					  (12*(param_TLS.Sigma**2)*Z -8*(param_TLS.Sigma**3)*Z**3)
        d3Pydz[iz] = 0.0
        d4Pxdz[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * \
					  (12*(param_TLS.Sigma**2) -48*(param_TLS.Sigma**3)*Z**2 +16*(param_TLS.Sigma**4)*Z**4)
        d4Pydz[iz] = 0.0

    # Just rescale on Px
	# TEx =-Px
    # Transverse
    # TEx =-ddPxddz
    # TEx = ddPxddz
    # Make intTEE zero
	# TEx = -ddPxddz + np.dot(ddPxddz, Px)/np.dot(Px, Px)*Px
    """
    Direction of rescaling in E and B
    we add additional term to minimize the interference
    1. E filed: let TE(0)=0
        TE = -curl^2(P) + 2a*P
    2. B field: let dTBdz(0) = 0 (flat near the center)
        TB = -curl(P) + (1/6a)*curl^3(P)
    """
    TEx = d2Pxdz + (2*param_TLS.Sigma)*Px
    # TEx = d2Pxdz + d4Pxdz *(1.0/6/param_TLS.Sigma)
    TBy = -dPxdz  - (1.0/6/param_TLS.Sigma)*d3Pxdz


    # create EM object
    EMP = EhrenfestPlusREB_MaxwellPropagator_1D(param_EM)
    EMP.initializeODEsolver(EB,T0)
    EMP.update_TETB(TEx,TEy,TBx,TBy)
    EMP.applyAbsorptionBoundaryCondition()

    #normalized
    TEx = TEx/np.sqrt(EMP.TE2)
    TBy = TBy/np.sqrt(EMP.TB2)
    EMP.update_TETB(TEx,TEy,TBx,TBy)

    print 'EMP.TE2', EMP.TE2, 'EMP.TB2', EMP.TB2
    """
    Start Time Evolution
    """
    for it in range(len(times)):
        # EMP.propagate(dt)

        dE = 1E-3
        sign = 1.0
        alpha,beta=EMP.MakeTransition_sign(dE,sign,UseRandomEB=False)
        print alpha,beta
        #3. Evolve the field
        EMP.propagate(dt)

        #5. Apply absorption boundary condition
        EMP.applyAbsorptionBoundaryCondition()
        #EB = EMP.EB

        # save far field out of the box
        EMP.saveFarField(times[it],Tmax)

        overlap_E = EMP.dZ*np.dot( alpha*EMP.TEx, np.array(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid]))
        overlap_B = EMP.dZ*np.dot( beta*EMP.TBy, np.array(EMP.EB[EMP._By:EMP._By+EMP.NZgrid]))
        Overlap[0,it] = overlap_E
        Overlap[1,it] = overlap_B
        """
        output:
        """

        """
        Plot
        """
        if it%AveragePeriod==0 and ShowAnimation:
            plt.sca(ax[0])
            plt.cla()
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.EB[EMP._Bx:EMP._Bx+EMP.NZgrid],alpha=0.5,color='blue',label='$B_x$')
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.EB[EMP._By:EMP._By+EMP.NZgrid],alpha=0.5,color='green',label='$B_y$')
            ax[0].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid]),alpha=0.5,color='red',label='$E_x$')
            ax[0].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ey:EMP._Ey+EMP.NZgrid]),alpha=0.5,color='orange',label='$E_y$')
            ax[0].plot(EMP.Zgrid,EMP.TEx,'--',color='red',lw=2)
            ax[0].plot(EMP.Zgrid,EMP.TBy,'--',color='green',lw=2)
            ax[0].axvline(x=param_TLS.Mu, color='k', linestyle='--')
            ax[0].legend()

            plt.sca(ax[1])
            plt.cla()
            #ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid]),alpha=0.5,color='red',label='$E_x$')
            #ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ey:EMP._Ey+EMP.NZgrid]),alpha=0.5,color='orange',label='$E_y$')
            ax[1].fill_between(EMP.Zgrid,0.0,(np.array(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid])**2+np.array(EMP.EB[EMP._By:EMP._By+EMP.NZgrid])**2),alpha=0.5,color='black',label='$E_x^2+B_y^2$')
            # ax[1].set_ylim([0,0.0001])
            ax[1].axvline(x=param_TLS.Mu, color='k', linestyle='--')
            ax[1].legend()

            plt.sca(ax[2])
            plt.cla()
            ax[2].plot(times[:it],2*Overlap[0,:it]/EMP.TE2/alpha**2 *dt,lw=2,label='overlap: E')
            ax[2].plot(times[:it],2*Overlap[1,:it]/EMP.TB2/beta**2  *dt,lw=2,label='overlap: B')
            # Lambda =(Overlap[1]+Overlap[0])/(EMP.TE2*alpha**2+EMP.TB2*beta**2)  *dt
            # Lambda = 2*Overlap[0]/EMP.TE2/alpha**2 *dt
            # Lambda = Overlap[0]/EMP.TE2/alpha**2 *dt + Overlap[1]/EMP.TB2/beta**2  *dt
            Lambda = (Overlap[0]/EMP.TE2/alpha**2 +0.5)*dt + (Overlap[1]/EMP.TB2/beta**2 +0.5)*dt
            print Lambda[it],1.0/Lambda[it]
            ax[2].plot(times[:it],Lambda[:it],lw=2,label='$\lambda$')
            ax[2].legend(loc='best')


            plt.sca(ax[3])
            plt.cla()
            #ax[3].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0],lw=2,label='ele energy')
            #ax[3].plot(times[:it]*AU.fs,-(Ut[1,:it]-Ut[1,0]),lw=2,label='EM energy')
            #ax[3].plot(times[:it]*AU.fs,dEnergy[0,:it], lw=2, label='coherent')
            #ax[3].plot(times[:it]*AU.fs,dEnergy[1,:it], lw=2, label='incoherent')
            #ax[3].plot(times[:it]*AU.fs,(-np.log(dEnergy[1,:it])+np.log(dEnergy[1,0]))/times[:it], lw=2, label='incoherent')
            #ax[3].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0]+Ut[1,:it]-Ut[1,0],lw=2,label='energy diff')
            ax[3].plot(EMP.Xs,np.array(EMP.Es)**2,lw=1,label='$E^2$')
            ax[3].plot(EMP.Xs,np.array(EMP.Bs)**2,lw=1,label='$B^2$')
            ax[3].legend(loc='best')
            # ax[3].set_xlim([0,Tmax*AU.fs])
            # ax[3].set_ylim([-max(EMP.Es),max(EMP.Es)])
            ax[3].set_xlabel('x')
            ax[3].legend()

            plt.sca(ax[4])
            plt.cla()
            fft_Ex = np.fft.rfft(EMP.Es[::-1])
            fft_Freq = np.array(range(len(fft_Ex))) * 2*np.pi /(EMP.Xs[0]-EMP.Xs[-1])
            ax[4].plot(fft_Freq,np.abs(fft_Ex)**2,lw=1,color='b',label='fft($E$)')
            ax[4].set_xlabel('$\omega$')
            ax[4].legend()
            # fig.savefig('test_plot.pdf')
            fig.canvas.draw()


    """
    End of Time Evolution
    """

    return output

if ShowAnimation:
	from matplotlib import pyplot as plt
	from matplotlib.ticker import MultipleLocator, AutoLocator, MaxNLocator
	import matplotlib.gridspec as gridspec
	plt.rc('text', usetex=True)
	plt.rc('font', family='Times New Roman', size='12')
	plt.ion()
	fig, ax= plt.subplots(5,figsize=(10.0,12.0))

data = []
for i in range(NumberTrajectories):
    output = precompute(param_EM,param_TLS,ShowAnimation=ShowAnimation)
    data.append(output)
with open(outfile, 'wb') as f:
    pickle.dump(data,f)
