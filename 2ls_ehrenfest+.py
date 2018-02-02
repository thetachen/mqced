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


#Default Options:
ShowAnimation = False
AveragePeriod = 10
UseInitialRandomPhase = True
NumberTrajectories = 1
UsePlusEmission = True
UseRandomEB = True

if (len(argv) == 1):
    execfile('param.in')
    outfile = 'data.pkl'
elif (len(argv) == 2):
    execfile(argv[1])
    outfile = 'data.pkl'
elif (len(argv) == 3):
    execfile(argv[1])
    outfile = argv[2]


def execute(param_EM,param_TLS,ShowAnimation=False):
    """
    Initialize
    """
    # dipole moment (Xi): Px, Py
    Px = np.zeros(param_EM.NZgrid)
    Py = np.zeros(param_EM.NZgrid)

    # curl P (for rescale B and keep div B =0)
    dPxdz = np.zeros(param_EM.NZgrid)
    dPydz = np.zeros(param_EM.NZgrid)

    # current: Jx, Jy
    Jx = np.zeros(param_EM.NZgrid)
    Jy = np.zeros(param_EM.NZgrid)

    # a long vector [Ex,Ey,Bx,By]
    EB = np.zeros(4*param_EM.NZgrid)

    # wave function
    Ct = np.zeros((param_TLS.nstates,len(times)),complex)

    # total energy
    Ut = np.zeros((2,len(times)))

    # energy change
    dEnergy = np.zeros((2,len(times)))

    # initialize Px,Py,dPxdz, dPydz
    for iz in range(param_EM.NZgrid):
        Z = param_EM.Zgrid[iz]    
        Px[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 )
        Py[iz] = 0.0
        dPxdz[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * -param_TLS.Sigma * (Z-param_TLS.Mu)*2
        dPydz[iz] = 0.0

    # create EM object
    EMP = EhrenfestPlusREB_MaxwellPropagator_1D(param_EM)
    EMP.initializeODEsolver(EB,T0)
    EMP.applyAbsorptionBoundaryCondition()

    # create TLS object
    if UseInitialRandomPhase:
        param_TLS.C0[1,0] = param_TLS.C0[1,0]*np.exp(1j*2*np.pi*random())
    TLSP = PureStatePropagator(param_TLS)
    #TLSP = DensityMatrixPropagator(param_TLS)

    # generate FGR rate 
    TLSP.FGR = np.zeros((TLSP.nstates,TLSP.nstates))
    for i in range(TLSP.nstates):
        for j in range(TLSP.nstates):
            TLSP.FGR[i,j] = (TLSP.H0[i,i]-TLSP.H0[j,j])*param_TLS.Pmax**2/AU.C/AU.E0 / AU.fs


    """
    Start Time Evolution
    """
    ave_fft_Ex = None
    for it in range(len(times)):
    
        #0 Compute All integrals
        intPE = EMP.dZ*np.dot(Px, np.array(EB[EMP._Ex:EMP._Ex+EMP.NZgrid])) \
              + EMP.dZ*np.dot(Py, np.array(EB[EMP._Ey:EMP._Ey+EMP.NZgrid]))
        intPP = EMP.dZ*np.dot(Px, Px) \
              + EMP.dZ*np.dot(Py, Py)
        intdPdzB = EMP.dZ*np.dot(-dPydz, np.array(EB[EMP._Bx:EMP._Bx+EMP.NZgrid])) \
                 + EMP.dZ*np.dot( dPxdz, np.array(EB[EMP._By:EMP._By+EMP.NZgrid]))
        intdPdzdPdz = EMP.dZ*np.dot(dPxdz, dPxdz) \
                    + EMP.dZ*np.dot(dPydz, dPydz)

        dEnergy[0,it] = TLSP.getEnergy()
        #1. Propagate the wave function
        TLSP.update_coupling(intPE)
        TLSP.propagate(dt)
        dEnergy[0,it] = dEnergy[0,it]-TLSP.getEnergy()

        #2. Compute Current: J
        dPdt = 0.0
        for i in range(param_TLS.nstates):
            for j in range(i+1,param_TLS.nstates):
                dPdt = dPdt + 2*(TLSP.H0[i,i]-TLSP.H0[j,j])*np.imag(TLSP.C[i,0]*np.conj(TLSP.C[j,0])) * TLSP.VP[i,j]
        Jx = dPdt * Px
        Jy = dPdt * Py
     
        #3. Evolve the field
        EMP.update_JxJy(Jx,Jy)
        EMP.propagate(dt)
        EB = EMP.EB

        if UsePlusEmission:
            #4. Implement additional population relaxation (1->0)
            gamma = TLSP.FGR[1,0]*np.abs(TLSP.C[1,0])**2
            drho = gamma*dt * np.abs(TLSP.C[1,0])**2
            dE = (TLSP.H0[1,1]-TLSP.H0[0,0])*gamma*dt * np.abs(TLSP.C[1,0])**2
            dEnergy[1,it] = dE
            TLSP.rescale(1,0,drho)

            if UseRandomEB:
                theta = random()*2*np.pi
                cos2, sin2  = np.cos(theta)**2, np.sin(theta)**2
                EMP.ContinuousEmission_E(dE*cos2,intPE,intPP,Px)
                EMP.ContinuousEmission_B(dE*sin2,intdPdzB,intdPdzdPdz,dPxdz)
            else:
                EMP.ContinuousEmission_B(dE,intdPdzB,intdPdzdPdz,dPxdz)
            EB = EMP.EB

        #5. Apply absorption boundary condition 
        EMP.applyAbsorptionBoundaryCondition()
        EB = EMP.EB

        """
        output:
        
        """
        #population
        for n in range(param_TLS.nstates):
            Ct[n,it] = TLSP.C[n,0]
        #energy
        UEB = EMP.getEnergyDensity()
        Uele = TLSP.getEnergy()
        Uemf = np.sum(UEB)*param_EM.dZ
        Ut[0,it] = Uele
        Ut[1,it] = Uemf

        #spectrum 
        Ex= EMP.EB[EMP._Ex+(EMP.NZgrid-1)/2:EMP._Ex+(EMP.NZgrid-1)]
        fft_Ex = np.fft.rfft(Ex, n=len(Ex) *10)
        fft_Freq = np.array(range(len(fft_Ex))) * 2*np.pi/max(param_EM.Zlim) / 10
        if ave_fft_Ex is None:
            ave_fft_Ex = np.abs(fft_Ex)
            rolling = 1
        else:
            ave_fft_Ex = np.abs(fft_Ex)+ave_fft_Ex
            rolling += 1
        """
        Plot
        """
        if it%10==0 and ShowAnimation:

            plt.sca(ax[0])
            plt.cla()
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.EB[EMP._Bx:EMP._Bx+EMP.NZgrid],alpha=0.5,color='blue',label='$B_x$')
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.EB[EMP._By:EMP._By+EMP.NZgrid],alpha=0.5,color='green',label='$B_y$')
            ax[0].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid]),alpha=0.5,color='red',label='$E_x$')
            ax[0].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ey:EMP._Ey+EMP.NZgrid]),alpha=0.5,color='orange',label='$E_y$')
            ax[0].axvline(x=param_TLS.Mu, color='k', linestyle='--')
            ax[0].legend()

            plt.sca(ax[1])
            plt.cla()
            #ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid]),alpha=0.5,color='red',label='$E_x$')
            #ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ey:EMP._Ey+EMP.NZgrid]),alpha=0.5,color='orange',label='$E_y$')
            ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(np.array(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid])**2+np.array(EMP.EB[EMP._By:EMP._By+EMP.NZgrid])**2),alpha=0.5,color='black',label='$E_x^2+B_y^2$')
            ax[1].set_ylim([0,0.0001])
            ax[1].axvline(x=param_TLS.Mu, color='k', linestyle='--')
            ax[1].legend()

            plt.sca(ax[2])
            plt.cla()
            for n in range(param_TLS.nstates):
                ax[2].plot(times[:it]*AU.fs,np.abs(Ct[n,:it])**2,'-',lw=2,label='$C_{d'+str(n)+'}(t)$')
            ax[2].set_xlim([0,Tmax*AU.fs])
            ax[2].set_xlabel('t [a.u.]')
            ax[2].legend(loc='best')


            plt.sca(ax[3])
            plt.cla()
            #ax[3].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0],lw=2,label='ele energy')
            #ax[3].plot(times[:it]*AU.fs,-(Ut[1,:it]-Ut[1,0]),lw=2,label='EM energy')
            ax[3].plot(times[:it]*AU.fs,dEnergy[0,:it], lw=2, label='coherent')
            ax[3].plot(times[:it]*AU.fs,dEnergy[1,:it], lw=2, label='incoherent')
            #ax[3].plot(times[:it]*AU.fs,(-np.log(dEnergy[1,:it])+np.log(dEnergy[1,0]))/times[:it], lw=2, label='incoherent')
            #ax[3].axhline(y=TLSP.FGR[1,0]*2, color='k', linestyle='--', lw=2)
            #ax[3].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0]+Ut[1,:it]-Ut[1,0],lw=2,label='energy diff')
            ax[3].legend(loc='best')
            ax[3].set_xlim([0,Tmax*AU.fs])
    

            plt.sca(ax[4])
            plt.cla()
            ax[4].plot(fft_Freq*AU.C,np.abs(fft_Ex)**2,lw=2,color='b')
            ax[4].axvline(x=param_TLS.H0[1,1]-param_TLS.H0[0,0], color='k', linestyle='--')
            ax[4].plot(fft_Freq*AU.C,(ave_fft_Ex/rolling)**2,lw=2,color='r')
            ax[4].set_xlim([0,2])
            ax[4].set_xlabel('$ck$')

            fig.canvas.draw()


        #data dictionaray
        if it%AveragePeriod==0:
            output={
                'Zgrid':EMP.Zgrid,
                'times':times,
                'Ex':   EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid],
                'By':   EMP.EB[EMP._By:EMP._By+EMP.NZgrid],
                'UEB':  UEB,
                'dE':   dEnergy,
                'Ct':   Ct,
                'fft_Ex':   fft_Ex,
                'fft_Freq': fft_Freq,
                'ave_fft_Ex':   ave_fft_Ex/rolling,
            }
            ave_fft_Ex = None

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
    output = execute(param_EM,param_TLS,ShowAnimation=ShowAnimation)
    data.append(output)
with open(outfile, 'wb') as f:
    pickle.dump(data,f)
