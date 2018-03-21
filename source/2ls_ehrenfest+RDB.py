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

    # a long vector [Dx,Dy,Bx,By]
    DB = np.zeros(4*param_EM.NZgrid)
    Ex = np.zeros(param_EM.NZgrid)
    Ey = np.zeros(param_EM.NZgrid)

    # wave function
    Ct = np.zeros((param_TLS.nstates,len(times)),complex)

    # total energy
    Ut = np.zeros((2,len(times)))

    # initialize Px,Py,dPxdz, dPydz, Dx, Dy
    for iz in range(param_EM.NZgrid):
        Z = param_EM.Zgrid[iz]    
        Px[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 )
        Py[iz] = 0.0
        dPxdz[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * -param_TLS.Sigma * (Z-param_TLS.Mu)*2
        dPydz[iz] = 0.0


	rho_12 = np.real(param_TLS.C0[0,0]*np.conj(param_TLS.C0[1,0]))
    Pxt = 2*rho_12*Px
    Pyt = 2*rho_12*Py
    for iz in range(param_EM.NZgrid):
        DB[param_EM._Ex + iz] = -Pxt[iz]
        DB[param_EM._Ey + iz] = -Pyt[iz]

    # create EM object
    EMP = EhrenfestPlusRDB_MaxwellPropagator_1D(param_EM)
    EMP.initializeODEsolver(DB,T0)
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
    ave_fft_Dx = None
    for it in range(len(times)):
    
        #0 Compute All integrals
        intPE = EMP.dZ*np.dot(Px, Ex) \
              + EMP.dZ*np.dot(Py, Ey)

        #intPD = EMP.dZ*np.dot(Px, np.array(DB[EMP._Dx:EMP._Dx+EMP.NZgrid])) \
              #+ EMP.dZ*np.dot(Py, np.array(DB[EMP._Dy:EMP._Dy+EMP.NZgrid]))
        #intPP = EMP.dZ*np.dot(Px, Px) \
              #+ EMP.dZ*np.dot(Py, Py)
        intdPdzB = EMP.dZ*np.dot(-dPydz, np.array(DB[EMP._Bx:EMP._Bx+EMP.NZgrid])) \
                 + EMP.dZ*np.dot( dPxdz, np.array(DB[EMP._By:EMP._By+EMP.NZgrid]))
        intdPdzdPdz = EMP.dZ*np.dot(dPxdz, dPxdz) \
                    + EMP.dZ*np.dot(dPydz, dPydz)

        #1. Propagate the wave function
        rho_12 = np.real(TLSP.C[0,0]*np.conj(TLSP.C[1,0]))
        Pxt = 2*rho_12*Px
        Pyt = 2*rho_12*Py
        curlPx = 2*rho_12*(-dPydz) 
        curlPy = 2*rho_12*( dPxdz)

        TLSP.update_coupling(intPE)
        TLSP.propagate(dt)
    
        #2. Compute curlP(t)
        #rho_12 = np.real(TLSP.C[0,0]*np.conj(TLSP.C[1,0]))
        #Pxt = 2*rho_12*Px
        #Pyt = 2*rho_12*Py
        #curlPx = 2*rho_12*(-dPydz) 
        #curlPy = 2*rho_12*( dPxdz)

        #3. Evolve the field
        EMP.update_curlP(curlPx,curlPy)
        EMP.propagate(dt)
        Ex = np.array(DB[EMP._Dx:EMP._Dx+EMP.NZgrid]) - Pxt
        Ey = np.array(DB[EMP._Dy:EMP._Dy+EMP.NZgrid]) - Pyt
        DB = EMP.DB

        if UsePlusEmission:
            #4. Implement additional population relaxation (1->0)
            gamma = TLSP.FGR[1,0]*np.abs(TLSP.C[1,0])**2
            drho = gamma*dt * np.abs(TLSP.C[1,0])**2
            dE = (TLSP.H0[1,1]-TLSP.H0[0,0])*gamma*dt * np.abs(TLSP.C[1,0])**2
            TLSP.rescale(1,0,drho)

            #if UseRandomEB:
                #theta = random()*2*np.pi
                #cos2, sin2  = np.cos(theta)**2, np.sin(theta)**2
                #EMP.ContinuousEmission_D(dE*cos2,intPD,intPP,Px)
                #EMP.ContinuousEmission_B(dE*sin2,intdPdzB,intdPdzdPdz,dPxdz)
            #else:
                #EMP.ContinuousEmission_B(dE,intdPdzB,intdPdzdPdz,dPxdz)
            EMP.ContinuousEmission_B(dE,intdPdzB,intdPdzdPdz,dPxdz)
            DB = EMP.DB

        #5. Apply absorption boundary condition 
        EMP.applyAbsorptionBoundaryCondition()
        DB = EMP.DB

        """
        output:
        
        """
        #population
        for n in range(param_TLS.nstates):
            Ct[n,it] = TLSP.C[n,0]
        #energy
        UDB = EMP.getEnergyDensity(Pxt,Pyt)
        Uele = TLSP.getEnergy()
        Uemf = np.sum(UDB)*param_EM.dZ
        Ut[0,it] = Uele
        Ut[1,it] = Uemf

        #spectrum 
        Dx= EMP.DB[EMP._Dx+(EMP.NZgrid-1)/2:EMP._Dx+(EMP.NZgrid-1)]
        fft_Dx = np.fft.rfft(Dx, n=len(Dx) *10)
        fft_Freq = np.array(range(len(fft_Dx))) * 2*np.pi/max(param_EM.Zlim) / 10
        if ave_fft_Dx is None:
            ave_fft_Dx = np.abs(fft_Dx)
            rolling = 1
        else:
            ave_fft_Dx = np.abs(fft_Dx)+ave_fft_Dx
            rolling += 1
        """
        Plot
        """
        if it%10==0 and ShowAnimation:
            plt.sca(ax[0])
            plt.cla()
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.DB[EMP._Bx:EMP._Bx+EMP.NZgrid],alpha=0.5,color='blue',label='$B_x$')
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.DB[EMP._By:EMP._By+EMP.NZgrid],alpha=0.5,color='green',label='$B_y$')
            ax[0].axvline(x=param_TLS.Mu, color='k', linestyle='--')
            ax[0].legend()

            plt.sca(ax[1])
            plt.cla()
            ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.DB[EMP._Dx:EMP._Dx+EMP.NZgrid]),alpha=0.5,color='red',label='$D_x$')
            ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.DB[EMP._Dy:EMP._Dy+EMP.NZgrid]),alpha=0.5,color='orange',label='$D_y$')
            ax[1].plot(EMP.Zgrid,np.sqrt(AU.E0)*(Ex),alpha=0.5,color='red',label='$E_x$')
            ax[1].plot(EMP.Zgrid,np.sqrt(AU.E0)*(Ey),alpha=0.5,color='orange',label='$E_y$')
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
            ax[3].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0],lw=2,label='ele energy')
            ax[3].plot(times[:it]*AU.fs,-(Ut[1,:it]-Ut[1,0]),lw=2,label='EM energy')
            #ax[3].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0]+Ut[1,:it]-Ut[1,0],lw=2,label='energy diff')
            ax[3].legend(loc='best')
            ax[3].set_xlim([0,Tmax*AU.fs])
    

            plt.sca(ax[4])
            plt.cla()
            ax[4].plot(fft_Freq*AU.C,np.abs(fft_Dx)**2,lw=2,color='b')
            ax[4].axvline(x=param_TLS.H0[1,1]-param_TLS.H0[0,0], color='k', linestyle='--')
            ax[4].plot(fft_Freq*AU.C,(ave_fft_Dx/rolling)**2,lw=2,color='r')
            ax[4].set_xlim([0,2])
            ax[4].set_xlabel('$ck$')

            fig.canvas.draw()


        #data dictionaray
        if it%AveragePeriod==0:
            output={
                'Zgrid':EMP.Zgrid,
                'times':times,
                'Dx':   EMP.DB[EMP._Dx:EMP._Dx+EMP.NZgrid],
                'Ex':   Ex,
                'UDB':  UDB,
                'Ct':  Ct,
                'fft_Dx':   fft_Dx,
                'fft_Freq': fft_Freq,
                'ave_fft_Dx':   ave_fft_Dx/rolling,
            }
            ave_fft_Dx = None

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
