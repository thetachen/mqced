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
AveragePeriod = 1
UseInitialRandomPhase = True
NumberTrajectories = 1
UsePlusEmission = True
UseRandomEB = False

if (len(argv) == 1):
    execfile('param.in')
    outfile = 'data.pkl'
elif (len(argv) == 2):
    execfile(argv[1])
    outfile = 'data.pkl'
elif (len(argv) == 3):
    execfile(argv[1])
    outfile = argv[2]

def initialize(param_EM,param_TLS):
    """
    Initialize
    """
    # static dipole moment of two sites (Mu_S): Sx, Sy
    Sx = np.zeros(param_EM.NZgrid)
    Sy = np.zeros(param_EM.NZgrid)

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

    # a long vector [Ex,Ey,Bx,By]
    EB = np.zeros(4*param_EM.NZgrid)

    # initialize Px,Py,dPxdz, dPydz, d2Pxdz, d2Pydz, Sx, Sy
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

        Sx[iz] = param_TLS.Smax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu_S)**2 )
        Sy[iz] = 0.0

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
    TBy = -dPxdz - (1.0/6/param_TLS.Sigma)*d3Pxdz

    # create EM object
    EMP = EhrenfestPlusREB_MaxwellPropagator_1D(param_EM)
    EMP.initializeODEsolver(EB,T0)
    EMP.update_TETB(TEx,TEy,TBx,TBy)
    EMP.Sx = Sx
    EMP.Sy = Sy
    EMP.Px = Px
    EMP.Py = Py
    EMP.dPxdz = dPxdz
    EMP.dPydz = dPydz
    EMP.applyAbsorptionBoundaryCondition()

    #normalized
    TEx = TEx/np.sqrt(EMP.TE2)
    TBy = TBy/np.sqrt(EMP.TB2)
    EMP.update_TETB(TEx,TEy,TBx,TBy)

    # create TLS object
    if UseInitialRandomPhase:
        param_TLS.C0[1,0] = param_TLS.C0[1,0]*np.exp(1j*2*np.pi*random())
    if Describer == 'vector':
        # TLSP = PureStatePropagator(param_TLS)
        TLSP = PureStatePropagator_Transform(param_TLS)
    if Describer == 'density':
        TLSP = DensityMatrixPropagator(param_TLS)
    #TLSP = FloquetStatePropagator(param_TLS,param_EM,dt)

    TLSP.VS = param_TLS.VS
    return EMP, TLSP

def execute(EMP,TLSP,ShowAnimation=False):
    # current: Jx, Jy
    Jx = np.zeros(param_EM.NZgrid)
    Jy = np.zeros(param_EM.NZgrid)


    # density matrix
    rhot = np.zeros((param_TLS.nstates,param_TLS.nstates,len(times)),complex)

    # density matrix
    rhot_trans = np.zeros((param_TLS.nstates,param_TLS.nstates,len(times)),complex)

    # total energy
    Ut = np.zeros((2,len(times)))

    # average current
    Jt = np.zeros(len(times))

    # energy change
    dEnergy = np.zeros((2,len(times)))
    """
    Start Time Evolution
    """
    for it in range(len(times)):

        #0 Compute All integrals
        intSE = EMP.dZ*np.dot(EMP.Sx, np.array(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid])) \
              + EMP.dZ*np.dot(EMP.Sy, np.array(EMP.EB[EMP._Ey:EMP._Ey+EMP.NZgrid]))
        intPE = EMP.dZ*np.dot(EMP.Px, np.array(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid])) \
              + EMP.dZ*np.dot(EMP.Py, np.array(EMP.EB[EMP._Ey:EMP._Ey+EMP.NZgrid]))
        intdPdzB = EMP.dZ*np.dot(-EMP.dPydz, np.array(EMP.EB[EMP._Bx:EMP._Bx+EMP.NZgrid])) \
                 + EMP.dZ*np.dot( EMP.dPxdz, np.array(EMP.EB[EMP._By:EMP._By+EMP.NZgrid]))

        #0.5 polarization interact with CW
        # ECWx = param_EM.A_CW*np.cos(param_EM.K_CW*(param_EM.Zgrid-AU.C*it*dt))
        # ECWy = np.zeros(len(param_EM.Zgrid))
        # BCWx = np.zeros(len(param_EM.Zgrid))
        # BCWy = param_EM.A_CW*np.sin(param_EM.K_CW*(param_EM.Zgrid-AU.C*it*dt))
        #0.5 polarization interact with CW
        ECWx = param_In.AIN*np.exp(-(param_EM.Zgrid-(param_In.start+AU.C*it*dt))**2/param_In.width)*\
               np.cos(param_In.KIN*(param_EM.Zgrid-AU.C*it*dt))
        ECWy = np.zeros(len(param_EM.Zgrid))
        BCWx = np.zeros(len(param_EM.Zgrid))
        BCWy = param_In.AIN*np.exp(-(param_EM.Zgrid-(param_In.start+AU.C*it*dt))**2/param_In.width)*\
               np.sin(param_In.KIN*(param_EM.Zgrid-AU.C*it*dt))
        intSE += EMP.dZ*np.dot(EMP.Sx,ECWx) \
               + EMP.dZ*np.dot(EMP.Sy,ECWy)
        intPE += EMP.dZ*np.dot(EMP.Px,ECWx) \
               + EMP.dZ*np.dot(EMP.Py,ECWy)
        intdPdzB += EMP.dZ*np.dot(-EMP.dPydz, BCWx ) \
                  + EMP.dZ*np.dot( EMP.dPxdz, BCWy )

        Imrho12 = np.imag(TLSP.rho[0,1])
        # print 'Rerho/absrho=',(np.real(TLSP.rho[0,1])/np.abs(TLSP.rho[0,1])),\
              # 'Imrho/absrho=',(np.imag(TLSP.rho[0,1])/np.abs(TLSP.rho[0,1]))
        if np.abs(TLSP.rho[0,1])==0.0:
            # angle = phase_shift
            angle = (TLSP.H0[1,1]-TLSP.H0[0,0])*it*dt + phase_shift
        else:
            angle = np.angle(TLSP.rho[0,1]/np.abs(TLSP.rho[0,1])) + phase_shift
        sign = np.sin(angle)
        # sign = np.imag(np.exp(1j*shift))
        dEnergy[0,it] = TLSP.getEnergy()

        #1. Propagate the wave function
        # TLSP.update_coupling(intPE)
        # TLSP.update_static_dipole(intSE)
        TLSP.update_coupling(intPE,intSE)
        TLSP.propagate(dt)
        dEnergy[0,it] = dEnergy[0,it]-TLSP.getEnergy()

        #2. Compute Current: J
        # dPdt = 0.0
        # for i in range(param_TLS.nstates):
        #     for j in range(i+1,param_TLS.nstates):
        #         dPdt = dPdt + 2*(TLSP.H0[i,i]-TLSP.H0[j,j])*np.imag(TLSP.rho[i,j]) * TLSP.VP[i,j]
        # dPdt = 2*(TLSP.H0[0,0]-TLSP.H0[1,1])*np.imag(TLSP.rho[0,1]) * TLSP.VP[0,1]
        # dSdt = 4*TLSP.H0[0,1]*np.imag(TLSP.rho[0,1])*TLSP.VS[0,0]
        #
        # Jx = dPdt * EMP.Px - dSdt * EMP.Sx
        # Jy = dPdt * EMP.Py - dSdt * EMP.Sy

        dXdt = -2*(TLSP.H0[1,1]-TLSP.H0[0,0])*np.imag(TLSP.rho[0,1])
        Jx = dXdt * (TLSP.VP[0,1]*EMP.Px +TLSP.VS[0,1]*EMP.Sx)
        Jy = dXdt * (TLSP.VP[0,1]*EMP.Py +TLSP.VS[0,1]*EMP.Sy)

        #3. Evolve the field
        EMP.update_JxJy(Jx,Jy)
        EMP.propagate(dt)
        #EB = EMP.EB
        if UsePlusEmission:
            #4. Implement additional population relaxation (1->0)
            # drho, dE = TLSP.getComplement(1,0,dt)
            # drho, dE = TLSP.getComplement_angle(1,0,dt,angle)
            # TLSP.rescale(1,0,drho)
            kRdt,kDdt,drho,dE = TLSP.getComplement_angle(1,0,dt,angle)
            TLSP.relaxation(1,0,kRdt)
            # TLSP.dephasing(1,0,kDdt)
            # TLSP.resetting(1,0,TLSP.FGR[1,0]*dt) # only work if the TLS is density matrix

            EMP.MakeTransition_sign(dE*dt/Lambda,sign,UseRandomEB=UseRandomEB)
            dEnergy[1,it] = dE

        #5. Apply absorption boundary condition
        EMP.applyAbsorptionBoundaryCondition()
        #EB = EMP.EB

        # save far field out of the box
        EMP.saveFarField(times[it],Tmax)
        """
        output:

        """
        # get unitary transformation
        # sin2 = param_TLS.H0[0,1]/np.sqrt( 0.25*(param_TLS.H0[1,1]-param_TLS.H0[0,0])**2+(param_TLS.H0[0,1])**2 )
        # cos2 = 0.5*(param_TLS.H0[1,1]-param_TLS.H0[0,0])/np.sqrt( 0.25*(param_TLS.H0[1,1]-param_TLS.H0[0,0])**2+(param_TLS.H0[0,1])**2 )
        # cos = np.sqrt((1+cos2)/2)
        # sin = np.sqrt((1-cos2)/2)
        # trans = np.array([[cos,-sin],[sin,cos]])
        TLSP.rho_trans =np.dot(TLSP.trans,np.dot(TLSP.rho,TLSP.trans.T))
        # density matrix
        for i in range(param_TLS.nstates):
            for j in range(param_TLS.nstates):
                rhot[i,j,it]=TLSP.rho[i,j]
                rhot_trans[i,j,it]=TLSP.rho_trans[i,j]
        #energy
        # UEB = EMP.getEnergyDensity()
        # Uemf = np.sum(UEB)*param_EM.dZ
        Uemf = EMP.getTotalEnergy(dt)
        Uele = TLSP.getEnergy()
        Ut[0,it] = Uele
        Ut[1,it] = Uemf
        Jt[it] = dXdt
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
            ax[0].axvline(x=param_TLS.Mu, color='k', linestyle='--')
            ax[0].legend()

            plt.sca(ax[1])
            plt.cla()
            #ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid]),alpha=0.5,color='red',label='$E_x$')
            #ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ey:EMP._Ey+EMP.NZgrid]),alpha=0.5,color='orange',label='$E_y$')
            # ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(np.array(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid])**2+np.array(EMP.EB[EMP._By:EMP._By+EMP.NZgrid])**2),alpha=0.5,color='black',label='$E_x^2+B_y^2$')
            # ax[1].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0],lw=2,label='ele energy')
            # ax[1].plot(times[:it]*AU.fs,-(Ut[1,:it]-Ut[1,0]),lw=2,label='EM energy')
            # ax[1].plot(times[:it]*AU.fs,Ut[0,:it]+Ut[1,:it]-(Ut[0,0]+Ut[1,0]),lw=2,label='Uele+Uemf')
            ax[1].plot(times[:it]*AU.fs,Jt[:it],lw=2,label='Jt')
            # ax[1].set_ylim([0,0.0001])
            # ax[1].axvline(x=param_TLS.Mu, color='k', linestyle='--')
            ax[1].legend()

            plt.sca(ax[2])
            plt.cla()
            # for n in range(param_TLS.nstates):
            for n in [1]:
                ax[2].plot(times[:it]*AU.fs,np.abs(rhot[n,n,:it]),'-',lw=2,label='$P_{'+str(n)+'}$')
                ax[2].plot(times[:it]*AU.fs,np.abs(rhot_trans[n,n,:it]),'-',lw=2,label='$P_{'+str(n)+'}$')
            ax[2].plot(times[:it]*AU.fs,rhot[1,1,0]*np.exp(-TLSP.FGR[1,0]*times[:it]),'--k',lw=2,label='FGR')
            # ax[2].plot(times[:it]*AU.fs,np.abs(param_TLS.C0[1,0])**2*np.exp(-KFGR*times[:it]),'--k',lw=2,label='FGR')
            # ax[2].plot(times[:it]*AU.fs,np.real(rhot[0,1,:it]),'-',lw=2,label='$\mathrm{Re}\rho_{01}$')
            # ax[2].plot(times[:it]*AU.fs,np.imag(rhot[0,1,:it]),'-',lw=2,label='$\mathrm{Im}\rho_{01}$')
            # ax[2].plot(times[:it]*AU.fs,np.real(rhot[0,1,:it]),'-',lw=2,label=r'$|\rho_{01}|$')
            # rho12 = np.sqrt(1.0-np.abs(param_TLS.C0[1,0])**2*np.exp(-KFGR*times[:it]))*np.abs(param_TLS.C0[1,0])*np.exp(-KFGR*times[:it]/2)
            # rho12 = np.sqrt(1.0-np.abs(param_TLS.C0[1,0])**2)*np.abs(param_TLS.C0[1,0])*np.exp(-KFGR*times[:it]/2)
            # ax[2].plot(times[:it]*AU.fs,rho12,'--k',lw=2,label='WW')
            # ax[2].plot(times[:it]*AU.fs,np.cos(0.25*times[:it]-(1.9)*np.pi)*rho12,'--b',lw=2,label='WW')
            # ax[2].plot(times[:it]*AU.fs,np.sin(0.25*times[:it]-(1.9)*np.pi)*rho12,'--g',lw=2,label='WW')
            # ax[2].set_xlim([0,Tmax*AU.fs])
            ax[2].set_xlabel('t')
            ax[2].legend(loc='best')


            plt.sca(ax[3])
            plt.cla()
            #ax[3].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0],lw=2,label='ele energy')
            #ax[3].plot(times[:it]*AU.fs,-(Ut[1,:it]-Ut[1,0]),lw=2,label='EM energy')
            #ax[3].plot(times[:it]*AU.fs,dEnergy[0,:it], lw=2, label='coherent')
            #ax[3].plot(times[:it]*AU.fs,dEnergy[1,:it], lw=2, label='incoherent')
            #ax[3].plot(times[:it]*AU.fs,(-np.log(dEnergy[1,:it])+np.log(dEnergy[1,0]))/times[:it], lw=2, label='incoherent')
            #ax[3].axhline(y=TLSP.FGR[1,0]*2, color='k', linestyle='--', lw=2)
            #ax[3].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0]+Ut[1,:it]-Ut[1,0],lw=2,label='energy diff')
            ax[3].plot(EMP.Xs,np.array(EMP.Es),label='$E_x$ (far)')
            ax[3].plot(EMP.Xs,np.array(EMP.Bs),label='$B_y$ (far)')
            # ax[3].plot(EMP.Xs,np.array(EMP.Es)**2,lw=1,label='$E^2$')
            # ax[3].plot(EMP.Xs,np.array(EMP.Bs)**2,lw=1,label='$B^2$')
            ax[3].legend(loc='best')
            # ax[3].set_xlim([0,Tmax*AU.fs])
            # ax[3].set_ylim([-max(EMP.Es),max(EMP.Es)])
            ax[3].set_xlabel('x')
            ax[3].legend()

            plt.sca(ax[4])
            plt.cla()
            #ax[4].plot(fft_Freq*AU.C,np.abs(fft_Ex)**2,lw=2,color='b')
            normal = np.sqrt( (param_TLS.H0[1,1]-param_TLS.H0[0,0])**2+(2*param_TLS.H0[0,1])**2 )
            ax[4].axvline(x=normal, color='k', linestyle='--')
            #ax[4].plot(fft_Freq*AU.C,(ave_fft_Ex/rolling)**2,lw=2,color='r')
            ax[4].set_xlim([0.0,1.0])
            #ax[4].set_xlabel('$ck$')
            truncate = 10000
            if len(EMP.Es)<truncate:
                truncate = len(EMP.Es)
            fft_Ex = np.fft.rfft(EMP.Es[::-1][:truncate])
            fft_Freq = np.array(range(len(fft_Ex))) * 2*np.pi /(EMP.Xs[::-1][truncate-1]-EMP.Xs[-1])
            ax[4].plot(fft_Freq,np.abs(fft_Ex)**2,lw=1,color='b',label='fft($E$)')
            ax[4].set_xlabel('$\omega$')
            ax[4].legend()
            # fig.savefig('test_plot.pdf')
            fig.canvas.draw()


        #data dictionaray
        if it%AveragePeriod==0 or it==len(times)-1:
            output={
                'Zgrid':EMP.Zgrid,
                'times':times,
                'Ex':   EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid],
                'By':   EMP.EB[EMP._By:EMP._By+EMP.NZgrid],
                'Es':   EMP.Es,
                'Bs':   EMP.Bs,
                'Xs':   EMP.Xs,
                'Ut':   Ut,
                'dE':   dEnergy,
                'rhot':   rhot,
            }

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

EMP,TLSP = initialize(param_EM,param_TLS)
output = execute(EMP,TLSP,ShowAnimation=ShowAnimation)

# data = []
# for i in range(NumberTrajectories):
    # output = execute(param_EM,param_TLS,ShowAnimation=ShowAnimation)
    # data.append(output)
# with open(outfile, 'wb') as f:
    # pickle.dump(data,f)

data = execute(param_EM,param_TLS,ShowAnimation=ShowAnimation)
np.savetxt("Es.dat",zip(np.array(data['Xs'][::-1]),np.array(data['Es'][::-1])))
rhot={
    'times':    data['times'],
    'rhot':     data['rhot'],
}
with open("rhot.pkl", 'wb') as f:
    pickle.dump(rhot,f)
