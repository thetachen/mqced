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
NumberTrajectories = 1

UseInitialRandomPhase = True
UsePlusEmission = True
UseRandomEB = True
UseThermalRelax = True

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

    # curl (curl P) (for rescale E)
    ddPxddz = np.zeros(param_EM.NZgrid)
    ddPyddz = np.zeros(param_EM.NZgrid)

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

    # initialize Px,Py,dPxdz, dPydz, ddPxddz, ddPyddz
    for iz in range(param_EM.NZgrid):
        Z = param_EM.Zgrid[iz]
        Px[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 )
        Py[iz] = 0.0
        dPxdz[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * -param_TLS.Sigma * (Z-param_TLS.Mu)*2
        dPydz[iz] = 0.0
        ddPxddz[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * \
                      (-param_TLS.Sigma *2) \
                    + param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * \
                      (-param_TLS.Sigma * (Z-param_TLS.Mu)*2) * \
                      (-param_TLS.Sigma * (Z-param_TLS.Mu)*2)
        ddPyddz[iz] = 0.0

    # Just rescale on Px
    # TEx = Px
    # Make intTEE zero
    #TEx = ddPxddz + np.dot(ddPxddz, Px)/np.dot(Px, Px)*Px
    # TEx =-ddPxddz + (-param_TLS.Sigma *2)*Px
    TEx = ddPxddz - (-param_TLS.Sigma *2)*Px
    #make Poynting vector zero
    #TEx = ddPxddz + np.dot(Px, dPxdz)/np.dot(ddPxddz, dPxdz)*Px

    # TBy = dPxdz
    TBy = -dPxdz

    # create EM object
    EMP = EhrenfestPlusREB_MaxwellPropagator_1D(param_EM)
    EMP.initializeODEsolver(EB,T0)
    EMP.update_TETB(TEx,TEy,TBx,TBy)
    EMP.applyAbsorptionBoundaryCondition()

	# create TLS object
    if UseInitialRandomPhase:
        param_TLS.C0[0,0] = param_TLS.C0[0,0]*np.exp(1j*2*np.pi*random())
    # TLSP = PureStatePropagator(param_TLS)
    TLSP = DensityMatrixPropagator(param_TLS)

    """
	Start Time Evolution
	"""
    for it in range(len(times)):

	    #0 Compute Polarization:
        intPE = EMP.dZ*np.dot(Px, np.array(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid])) \
                + EMP.dZ*np.dot(Py, np.array(EMP.EB[EMP._Ey:EMP._Ey+EMP.NZgrid]))
        intdPdzB = EMP.dZ*np.dot(-dPydz, np.array(EMP.EB[EMP._Bx:EMP._Bx+EMP.NZgrid])) \
                 + EMP.dZ*np.dot( dPxdz, np.array(EMP.EB[EMP._By:EMP._By+EMP.NZgrid]))

        #0.5 polarization interact with CW
        ECWx = param_EM.A_CW*np.cos(param_EM.K_CW*(param_EM.Zgrid-AU.C*it*dt))
        ECWy = np.zeros(len(param_EM.Zgrid))
        BCWx = np.zeros(len(param_EM.Zgrid))
        BCWy = param_EM.A_CW*np.sin(param_EM.K_CW*(param_EM.Zgrid-AU.C*it*dt))
        intPE += EMP.dZ*np.dot(Px,ECWx) \
        	   + EMP.dZ*np.dot(Py,ECWy)
        intdPdzB += EMP.dZ*np.dot(-dPydz, BCWx ) \
    	          + EMP.dZ*np.dot( dPxdz, BCWy )

	    #1 Propagate the wave function
        TLSP.update_coupling(intPE)
        TLSP.propagate(dt)

    	#2 Compute Current: J
        dPdt = 0.0
        for i in range(param_TLS.nstates):
            for j in range(i+1,param_TLS.nstates):
                dPdt = dPdt + 2*(TLSP.H0[i,i]-TLSP.H0[j,j])*np.imag(TLSP.rho[i,j]) * TLSP.VP[i,j]
        Jx = dPdt * Px
        Jy = dPdt * Py

	    #3. Evolve the field
        EMP.update_JxJy(Jx,Jy)
        EMP.propagate(dt)

        if UsePlusEmission:
    	    #4. Implement additional population relaxation
        	#(2->0)
            if np.abs(TLSP.rho[0,2])==0.0:
                angle = phase_shift
            else:
                angle = np.angle(TLSP.rho[0,2]/np.abs(TLSP.rho[0,2])) + phase_shift
            sign = np.sin(angle)
            # drho20, dE20 = TLSP.getComplement(2,0,dt)
            # drho20, dE20 = TLSP.getComplement_angle(2,0,dt,angle)
            # TLSP.rescale(2,0,drho20)
            kRdt,kDdt,drho20,dE20 = TLSP.getComplement_angle(2,0,dt,angle)
            TLSP.relaxation(2,0,kRdt)
            # TLSP.dephasing(2,0,kDdt)
            EMP.MakeTransition_sign(dE20*dt/Lambda,sign,UseRandomEB=UseRandomEB)

    	    #(2->1)
            if np.abs(TLSP.rho[1,2])==0.0:
                angle = phase_shift
            else:
                angle = np.angle(TLSP.rho[1,2]/np.abs(TLSP.rho[1,2])) + phase_shift
            sign = np.sin(angle)
            # drho21, dE21 = TLSP.getComplement(2,1,dt)
            # drho21, dE21 = TLSP.getComplement_angle(2,1,dt,angle)
            # TLSP.rescale(2,1,drho21)
            kRdt,kDdt,drho21,dE21 = TLSP.getComplement_angle(2,1,dt,angle)
            TLSP.relaxation(2,1,kRdt)
            # TLSP.dephasing(2,1,kDdt)
            EMP.MakeTransition_sign(dE21*dt/Lambda,sign,UseRandomEB=UseRandomEB)

            # print dE20, dE21
            # dE = dE20 + dE21
            # EMP.MakeTransition(dE,UseRandomEB=UseRandomEB)

        if UseThermalRelax:
            #4.5. Apply non-radiative thermal equlibration
            #(1->0)
            # transition=[0,1] --> only downward transistion
            # TLSP.equilibrate(0,1,dt,transition=[0,1])
            kRdt = TLSP.param.gamma_vib*dt
            TLSP.relaxation(1,0,kRdt)

        #5. Apply absorption boundary condition
        EMP.applyAbsorptionBoundaryCondition()

        # save far field out of the box
        EMP.saveFarField(times[it],Tmax)

    	"""
	    output:
	    """
        # density matrix
        for i in range(param_TLS.nstates):
            for j in range(param_TLS.nstates):
                rhot[i,j,it]=TLSP.rho[i,j]

		#energy
        UEB = EMP.getEnergyDensity()
        Uele = TLSP.getEnergy()
        Uemf = np.sum(UEB)*param_EM.dZ
        Ut[0,it] = Uele
        Ut[1,it] = Uemf

        """
        Plot
        """
	    # Plot
        if it%AveragePeriod==0 and ShowAnimation:
            plt.sca(ax[0])
            plt.cla()
            ax[0].fill_between(EMP.Zgrid,0.0,BCWy,alpha=0.5,color='cyan',label='$B_{CW,y}$')
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.EB[EMP._Bx:EMP._Bx+EMP.NZgrid],alpha=0.5,color='blue',label='$B_x$')
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.EB[EMP._By:EMP._By+EMP.NZgrid],alpha=0.5,color='green',label='$B_y$')
            ax[0].axvline(x=param_TLS.Mu, color='k', linestyle='--')
            ax[0].legend()

            plt.sca(ax[1])
            plt.cla()
            ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(ECWx),alpha=0.5,color='yellow',label='$E_{CW,x}$')
            ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ex:EMP._Ex+EMP.NZgrid]),alpha=0.5,color='red',label='$E_x$')
            ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[EMP._Ey:EMP._Ey+EMP.NZgrid]),alpha=0.5,color='orange',label='$E_y$')
	        #ax[1].plot(EMP.Zgrid,EMP.Ax,alpha=0.5,color='red',lw=2,label='$A_x$')
    	    #ax[1].plot(EMP.Zgrid,EMP.Ay,alpha=0.5,color='orange',lw=2,label='$A_y$')
        	#ax[1].set_xlim([-990/np.pi,990/np.pi])
            ax[1].axvline(x=param_TLS.Mu, color='k', linestyle='--')
            ax[1].legend()

            plt.sca(ax[2])
            plt.cla()
            for n in range(param_TLS.nstates):
                ax[2].plot(times[:it]*AU.fs,np.abs(rhot[n,n,:it]),'-',lw=2,label='$C_{d'+str(n)+'}(t)$')
            ax[2].set_xlim([0,Tmax*AU.fs])
            ax[2].set_xlabel('t [a.u.]')
            ax[2].legend(loc='best')

            plt.sca(ax[3])
            plt.cla()
        	#ax[3].plot(times[:it]*AU.fs,PESt[0,:it],lw=2,label='$\lambda_-$')
	        #ax[3].plot(times[:it]*AU.fs,PESt[1,:it],lw=2,label='$\lambda_+$')
    	    #ax[3].plot(times[:it]*AU.fs,Probt[:it],lw=2,label='Prob')
            #ax[3].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0],lw=2,label='ele energy')
            #ax[3].plot(times[:it]*AU.fs,-(Ut[1,:it]-Ut[1,0]),lw=2,label='EM energy')
            #ax[3].legend(loc='best')
            #ax[3].set_xlim([0,Tmax*AU.fs])
            ax[3].plot(EMP.Xs,np.array(EMP.Es)**2,lw=1)
            ax[3].legend(loc='best')
            ax[3].set_xlim([0,Tmax*AU.fs])


            plt.sca(ax[4])
            plt.cla()
            #ax[4].plot(fft_Freq*AU.C,np.abs(fft_Ex)**2,lw=2,color='b')
            ax[4].axvline(x=param_TLS.H0[2,2]-param_TLS.H0[1,1], color='k', linestyle='--')
            ax[4].axvline(x=param_TLS.H0[2,2]-param_TLS.H0[0,0], color='k', linestyle='--')
            #ax[4].plot(fft_Freq*AU.C,(ave_fft_Ex/rolling)**2,lw=2,color='r')
            ax[4].set_xlim([0,0.5])
            #ax[4].set_xlabel('$ck$')
            if len(EMP.Es)<10005:
                fft_Ex = np.fft.rfft(EMP.Es[::-1])
                fft_Freq = np.array(range(len(fft_Ex))) * 2*np.pi /(EMP.Xs[0]-EMP.Xs[-1])
            else:
                fft_Ex = np.fft.rfft(EMP.Es[-10000:-1])
                fft_Freq = np.array(range(len(fft_Ex))) * 2*np.pi /(EMP.Xs[-10000]-EMP.Xs[-1])

            ax[4].plot(fft_Freq,np.abs(fft_Ex)**2,lw=1,color='b')

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
                'UEB':  UEB,
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


data = []
for i in range(NumberTrajectories):
    output = execute(param_EM,param_TLS,ShowAnimation=ShowAnimation)
    data.append(output)
with open(outfile, 'wb') as f:
    pickle.dump(data,f)
