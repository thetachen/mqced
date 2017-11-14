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
    # dipole moment (Xi): Dx, Dy
    Dx = np.zeros(param_EM.NZgrid)
    Dy = np.zeros(param_EM.NZgrid)

    # curl D (for rescale B and keep div B =0)
    dDxdz = np.zeros(param_EM.NZgrid)
    dDydz = np.zeros(param_EM.NZgrid)

    # current: Jx, Jy
    Jx = np.zeros(param_EM.NZgrid)
    Jy = np.zeros(param_EM.NZgrid)

    # a long vector [Ex,Ey,Bx,By]
    EB = np.zeros(4*param_EM.NZgrid)

    # wave vector
    Ct = np.zeros((param_TLS.nstates,len(times)),complex)

    # total energy
    Ut = np.zeros((2,len(times)))

    # initialize Dx,Dy,dDxdz, dDydz
    for iz in range(param_EM.NZgrid):
    	Z = param_EM.Zgrid[iz]    
        Dx[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 )
        Dy[iz] = 0.0
        dDxdz[iz] = param_TLS.Pmax * np.sqrt(param_TLS.Sigma/np.pi) * np.exp( -param_TLS.Sigma * (Z-param_TLS.Mu)**2 ) * -param_TLS.Sigma * (Z-param_TLS.Mu)*2
        dDydz[iz] = 0.0

    # create EM object
    EMP = AugmentedEhrenfestMaxwellPropagator_1D(param_EM)
    EMP.initializeODEsolver(EB,T0)
    EMP.applyAbsorptionBoundaryCondition()

	# create TLS object
    TLSP = ThreeLevelSystemPropagator(param_TLS)

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

	    #0 Compute Polarization:
    	intDE = EMP.dZ*np.dot(Dx, np.array(EB[param_EM._Ex:param_EM._Ex+EMP.NZgrid])) \
        	  + EMP.dZ*np.dot(Dy, np.array(EB[param_EM._Ey:param_EM._Ey+EMP.NZgrid]))
        intDD = EMP.dZ*np.dot(Dx, Dx) \
                + EMP.dZ*np.dot(Dy, Dy)
        intdDdzB = EMP.dZ*np.dot(-dDydz, np.array(EB[param_EM._Bx:param_EM._Bx+EMP.NZgrid])) \
                + EMP.dZ*np.dot( dDxdz, np.array(EB[param_EM._By:param_EM._By+EMP.NZgrid]))
        intdDdzdDdz = EMP.dZ*np.dot(dDxdz, dDxdz) \
                    + EMP.dZ*np.dot(dDydz, dDydz)

        #0.5 polarization interact with CW
        ECWx = param_EM.A_CW*np.cos(param_EM.K_CW*(param_EM.Zgrid-AU.C*it*dt))
        ECWy = np.zeros(len(param_EM.Zgrid))
        BCWx = np.zeros(len(param_EM.Zgrid))
        BCWy = param_EM.A_CW*np.sin(param_EM.K_CW*(param_EM.Zgrid-AU.C*it*dt))
        intDE += EMP.dZ*np.dot(Dx,ECWx) \
        	   + EMP.dZ*np.dot(Dy,ECWy)
        intdDdzB += EMP.dZ*np.dot(-dDydz, BCWx ) \
    	          + EMP.dZ*np.dot( dDxdz, BCWy )

	    #1 Propagate the wave function
        TLSP.update_coupling(intDE)
        TLSP.propagate(dt)
    
    	#2 Compute Current: J
        dPdt = 0.0
        for i in range(param_TLS.nstates):
            for j in range(i+1,param_TLS.nstates):
                dPdt = dPdt + 2*(TLSP.H0[i,i]-TLSP.H0[j,j])*np.imag(TLSP.C[i,0]*np.conj(TLSP.C[j,0])) * TLSP.VP[i,j]
        Jx = dPdt * Dx
        Jy = dPdt * Dy
     
	    #3. Evolve the field
        EMP.update_JxJy(Jx,Jy)
        EMP.propagate(dt)
        EB = EMP.EB

	    #4. Implement additional population relaxation 
    	#(2->0)
        gamma = TLSP.FGR[2,0]*(np.abs(TLSP.C[1,0])**2 + np.abs(TLSP.C[2,0])**2)
        drho = gamma*dt * np.abs(TLSP.C[2,0])**2
        dE = (TLSP.H0[2,2]-TLSP.H0[0,0])*gamma*dt * np.abs(TLSP.C[2,0])**2
        TLSP.rescale(2,0,drho)

        theta = random()*2*np.pi
        cos2, sin2  = np.cos(theta)**2, np.sin(theta)**2
        EMP.ContinuousEmission_E(dE*cos2,intDE,intDD,Dx)
        EMP.ContinuousEmission_B(dE*sin2,intdDdzB,intdDdzdDdz,dDxdz)
        #EMP.ContinuousEmission_B(dE,intdDdzB,intdDdzdDdz,dDxdz)
        EB = EMP.EB

	    #(2->1)
        gamma = TLSP.FGR[2,1]*(np.abs(TLSP.C[0,0])**2 + np.abs(TLSP.C[2,0])**2)
        drho = gamma*dt * np.abs(TLSP.C[2,0])**2
        dE = (TLSP.H0[2,2]-TLSP.H0[1,1])*gamma*dt * np.abs(TLSP.C[2,0])**2
        TLSP.rescale(2,1,drho)

        theta = random()*2*np.pi
        cos2, sin2  = np.cos(theta)**2, np.sin(theta)**2
        EMP.ContinuousEmission_E(dE*cos2,intDE,intDD,Dx)
        EMP.ContinuousEmission_B(dE*sin2,intdDdzB,intdDdzdDdz,dDxdz)
        #EMP.ContinuousEmission_B(dE,intdDdzB,intdDdzdDdz,dDxdz)
        EB = EMP.EB

	    #5. Apply dissipative relaxation
    	#(1->0) Dissipative Relaxation
        gamma = param_TLS.gamma02
        drho = -gamma*dt
        #drho = gamma*dt *( np.abs(TLSP.C[0,0])**2 * np.exp(-param_TLS.beta*TLSP.H0[1,1]) \
	                      #-np.abs(TLSP.C[1,0])**2 * np.exp(-param_TLS.beta*TLSP.H0[0,0]) )
        if drho > 0.0:
            TLSP.rescale(0,1,np.abs(drho))
        elif drho < 0.0:
            TLSP.rescale(1,0,np.abs(drho))
        else:
            pass

        
        #5. Apply absorption boundary condition
        EMP.applyAbsorptionBoundaryCondition()
	
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
    	Ex= EMP.EB[param_EM._Ex+(EMP.NZgrid-1)/2:param_EM._Ex+(EMP.NZgrid-1)]
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
	    # Plot
        if it%10==0 and ShowAnimation:
            plt.sca(ax[0])
            plt.cla()
            ax[0].fill_between(EMP.Zgrid,0.0,BCWy,alpha=0.5,color='cyan',label='$B_{CW,y}$')
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.EB[param_EM._Bx:param_EM._Bx+EMP.NZgrid],alpha=0.5,color='blue',label='$B_x$')
            ax[0].fill_between(EMP.Zgrid,0.0,EMP.EB[param_EM._By:param_EM._By+EMP.NZgrid],alpha=0.5,color='green',label='$B_y$')
            ax[0].axvline(x=param_TLS.Mu, color='k', linestyle='--')
            ax[0].legend()

            plt.sca(ax[1])
            plt.cla()
            ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(ECWx),alpha=0.5,color='yellow',label='$E_{CW,x}$')
            ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[param_EM._Ex:param_EM._Ex+EMP.NZgrid]),alpha=0.5,color='red',label='$E_x$')
            ax[1].fill_between(EMP.Zgrid,0.0,np.sqrt(AU.E0)*(EMP.EB[param_EM._Ey:param_EM._Ey+EMP.NZgrid]),alpha=0.5,color='orange',label='$E_y$')
	        #ax[1].plot(EMP.Zgrid,EMP.Ax,alpha=0.5,color='red',lw=2,label='$A_x$')
    	    #ax[1].plot(EMP.Zgrid,EMP.Ay,alpha=0.5,color='orange',lw=2,label='$A_y$')
        	#ax[1].set_xlim([-990/np.pi,990/np.pi])
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
        	#ax[3].plot(times[:it]*AU.fs,PESt[0,:it],lw=2,label='$\lambda_-$')
	        #ax[3].plot(times[:it]*AU.fs,PESt[1,:it],lw=2,label='$\lambda_+$')
    	    #ax[3].plot(times[:it]*AU.fs,Probt[:it],lw=2,label='Prob')
            ax[3].plot(times[:it]*AU.fs,Ut[0,:it]-Ut[0,0],lw=2,label='ele energy')
            ax[3].plot(times[:it]*AU.fs,-(Ut[1,:it]-Ut[1,0]),lw=2,label='EM energy')
            ax[3].legend(loc='best')
            ax[3].set_xlim([0,Tmax*AU.fs])


            plt.sca(ax[4])
            plt.cla()
            ax[4].plot(fft_Freq*AU.C,np.abs(fft_Ex)**2,lw=2,color='b')
            ax[4].axvline(x=param_TLS.H0[2,2]-param_TLS.H0[0,0], color='k', linestyle='--')
            ax[4].plot(fft_Freq*AU.C,(ave_fft_Ex/rolling)**2,lw=2,color='r')
            ax[4].set_xlim([0,2])
            ax[4].set_xlabel('$ck$')

            fig.canvas.draw()

		#data dictionaray
    	if it%AveragePeriod==0:
	        output={
    	        'Zgrid':EMP.Zgrid,
        	    'times':times,
            	'Ex':   EMP.EB[param_EM._Ex:param_EM._Ex+EMP.NZgrid],
	            'UEB':  UEB,
    	        'Ct':  Ct,
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

