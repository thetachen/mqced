#!/usr/bin/python
from sys import argv
import numpy as np
from scipy.integrate import ode
from utility import *
from random import random
import cPickle as pickle

from units import AtomicUnit
AU = AtomicUnit()

from SystemPropagator import *
from MMSTPropagator import *

#Default Options:
ShowAnimation = False
AveragePeriod = 1
UseInitialRandomPhase = True
NumberTrajectories = 1
UsePlusEmission = True
UseRandomEB = False

# if (len(argv) == 1):
#     execfile('param.in')
#     outfile = 'data.pkl'
# elif (len(argv) == 2):
#     execfile(argv[1])
#     outfile = 'data.pkl'
# elif (len(argv) == 3):
#     execfile(argv[1])
#     outfile = argv[2]


#Options:
ShowAnimation = True
AveragePeriod = 1000


Describer = 'vector'
#Describer = 'density'
NumberTrajectories = 1

# Set up time steps
T0 = 0.0
Tmax = 10000.0
dt = 0.1/2
times = np.arange(T0, Tmax, dt)

param_TLS={
    'nstates':  2,
    # Hamiltoniam
    'H0':		np.array([[0.0,0.0],\
                          [0.0,0.25]]),
	# Coupling (Transition Dipole Moment)
    'VP':       np.array([[0.0,1.0],\
                          [1.0,0.0]]),
    # initial diabatic state vector
    # 'C0':				np.sqrt(np.array([[0.0],[1.0]],complex)),#*np.exp(1j*np.pi*-0.5),
    'C0':				np.sqrt(np.array([[0.5],[0.5]],complex)),
    # polarization
    'Mu':       0.0,
    'Sigma':    0.5, #0.0556 nm-2
    'Pmax':     0.025*np.sqrt(2.0),
}
param_TLS=Struct(**param_TLS)

dW = 0.001/2
NWmax = 50

def execute(param_TLS,ShowAnimation=False):
    """
    Initialize
    """
    # create TLS object
    if UseInitialRandomPhase:
        param_TLS.C0[1,0] = param_TLS.C0[1,0]*np.exp(1j*2*np.pi*random())
    if Describer == 'vector':
        TLSP = PureStatePropagator(param_TLS,KFGR_dimension="1D")
    if Describer == 'density':
        TLSP = DensityMatrixPropagator(param_TLS,KFGR_dimension="1D")
    #TLSP = FloquetStatePropagator(param_TLS,param_EM,dt)

    # create MMST object
    W0 = param_TLS.H0[1,1]-param_TLS.H0[0,0]
    KFGR = TLSP.FGR[1,0]
    MMSTP = MMSTlight_Nmode(W0,dW,NWmax)
    XP0 = np.zeros(2*MMSTP.NW)
    MMSTP.initializeODEsolver(XP0,T0)

    MMSTP_Ehrenfest = MMSTlight_Nmode(W0,dW,NWmax)
    MMSTP_Ehrenfest.initializeODEsolver(XP0,T0)

    # Coupling Strength Per Mode
    # CouplePerMode = param_TLS.Pmax*np.sqrt(2.0*dW/np.pi)*MMSTP.Ws *np.pi*2
    CouplePerMode = param_TLS.Pmax*MMSTP.Ws*np.sqrt(1.0/np.pi)

    # output observables
    rhot = np.zeros((param_TLS.nstates,param_TLS.nstates,len(times)),complex)
    TemperaturePerMode = np.zeros((MMSTP.NW,len(times)),complex)
    Xs = np.zeros((MMSTP.NW,len(times)))
    Ps = np.zeros((MMSTP.NW,len(times)))
    Action = np.zeros(len(times))
    RadiationEnergy = np.zeros(len(times))
    RadiationEnergyDistribution = np.zeros(MMSTP.NW)

    """
    Start Time Evolution
    """

    for it in range(len(times)):
        """
        OUTPUTS:
        """
        # density matrix
        for i in range(param_TLS.nstates):
            for j in range(param_TLS.nstates):
                rhot[i,j,it]=TLSP.rho[i,j]

        # TemperaturePerMode
        for i in range(MMSTP.NW):
            TemperaturePerMode[i,it]= MMSTP.TemperaturePerMode[i]

        # XPs
        for i in range(MMSTP.NW):
            Xs[i,it] = MMSTP.XP[i]
            Ps[i,it] = MMSTP.XP[MMSTP.NW+i]

        # Total Eenergy
        RadiationEnergy[it] = MMSTP.getEnergy()

        # Energy Distribution
        RadiationEnergyDistribution = MMSTP.getEnergyDistribution()

        # Action
        Action[it] = MMSTP.Action

        """
        MAIN PROPAGATION
        """


        # Evaluate the Current Per Mode
        Current = 0.5*CouplePerMode/(MMSTP.Ws**2) * np.imag(TLSP.rho[0,1])
        MMSTP.updateCurrent(Current)
        MMSTP_Ehrenfest.updateCurrent(Current)

        # copy #1: Raditiaon Field in Ehrenfest
        MMSTP_Ehrenfest.propagate(dt)
        XP1 = MMSTP_Ehrenfest.XP

        # copy #2: Radiation Field with Damping
        # Update the Damping
        MMSTP.updateDamp(KFGR,TLSP.rho[1,1])

        # Update the Random Force
        # EnergyChange = MMSTP.getEnergy() - RadiationEnergy[it-1] + W0*(np.abs(TLSP.rho[1,1])-np.abs(rhot[1,1,it-1]))
        EnergyChange = MMSTP.getEnergy() - W0*np.abs(rhot[1,1,0])
        MMSTP.updateRandomForce(EnergyChange,dt)

        # Evolve the field
        MMSTP.propagate(dt)
        XP2 = MMSTP.XP


        intPE_diff = np.dot(CouplePerMode, XP2[:MMSTP.NW]-XP1[:MMSTP.NW]) *dW
        # intPE_diff = np.dot(CouplePerMode, XP2[:MMSTP.NW]) *dW
        PopulationChange = -intPE_diff*np.imag(TLSP.rho[0,1])

        if PopulationChange <= 0.0:# and times[it]<1000.0:
        # if False: # Just do Ehrenfest
        # if True:
            # if the populaiton will go down; keep going
            MMSTP_Ehrenfest.initializeODEsolver(XP2,times[it]+dt)
            intPE = np.dot(CouplePerMode, XP2[:MMSTP.NW]) *dW
        else:
            # if the populaiton will go up; reset
            MMSTP.initializeODEsolver(XP1,times[it]+dt)
            intPE = np.dot(CouplePerMode, XP1[:MMSTP.NW]) *dW

        # intPE = np.dot(CouplePerMode, MMSTP.XP[:MMSTP.NW]) *dW*np.pi
        # intPE = np.dot(CouplePerMode, MMSTP.XP[:MMSTP.NW]) *dW
        #1. Propagate the wave function
        TLSP.update_coupling(intPE)
        TLSP.propagate(dt)


        """
        Plot
        """
        if it%AveragePeriod==0 and ShowAnimation:
            plt.sca(ax[0])
            plt.cla()
            KFGR = TLSP.FGR[1,0]

            ax[0].plot(times[:it],np.abs(rhot[1,1,:it]),'-',lw=2,label='$P_{'+str(1)+'}$')
            ax[0].plot(times[:it],np.abs(param_TLS.C0[1,0])**2*np.exp(-KFGR*times[:it]),'--k',lw=2,label='FGR')
            ax[0].plot(times[:it],np.exp(-KFGR*times[:it])/((np.abs(param_TLS.C0[0,0])**2)/np.abs(param_TLS.C0[1,0])**2+np.exp(-KFGR*times[:it])),'--r',lw=2,label='Ehrenfest')
            ax[0].legend()

            plt.sca(ax[1])
            plt.cla()
            ax[1].plot(times[:it],TemperaturePerMode[NWmax,:it],'-',lw=2,label='temperature')
            ax[1].plot(times[:it],TemperaturePerMode[NWmax+10,:it],'-',lw=2,label='temperature')
            ax[1].plot(times[:it],TemperaturePerMode[NWmax-10,:it],'-',lw=2,label='temperature')
            # ax[1].plot(times[:it],Action[:it],'-',lw=2,label='Action')
            ax[1].legend()

            plt.sca(ax[2])
            plt.cla()

            ax[2].plot(times[:it],RadiationEnergy[:it]+W0*(np.abs(rhot[1,1,:it])),'-',lw=2,label='Total Energy')
            # ax[2].plot(times[:it],-W0*(np.abs(rhot[1,1,:it])-np.abs(rhot[1,1,0])),'-',lw=2,label='System Energy')

            ax[2].set_xlabel('t')
            ax[2].legend(loc='best')

            plt.sca(ax[3])
            plt.cla()

            ax[3].plot(times[:it],RadiationEnergy[:it]+W0*(np.abs(rhot[1,1,:it])-np.abs(rhot[1,1,0])),'-',lw=2,label='Additional Energy')
            ax[3].plot(times[:it],RadiationEnergy[:it],'-',lw=2,label='Radiation Energy')
            ax[3].plot(times[:it],W0*(np.abs(rhot[1,1,:it])-np.abs(rhot[1,1,0])),'-',lw=2,label='System Energy')
            ax[3].axhline(y=0.0, color='k', linestyle='--', lw=2)

            ax[3].set_xlabel('t')
            ax[3].legend(loc='best')

            plt.sca(ax[4])
            plt.cla()
            ax[4].plot(MMSTP.Ws,RadiationEnergyDistribution,'-o',label='Radiation Energy Distribution')

            ax[4].legend(loc='best')
            ax[4].set_xlabel('x')
            ax[4].legend()

            # plt.sca(ax[4])
            # plt.cla()
            # ax[4].plot(times[:it],Xs[NWmax,:it],lw=2,color='b',label='X')
            # ax[4].plot(times[:it],Ps[NWmax,:it],lw=2,color='r',label='P')
            # # ax[4].plot(times[:it],np.arctan2(Xs[NWmax,:it],Ps[NWmax,:it]/W0),lw=2,color='b',label='X-P')
            # ax[4].legend()
            # # fig.savefig('test_plot.pdf')
            fig.canvas.draw()


        #data dictionaray
        if it%AveragePeriod==0 or it==len(times)-1:
            output={
                # 'Zgrid':EMP.Zgrid,
                'times':times,
                'rhot':rhot,
                'Action':Action,
                'RadiationEnergy':RadiationEnergy,
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

rhot11_all=[]
rhot01_all=[]
# for i in range(NumberTrajectories):
data = execute(param_TLS,ShowAnimation=ShowAnimation)
rhot11_all.append(data['rhot'][1,1])
rhot01_all.append(data['rhot'][0,1])


np.savetxt("rhot.dat",zip(np.array(data['times']),np.array(np.abs(data['rhot'][1,1])),np.array(np.real(data['rhot'][0,1])),np.array(np.imag(data['rhot'][0,1])) ))
np.savetxt("Action.dat",zip(np.array(data['times']),np.array(data['Action'])))
np.savetxt("RadiationEnergy.dat",zip(np.array(data['times']),np.array(data['RadiationEnergy'])))


# with open("data.pkl", 'wb') as f:
#     pickle.dump(data,f)

exit()

rhot11_average = np.average(rhot11_all, axis=0)
rhot01_average = np.average(rhot01_all, axis=0)
rhot11_deviate = np.std(rhot11_all, axis=0)
rhot01_deviate = np.std(rhot01_all, axis=0)

# from matplotlib import pyplot as plt
# fig1, ax1= plt.subplots(1,figsize=(8.0,4.0))
# ax1.plot(data['times']*AU.fs,np.abs(rhot11_average),'--k',lw=2)
# ax1.fill_between(data['times']*AU.fs,np.abs(rhot11_average)-rhot11_deviate,np.abs(rhot11_average)+rhot11_deviate)
# plt.show()
rhot={
    'times':    data['times'],
    'rhot11_average':   rhot11_average,
    'rhot01_average':   rhot01_average,
    'rhot11_deviate':   rhot11_deviate,
    'rhot01_deviate':   rhot01_deviate,
}
with open("rhot.pkl", 'wb') as f:
    pickle.dump(rhot,f)

# with open(outfile, 'wb') as f:
    # pickle.dump(data,f)
#
# data = execute(param_EM,param_TLS,ShowAnimation=ShowAnimation)
# # np.savetxt("Es.dat",zip(np.array(data['Xs'][::-1]),np.array(data['Es'][::-1])))
# rhot={
#     'times':    data['times'],
#     'rhot':     data['rhot'],
# }
# with open("rhot.pkl", 'wb') as f:
#     pickle.dump(rhot,f)
