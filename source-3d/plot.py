#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#from scipy.optimize import curve_fit
import sys
#plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman', size='14')

#input
filename = sys.argv[1]
filename_z = filename+'.extract_z'
filename_r = filename+'.extract_r' 
filename_theta = filename+'.extract_theta'
time2fs = 0.01
k2fs_1 = 100 # fs^{-1}
x2nm = 2.998 # nm
#x2nm  = 1.0
mu2C_nm_mole = 1.5884*1e-27*(6.023*1e+23)*1e+9 # C*nm/mole
e2ev = 65.82 # ev
energydensity = e2ev/x2nm/x2nm/x2nm/2 # ev/nm

# fit data
def func(theta, coeff):
    return coeff*np.sin(theta)**2

# x-z plane
omega = 0.25
p0 = 0.025


def plot_e2_r(filename):

    def plot_e2_r_theoretical(ax, theta=np.pi/2, color='k--',label='Dipole radiation'):
        rt = r
        u_theoretical = omega**4*p0**2/32/np.pi**2/rt**2*np.sin(theta)**2
        ax.plot(rt*x2nm,  1*u_theoretical*np.cos(2*omega*(98-rt))**2*energydensity, color, lw=2, label=label)
        #ax.plot(rt*x2nm, 2*u_theoretical*np.cos(omega*(100-rt))**2*energydensity, color, lw=2, label=label)
	#data = np.loadtxt('electric+0.5_t_100.000000.txt.extract_r')
    data = np.loadtxt(filename)
    r, e1, e2 = data[:,0], data[:,1], data[:, 2]

    fig, axes = plt.subplots(2,1, sharex=True)
    fig.set_size_inches(7, 7)
        
    #axes[0].plot(r[10:]*x2nm, 5e-7/(r[10:]*x2nm)**2, '-', lw=2)
    axes[0].plot(r*x2nm, 2*e1*energydensity, '-o', lw=2, markersize=8, label='Ehrenfest',mfc='none',alpha=0.7)
    plot_e2_r_theoretical(axes[0], theta=np.pi/2, color='k--', label='dipole radiation')
	#axes[0].plot(r_2*x2nm, e1_2*energydensity, 'b+', lw=2, markersize=8, label='CPA', mfc='none', alpha=0.50)
    axes[0].legend()
	#axes[0].set_ylim(np.min(e1*energydensity)-0.15e-10, np.max(e1*energydensity)*1.4)

    axes[1].plot(r*x2nm, e2*energydensity, '-o', lw=2, markersize=8, label='Ehrenfest ', mfc='none',alpha=0.7)
    plot_e2_r_theoretical(axes[1], theta=np.pi/4, color='k--', label='dipole radiation')
	#axes[1].plot(r_2*x2nm, e2_2*energydensity, 'b+', lw=2, markersize=8, label='CPA', mfc='none', alpha=0.50)
    axes[1].legend()
	#axes[1].set_ylim(np.min(e2*energydensity)-0.15e-10, np.max(e2*energydensity)*1.4)
    axes[0].set_ylim([0,1e-10])
    axes[1].set_ylim([0,1e-10])
	#axes[0].ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
	#axes[0].yaxis.major.formatter._useMathText = True
	#axes[1].ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
	#axes[1].yaxis.major.formatter._useMathText = True

    axes[-1].set_xlabel('$r$ (nm)')
    fig.text(0, 0.5, 'Energy density (eV/nm$^{3}$)', va='center', rotation='vertical')

    plt.subplots_adjust(hspace=0)

	#plt.tight_layout()
	#plt.savefig("E2_vs_r.eps", dpi=300)
    plt.show(block=False)


def plot_e2_theta(filename, r=98):
    #filename='electric+0.5_t_100.000000.txt.extract_theta'
    data = np.loadtxt(filename)
    #print len(data[0])
    theta = data[:,0]
    N=len(data[0])-1
    #for i in range(len(data[0])-1):
        #data[:,1]
    #filename2='check_em_distribution_short_time/e2_theta_r_%d_2.txt' %r
    #data2 = np.loadtxt(filename2)
    #theta2, E_magnitude2 = data2[:,0], data2[:,1]

    theoretical_coeff = omega**4*p0**2/32/np.pi**2/r**2
    u_theoretical = theoretical_coeff*np.sin(theta)**2

    #popt, pocv = curve_fit(func, theta, E_magnitude, p0=1e-12)
    #fitted_coeff = popt[0]
    #E_fitted = np.array(np.sin(theta)**2*fitted_coeff)

    #fig, ax = plt.subplots(N)
    #fig.set_size_inches(4, 10)
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(5, 5)

    #ax.plot(theta/np.pi, E_magnitude*energydensity, '-o', lw=2, markersize=8, mfc='none',label='Ehrenfest')
    ave = data.T
    ave = ave[1:]
    ave = np.average(ave, axis=0)
    #for i in range(len(data[0]))[1:]:
    ##for i in [3]:
        #ax.plot(theta/np.pi, data[:,i]*energydensity, '-o', lw=2, markersize=8, mfc='none',label=str(i),alpha=0.7)
    ax.plot(theta/np.pi, ave*energydensity, '-ob', lw=2, markersize=8, mfc='none',label='ave',alpha=0.7)
    ax.plot(theta/np.pi, max(ave*energydensity*0.97)*np.sin(theta)**2, '--k',label='ave')
    #ax.legend()
    #ax.plot(theta2/np.pi, E_magnitude2*energydensity, 'b+', lw=2, markersize=8, mfc='none', label='CPA', alpha=0.5)
    #ax.plot(theta/np.pi, 2*u_theoretical*np.cos(omega*(100-r))**2*energydensity, 'k--', lw=2, label='dipole radiation')
    #ax.plot(theta/np.pi, E_fitted*energydensity, 'g--', lw=2, label='fitted curve')
    #ax.set_xlabel('$ \\theta $')
    #ax.set_ylabel('Energy density (eV/nm$^{3}$)')
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, np.max(E_magnitude)*energydensity*1.6)
    #ax.legend()
    #ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
    #ax.yaxis.major.formatter._useMathText = True
    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f$\pi$"))
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
    #ax.yaxis.set_ticks_position('both')
    #ax.xaxis.set_ticks_position('both')
#
    #print("fitted curve: coefficient = %.2E" %(fitted_coeff*energydensity))
    print("theoretical curve: coefficient = %.2E" %(theoretical_coeff*energydensity))

    #plt.tight_layout()
    #plt.savefig("E2_vs_theta.eps", dpi=300)
    plt.show(block=False)


def plot_e2_z(filename):
    #filename='electric+0.5_t_100.000000.txt.extract_theta'
    data = np.loadtxt(filename)
    #print len(data[0])
    z = data[:,0]
    N=len(data[0])-1

    fig, ax = plt.subplots(N)

    ax.plot(z, data[:,1], '-o', lw=2, markersize=8, mfc='none',label='Ehrenfest',alpha=0.7)


plot_e2_z(filename_z)
plot_e2_theta(filename_theta)
plot_e2_r(filename_r)
plt.show()
