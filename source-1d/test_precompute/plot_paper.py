#!/usr/bin/python
import sys
import gc
sys.path.append("/home/theta/sync/mqced/source-1d")
import numpy as np
import cPickle as pickle
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, MaxNLocator
import matplotlib.gridspec as gridspec
from utility import *
plt.rc('text', usetex=True)
plt.rc('font', family='Times New Roman', size='9')
xsize = 8.6 #cm
ysize = 20 #cm
cm2inch = 1.0/2.54
t2fs = 0.01
x2nm = 2.998
e2ev = 65.82

W0 = 0.25
P0 = 0.025*np.sqrt(2.0)
field_unit = W0*P0

syncoutput = "/home/theta/sync/mqced/documents/"
prefix = "source-1d.test_precompute."
def main():
    fig, ax= plt.subplots(2,1,figsize=(xsize*cm2inch,8.0*cm2inch))
    plot_Field(ax,'data.pkl',show_legend=False)
    fig.tight_layout(pad=0.3, w_pad=0.0, h_pad=0.0)
    # fig.text(0.0,0.97,'(a)', fontsize=9)
    fig.savefig('fig.pdf')
    fig.savefig(syncoutput+prefix+'fig.pdf')
    plt.show(block=False)

    plt.show()


def plot_Field(ax,datafile,show_legend=True):

    with open(datafile, 'r') as file:
        data = pickle.load(file)
        for d in data:
            D = Struct(**d)

            ax[0].fill_between(np.array(D.Zgrid)*x2nm,0.0,D.Ex/field_unit,color='red', lw=1,alpha=0.2, label=r'Eh+R')
            ax[1].fill_between(np.array(D.Zgrid)*x2nm,0.0,D.By/field_unit,color='green', lw=1,alpha=0.2, label=r'Eh+R')
        ax[0].plot(np.array(D.Zgrid)*x2nm,D.TEx*0.1/field_unit,'--',color='red', lw=2,alpha=0.8, label=r'Eh+R')
        ax[1].plot(np.array(D.Zgrid)*x2nm,D.TBy*0.1/field_unit,'--',color='green', lw=2,alpha=0.8, label=r'Eh+R')
    # ax.set_ylim([-5E-3,5E-3])
    ax[0].set_xlim([-20.0,20.0])
    ax[0].set_xlabel(r'$x$ [nm]')
    ax[0].set_ylabel(r'$E_z$')
    ax[1].set_xlim([-20.0,20.0])
    ax[1].set_xlabel(r'$x$ [nm]')
    ax[1].set_ylabel(r'$B_y$')
    # ax.text(2.5E4,4.0E-5,r'$\rho_{11}='+str(C0)+'$')
    if show_legend: ax.legend(fontsize=8,loc='upper left')

    # ax.ticklabel_format(style='sci',scilimits=(0,0),axis='both',useLocale=True)


if __name__ == '__main__':
    main()
