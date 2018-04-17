import numpy as np

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def MaxOne(A):
    return A/max(A)


def exp_fit(x,Fx,C0):
    #exp fitting
    from scipy.optimize import curve_fit
    def func1(x, b, a):
        return a*np.exp(-b*x) + (C0-a)
    def func2(x, b):
        return C0*np.exp(-b*x)
    #popt1, pcov1 = curve_fit(func1, x, Fx)
    popt2, pcov2 = curve_fit(func2, x, Fx)
    #ax[0].plot(x,func(x,*popt1),lw=2,label='exp fitting')
    #KR = min([popt1[0],popt2[0]], key= lambda x: np.abs(x-KFGR))
    KR = popt2[0]
    return KR

def step(x):
    return (np.sign(x)+1)*0.5

def moving_average(a, n=500):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
