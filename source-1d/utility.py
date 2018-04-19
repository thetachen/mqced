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


def find_peak(x,y,x0,dx):
    ix_min,ix_max = np.argmin(np.abs(x-(x0-dx))),np.argmin(np.abs(x-(x0+dx)))
    y_max = max(y[ix_min:ix_max])
    x_max = x[np.argmin(np.abs(y-y_max))]
    return x_max,y_max

def window_fft(t,ft,tmin,tmax):
    dt = t[1]-t[0]
    itmin, itmax = np.argmin(np.abs(t-tmin)),np.argmin(np.abs(t-tmax))
    fftw = np.fft.rfft(ft[itmin:itmax])/(t[itmax]-t[itmin])
    freq = np.array(range(len(fftw))) * 2*np.pi/(t[itmax]-t[itmin])
    #fft_Ex = np.fft.rfft(Es)
    #fft_Freq = np.array(range(len(fft_Ex))) * 2*np.pi /(Xs[-1]-Xs[0])

    return freq,fftw
