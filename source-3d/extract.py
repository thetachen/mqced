#!/data/home/theta/anaconda2/bin/python
'''
This script is used to extract key info for plotting in server because the data
is too big to scyn in Github.com
'''
import numpy as np
import sys

filename = sys.argv[1]

x_min = -105.0
x_max = 105.0
n_grids = 210
grid = np.linspace(x_min, x_max, n_grids)
dx = (x_max - x_min)/(n_grids-1)

omega = 0.25
p0 = 0.025

def get_energy(filename):
    #E, __, __ = np.mgrid[x_min:x_max+dx:dx, x_min:x_max+dx:dx, x_min:x_max+dx:dx]
    E = np.zeros((n_grids,n_grids,n_grids))
    data = np.loadtxt(filename)
    for i in range(n_grids):
        for j in range(n_grids):
            for k in range(n_grids):
                ijk = i*n_grids**2 + j*n_grids + k
                # E is the norm of EM field
                #data[ijk] *= 1e10
                #ene = data[ijk][0]**2 + data[ijk][1]**2 + data[ijk][2]**2 + data[ijk][3]**2 + data[ijk][4]**2 + data[ijk][5]**2
                E[i,j,k] = data[ijk][0]**2 + data[ijk][1]**2 + data[ijk][2]**2 + data[ijk][3]**2 + data[ijk][4]**2 + data[ijk][5]**2
                #if ene!=0.0: print ene,E[i,j,k]
    return E


# Select a sphere and plot |E| vs \theta

def nearest_idx(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def find_cloest_index_3d(x, y, z, x_max=105, x_min=-105, n_grids=210):
    ''' Given a point (x,y,z), find nearest value of data from this point'''
    x_lst = np.linspace(x_min, x_max, n_grids)
    return nearest_idx(x_lst, x), nearest_idx(x_lst, y), nearest_idx(x_lst, z)


U = get_energy(filename)

# extract r data
rs = np.linspace(x_max*0.05, x_max, 60)
thetas = [np.pi/2,np.pi/4]
#phi = 0.4*np.pi
phis = np.linspace(0, 2.0*np.pi, 100)
#phis = [0.0]
#ENERGY = []
data = [rs]
for theta in thetas:
    E = []
    for r in rs:
        Er = 0.0
        for phi in phis:
            x = r*np.sin(theta) * np.cos(phi)
            y = r*np.sin(theta) * np.sin(phi)
            z = r*np.cos(theta)
            Er += U[find_cloest_index_3d(x, y, z, x_max=x_max, x_min=x_min, n_grids=n_grids)]
        E.append(Er/len(phis))
        #E.append( U[find_cloest_index_3d(x, y, z, x_max=x_max, x_min=x_min, n_grids=n_grids)] ) 
    #ENERGY.append(E)
    data.append(E)
#data = np.array([[r, E1, E2] for r, E1, E2 in zip(rs, ENERGY[0], ENERGY[1])])
data = np.array(data).T
np.savetxt(filename+'.extract_r', data)

# extract theta data
#r = 98.0
r = 80.0
#r = x_max*0.9
#r = 6.0
#r = 2.0
thetas = np.linspace(0, np.pi, 100)
#phis = [0.0,0.1*np.pi,0.2*np.pi,0.3*np.pi,0.4*np.pi,0.5*np.pi] #np.linspace(0, 2.0*np.pi, 100)
phis = np.linspace(0, 2.0*np.pi, 100)
#ENERGY = []
data = [thetas]
for phi in phis:
    E = []
    for theta in thetas:
        x = r*np.sin(theta) * np.cos(phi)
        y = r*np.sin(theta) * np.sin(phi)
        z = r*np.cos(theta)
        E.append( U[find_cloest_index_3d(x, y, z, x_max=x_max, x_min=x_min, n_grids=n_grids)] )

    #ENERGY.append(E)
    data.append(E)
    #E_magnitude = [np.average([E[find_cloest_index_3d(x, y, z,n_grids=n_grids)] for y in Y]) for x, z in zip(X, Z)]
# Save data to file
#data = np.array([[theta, E1, E2] for theta, E1, E2 in zip(thetas, ENERGY[0], ENERGY[1])])
data = np.array(data).T
print data.shape
np.savetxt(filename+'.extract_theta', data)

# extract z data
x = 0.0
y = 0.0
print x_min, x_max
zs = np.linspace(x_min, x_max, 100)

#ENERGY = []
data = [zs]
E = []
for z in zs:
	E.append( U[find_cloest_index_3d(x, y, z, x_max=x_max, x_min=x_min, n_grids=n_grids)] )
data.append(E)
# Save data to file
#data = np.array([[theta, E1, E2] for theta, E1, E2 in zip(thetas, ENERGY[0], ENERGY[1])])
data = np.array(data).T
print data.shape
np.savetxt(filename+'.extract_z', data)


execfile('plot.py')
