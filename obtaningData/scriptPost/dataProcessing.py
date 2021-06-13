#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from scipy.integrate import simps

def ecdfMine(F, xProb):
    yProb = np.zeros(len(xProb))
    dataSize = len(F)
    Fsorted = np.sort(F)
    temp2 = 0
    for ii in range(0, len(xProb)):
        temp = 0.
        temp += temp2
        for jj in range(temp2, len(Fsorted)):
            if Fsorted[jj] <= xProb[ii]:
                temp += 1.
            else:
                break
    
        yProb[ii] = temp/dataSize
        temp2 = int(temp)
    return yProb

def deriv1O6(n, h):
    A = np.zeros((n,n))
    B = np.zeros((n,n))
    alpha6 = 1./3.
    a6 = 14./(9. * 2. * h)
    b6 = 1./(9. * 4. * h)
    for ii in range(0, n):
        if ii == 0:
            A[0, 0] = 1.
            A[0, 1] = 2.
            B[0, 0] = -5. / (2. * h)
            B[0, 1] = 4. / (2. * h)
            B[0, 2] = 1. / (2. * h)
        elif ii == 1:
            A[1, 1] = 1.
            A[1, 0] = 1. / 4.
            A[1, 2] = 1. / 4.
            B[1, 2] = 3. / (4. * h)
            B[1, 0] = -3. / (4. * h)
        elif ii == n-1:
            A[n-1, n-1] = 1.
            A[n-1, n-2] = 2.
            B[n-1, n-1] = 5. / (2. * h)
            B[n-1, n-2] = -4. / (2. * h)
            B[n-1, n-3] = -1. / (2. * h)
        elif ii == n-2:
            A[n-2, n-2] = 1.
            A[n-2, n-1] = 1. / 4.
            A[n-2, n-3] = 1. / 4.
            B[n-2, n-3] = -3. / (4. * h)
            B[n-2, n-1] = 3. / (4. * h)
        else:
            A[ii, ii] = 1.
            A[ii, ii-1] = alpha6
            A[ii, ii+1] = alpha6
            B[ii, ii+1] = a6
            B[ii, ii+2] = b6
            B[ii, ii-2] = -b6
            B[ii, ii-1] = -a6

    D = (np.linalg.inv(A)).dot(B)
    return D

def deriv1d(field, nx, dx):
    D1 = deriv1O6(nx, dx)
    dfielddx = D1.dot(field)

    return dfielddx

def derivx3d(field, nx, ny, nz, dx):
    D1x = deriv1O6(nx, dx)
    dfielddx = np.zeros((nz, ny, nx))
    for ii in range(0, nz):
        for jj in range(0, ny):
            dfielddx[ii, jj, :] = D1x.dot(field[ii, jj, :])
    
    return dfielddx


# In[25]:


simu = 'LR_Re8950'
Lx = 25.
Lxb = 1.
Ly = 1.
Lz = 4.

nx = 1601
ny = 145
nz = 256

Re = 8950.
Ri = 1.
Sc = 1.

x = np.linspace(0., Lx, nx, endpoint = True)
x -= Lxb
y = np.linspace(0., Ly, ny, endpoint = True)
z = np.linspace(0., Lz, nz, endpoint = False)

dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

dt = 5.e-4
imodulo = 200
dtSnap = dt * imodulo
snapInit = 0
snapFin = 1000
tSnap = np.linspace(snapInit*dtSnap, snapFin*dtSnap, snapFin-snapInit+1, endpoint = True)


# In[26]:


fileName = 'xCoord_' + simu
np.save(fileName, x)

fileName = 'yCoord_' + simu
np.save(fileName, y)

fileName = 'zCoord_' + simu
np.save(fileName, z)

fileName = 'tCoord_' + simu
np.save(fileName, tSnap)


# In[47]:


pBack = np.zeros(len(tSnap))
pTotal = np.zeros(len(tSnap))
hLA = np.zeros((len(tSnap), len(x)))
hLA2 = np.zeros((len(tSnap), len(x)))
uLA = np.zeros((len(tSnap), len(x)))
phiLA = np.zeros((len(tSnap), len(x)))
hProj = np.zeros((len(tSnap), len(x)))
uPhiRef = np.zeros((len(tSnap), len(x)))

for ii in range(0, snapFin-snapInit+1):
    print(ii)
    fileName = '../data/phi1' + str(ii + snapInit).zfill(4)
    phi = np.fromfile(fileName, dtype=np.float32)
    phi[phi >= 1.] = 1.
    phi[phi <= 0.] = 0.
    
    # P_b
#    xProb = np.sort(np.unique(phi))
#    yProb = ecdfMine(phi, xProb)
#    yProb = (1.-yProb)
#    iSort = np.argsort(yProb)
#    yProb = yProb[iSort]
#    xProb = xProb[iSort]
#    pBack[ii] = simps(yProb*xProb, yProb)*Lx*Lz
    
    phi = phi.reshape((nz, ny, nx))
    
    # P_t
#    pTotal[ii] = simps(simps(np.sum(phi, axis = 0)*dz, y, axis = 0), x, axis = 0)
    
    # \varphi_p
#    fileName = './proj_'+ simu + '/phiProj' + str(ii + snapInit).zfill(4)
#    np.save(fileName, simps(phi, y, axis = 1))
    
    fileName = '../data/ux' + str(ii + snapInit).zfill(4)
    ux = np.fromfile(fileName, dtype=np.float32)
    ux = ux.reshape((nz, ny, nx))
    
    fMask = np.zeros((nz, ny, nx))
    fMask[ux >= 0.] = 1.
    
    uh = simps(np.sum(ux*fMask, axis = 0)*dz/Lz, y, axis = 0)
    uuh = simps(np.sum(ux*ux*fMask, axis = 0)*dz/Lz, y, axis = 0)
#    uch = simps(np.sum(ux*phi*fMask, axis = 0)*dz/Lz, y, axis = 0)
#    uucch = simps(np.sum(ux*phi*ux*phi*fMask, axis = 0)*dz/Lz, y, axis = 0)
    
    # h_LA
    aux = uh*uh/uuh
    iFilt = np.isnan(aux)
    aux[iFilt] = 0.
    iFilt = np.isinf(aux)
    aux[iFilt] = 0.
    hLA[ii, :] = aux
    
    # h_LA2
#    aux = uch*uch/uucch
#    iFilt = np.isnan(aux)
#    aux[iFilt] = 0.
#    iFilt = np.isinf(aux)
#    aux[iFilt] = 0.
#    hLA2[ii, :] = aux
    
    # u_LA
#    aux = uuh/uh
#    iFilt = np.isnan(aux)
#    aux[iFilt] = 0.
#    iFilt = np.isinf(aux)
#    aux[iFilt] = 0.
#    uLA[ii, :] = aux
    
    # \varphi_LA
#    aux = uch/uh
#    iFilt = np.isnan(aux)
#    aux[iFilt] = 0.
#    iFilt = np.isinf(aux)
#    aux[iFilt] = 0.
#    phiLA[ii, :] = aux
    
    # h_p
    aux = simps(np.sum(phi, axis = 0)*dz/Lz, y, axis = 0)
    iFilt = np.isnan(aux)
    aux[iFilt] = 0.
    iFilt = np.isinf(aux)
    aux[iFilt] = 0.
    hProj[ii, :] = aux
    
    # uPhi_ref
    aux = simps(np.sum(ux*phi - derivx3d(phi, nx, ny, nz, dx)/(Re*Sc), axis = 0)*dz, y, axis = 0)
    iFilt = np.isnan(aux)
    aux[iFilt] = 0.
    iFilt = np.isinf(aux)
    aux[iFilt] = 0.
    uPhiRef[ii, :] = aux
    
fileName = 'pBack_' + simu
np.save(fileName, pBack)

fileName = 'pTotal_' + simu
np.save(fileName, pTotal)

fileName = 'hLA_' + simu
np.save(fileName, hLA)

fileName = 'hLA2_' + simu
np.save(fileName, hLA2)

fileName = 'uLA_' + simu
np.save(fileName, uLA)

fileName = 'phiLA_' + simu
np.save(fileName, phiLA)

fileName = 'hProj_' + simu
np.save(fileName, hProj)

fileName = 'uPhiRef_' + simu
np.save(fileName, uPhiRef)
