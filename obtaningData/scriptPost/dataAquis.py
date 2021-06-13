#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# In[8]:


# Entrada de dados
# Precisão do campo
fieldPrec = 'simple' # 'simple' ou 'double'

# Diretório da entrada de dados
dirInput = '../data/'

# Tag
simu = 'LR_8950'

# Iso-valor para cria a isolinha da projeção de phi
isoValue = 0.001 # 0,1%

# Dados simulação (.prm)
Lx = 25.
Lxb = 1.
Ly = 1.
Lz = 4.

nx = 1601
ny = 145
nz = 256

Re = 8950.
Sc = 1.

x = np.linspace(0., Lx, nx, endpoint = True)
y = np.linspace(0., Ly, ny, endpoint = True)
z = np.linspace(0., Lz, nz, endpoint = True)
dx = x[1] - x[0]
dy = y[1] - y[0]
dz = z[1] - z[0]

#x = x[1:]
#z = z[1:]

dt = 5.e-4
imodulo = 200
dtSnap = dt * imodulo
snapInit = 0
snapFin = 1001
tSnap = np.linspace(snapInit*dtSnap, snapFin*dtSnap, snapFin-snapInit+1, endpoint = True)


# In[10]:


# Inicialização de variaveis
xFront = np.zeros(len(tSnap))
lWindow = 0.25

if fieldPrec == 'simple':
    dataType = np.float32
    
elif fieldPrec == 'double':
    dataType = np.float64

else:
    print('Oi?')


for ii in range(0, snapFin-snapInit+1):
    print('snapshot: %i' %(ii + snapInit))
    fileNameInput = dirInput + 'phi1' + str(ii + snapInit).zfill(4)
    phi = np.fromfile(fileNameInput, dtype=dataType)
    phi[phi>1.] = 1.
    phi[phi<0.] - 0.
    phi = phi.reshape((nz, ny, nx))
    phi = simps(phi, y, axis = 1)
#    phi = phi[1:, 1:]
#    phi[0, :] = 0.
    
    # h_proj
    hProj = phi[0, :] #np.sum(phi, axis = 0)*dz/Lz
    # x_front
    xFront[ii] = np.max(x[hProj > isoValue])
    
    # fazer um corte no campo para obter apenas a isolinha correspondente a frente da corrente
    xNew = x#[x >= (xFront[ii]-lWindow)]
    #phi = phi[:, x >= (xFront[ii]-lWindow)]
    #phi = phi[:, xNew <= (xFront[ii]+lWindow)]
    #xNew = xNew[xNew <= (xFront[ii]+lWindow)]
    
    zz, xx = np.meshgrid(z, xNew, indexing = 'ij')
    rr = np.sqrt(xx**2. + zz**2.)
    phi[rr <= (xFront[ii]-lWindow)] = 1.
#    phi[rr >= (xFront[ii]+lWindow)] = 0.


    fig0, ax0 = plt.subplots()
    C = plt.contour(xx, zz, phi, levels = [isoValue])
    p = C.collections[0].get_paths()[0]
    v = p.vertices
    xLine = v[:,0]
    zLine = v[:,1]
    plt.close(fig0)
    
    # Nome do arquivo de saida de dados
    fileNameOutPut = './lineData/' + simu + '_' + str(ii + snapInit).zfill(4)
    np.save(fileNameOutPut, np.array([zLine, xLine]))
    

fileName = './lineData/xFront_' + simu
np.save(fileName, np.array([tSnap, xFront]))    
print('Feitoooo!')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




