#!/usr/bin/env python

import matplotlib as m
m.use('Agg') # to plot without X11
#http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server

import matplotlib.pyplot as plt
import numpy as np
import re
import os

### the following imports are there solely for debuging purposes
import pprint
from sys import exit
###


## TODO LIST
# get simulation name and prefix from command line arguments
# check if simulation path and required files exists and report any error
# ovewrite any parameter (nx, ny, etc, from command line parameter)
# choose plotting axis from command line

simulation = {}
simulation['prefix'] = '/damarea/ricardo'
#simulation['prefix'] = '/media/miriam/c725c456-6282-42d6-831a-94d4649daf3a/backup/DOUTORADO'
#simulation['name'] = 'BASIN_DNS_5K'
#simulation['name'] = 'data_channel'
simulation['output'] = './out_planes'

if not simulation['output']:
    simulation['output'] = simulation['prefix'] + simulation['name'] + '/output/'

if ( not (os.path.exists(simulation['output']) and os.path.isdir(simulation['output'])) ):
    print 'Warning! Output dir not found or is not a directory. Set to ".".'
    simulation['output'] = '.'

# Avaliable simulations currently stored on /damarea/ricardo
simulationNames= [ 'BASIN_DNS_10K',                  # 0
                   'BASIN_DNS_5K',                   # 1 
                   'BASIN_LES_10K',                  # 2
                   'BASIN_LES_5K',                   # 3
                   'BASIN_LES_5K_h_reduzido',        # 4
                   'BL100K',                         # 5
                   'eptt2016.P',                     # 6
                   'eptt2016.QS',                    # 7
                   'incompact3d_channel_temporal2',  # 8
                   'teste',                          # 9
                   'newCaGeo2k_DNS']                 # 10

simulation['name'] = simulationNames[10]

if (os.path.exists(simulation['prefix'] + '/' + simulation['name'] + '/data') and
    os.path.isdir(simulation['prefix'] + '/' + simulation['name'] + '/data')
    ):
    simulation['dataplace'] = '/data/'
else:
    simulation['dataplace'] = '/'
    
##########################################################
# plot config options
#from numba import jit #http://numba.pydata.org/numba-doc/dev/user/examples.html
plt.rc('text', usetex=True)
plt.rc('font', family='Linux Libertine O')

# http://stackoverflow.com/questions/3373256/set-colorbar-range-in-matplotlib
# rainbow desaturated
# cdict = {
# 0.000 :(0.278431, 0.278431, 0.858824)
# 0.143 :(0.000000, 0.000000, 0.360784)
# 0.285 :(0.000000, 1.000000, 1.000000)
# 0.429 :(0.000000, 0.501961, 0.000000)
# 0.571 :(1.000000, 1.000000, 0.000000)
# 0.714 :(1.000000, 0.380392, 0.000000)
# 0.857 :(0.419608, 0.000000, 0.000000)
# 1.000 :(0.878431, 0.301961, 0.301961)}

cdict = {
    'red'  : ((0.000000, 0.278431, 0.278431), (0.143000, 0.000000, 0.000000), (0.285000, 0.000000, 0.000000), (0.429000, 0.000000, 0.000000), (0.571000, 1.000000, 1.000000), (0.714000, 1.000000, 1.000000), (0.857000, 0.419608, 0.419608), (1.000000, 0.878431, 0.878431)),
    'green': ((0.000000, 0.278431, 0.278431), (0.143000, 0.000000, 0.000000), (0.285000, 1.000000, 1.000000), (0.429000, 0.501961, 0.501961), (0.571000, 1.000000, 1.000000), (0.714000, 0.380392, 0.380392), (0.857000, 0.000000, 0.000000), (1.000000, 0.301961, 0.301961)),
    'blue' : ((0.000000, 0.858824, 0.858824), (0.143000, 0.360784, 0.360784), (0.285000, 1.000000, 1.000000), (0.429000, 0.000000, 0.000000), (0.571000, 0.000000, 0.000000), (0.714000, 0.000000, 0.000000), (0.857000, 0.000000, 0.000000), (1.000000, 0.301961, 0.301961))}
'''
cdict = {
  'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., 1.)),
  'green':  ( (0.0, 0.0, 0.0), (0.02, .45, .45), (1., .97, .97)),
  'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))}
'''
cm = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

#################################################################
# get parameters from fortran code
f = open(simulation['prefix'] + '/' + simulation['name'] + '/module_param.f90', 'r')
module_param = f.readlines()
f.close()
f = open(simulation['prefix'] + '/' + simulation['name'] + '/incompact3d.prm', 'r')
incompact3d = f.readlines()
f.close()

def getVarFromFortranSource(line, variable):
    """From a line from a fortran program, extracts the value of a variable assignment

    Keyword arguments:
    line -- one line from a fortran program
    variable -- the variable to look for the assignment
    """
    line = line.strip()
    if not line.startswith('!'): # to ignore fortran commented lines
        result = re.findall(r'.*?' + variable + '=(\d*)', line)
        if result :
            print(variable + ': ' + result[0])# + ' [' + line + ']')            
            return int(result[0])
        else:
            result = re.findall(r'(\d?\.\d.*?\s).*?(#*' + variable + ')', line)
            if result:
                print(variable + ': ' + result[0][0])
                return float(result[0][0])
            else:
                if '::' not in line:
                    result = re.findall(r'(\d*).*?(#*' + variable + ')', line)
                    if result and result[0] and result[0][0]:
                        #print(len(result))
                        #print(type(result))
                        print(variable + ': ' + result[0][0])
                        return int(result[0][0])
                    else:
                        return 0
                else:
                    return 0
    else:
        return 0

parameters = {}
parameters = ['nx', 'ny', 'nz', 'Lx', 'Ly', 'Lz', 'nclx', 'ncly', 'nclz', 'ifirst', 'ifin', 'imodulo']

for parameter in parameters:
    simulation[parameter] = 0

for line in module_param:
    for parameter in parameters:
        parameterValue = getVarFromFortranSource(line, parameter)
        if parameterValue:
            simulation[parameter] = parameterValue
for line in incompact3d:
    for parameter in parameters:
        parameterValue = getVarFromFortranSource(line, parameter)
        if parameterValue:
            simulation[parameter] = parameterValue

###############################################################
def di_calc(ncli, Li, ni):
    if ncli == 0:
        return float(Li) / (float(ni) - 0)
    else:
        return float(Li) / (float(ni) - 1)
simulation['dx'] = di_calc(simulation['nclx'],simulation['Lx'],simulation['nx'])
simulation['dy'] = di_calc(simulation['ncly'], simulation['Ly'], simulation['ny'])
simulation['dz'] = di_calc(simulation['nclz'], simulation['Lz'], simulation['nz'])

simulation['ifinal'] = 100

x = np.arange(0, simulation['Lx'] + simulation['dx']/2, simulation['dx'])
y = np.arange(0, simulation['Ly'] + simulation['dy']/2, simulation['dy'])
z = np.arange(0, simulation['Lz'] + simulation['dz']/2, simulation['dz'])

# plt.style.use('bmh')
# plt.style.use('fivethirtyeight')
# plt.style.use('ggplot')
# plt.style.use('grayscale')

X, Y = np.meshgrid(x, y)


#@jit
def plot_plane_xy_avg (field):
    for i in range(simulation['ifirst'], simulation['ifinal']):  # UX
        filename = simulation['prefix'] + '/' + simulation['name'] + simulation['dataplace'] + field + str(i).zfill(4)
        print(filename)
        A = np.fromfile(filename, dtype=np.float32)
        #np.set_printoptions(threshold=np.nan)
        print(A)
        #pprint.pprint(locals())
        print('shape of A: ' + str(A.shape))
        print('simulation[\'nx\']: ' + str(simulation['nx']))
        print('simulation[\'ny\']: ' + str(simulation['ny']))
        print('simulation[\'nz\']: ' + str(simulation['nz']))
        print('simulation[\'nx\'] x simulation[\'ny\' x simulation[\'nz\']: ' + str(simulation['nx'] * simulation['ny'] * simulation['nz'] ))

        A = A.reshape((simulation['nx'], simulation['ny'], simulation['nz']), order='FORTRAN')     
        #A = A.reshape((simulation['nx'], simulation['ny']), order='FORTRAN')     
        plt.figure(1, figsize=(simulation['Lx'], simulation['Ly'] * 1.5))
        plt.axes().set_aspect('equal')
        plt.ylabel(r'$x$')
        plt.xlabel(r'$y$')
        plt.contourf(X, Y, np.transpose(A[:,:,0]), 256, cmap=cm, vmin=0., vmax=1.)
        plt.colorbar()
        plt.grid(True)
        plt.minorticks_on()        
        #plt.savefig(simulation['output'] +   '/plano_' + field + str(i).zfill(4) + '.svg', format='svg', dpi=300)  # 'pdf')
        plt.savefig(simulation['output'] +   '/plano_' + field + str(i).zfill(4) + '.png', format='png', dpi=300)  # 'pdf')

        print(field + str(i).zfill(4))

        plt.close('all')

plot_plane_xy_avg('phi1')

"""

plot_plane_xy_avg('uy')
plot_plane_xy_avg('dissm')

nphi = 1
for j in range(1, nphi + 1):
    for i in range(simulation['ifirst'], simulation['ifinal']):  # PHI
        A = np.fromfile('data/phim' + str(j).zfill(1) + str(i).zfill(4), dtype=np.float32)
        print('data/phim' + str(j).zfill(1) + str(i).zfill(4))
        A = A.reshape((nx, ny), order='FORTRAN')
        plt.figure(1, figsize=(Lx, Ly * 1.5))
        plt.axes().set_aspect('equal')
        plt.ylabel(r'$x$')
        plt.xlabel(r'$y$')
        plt.contourf(X, Y, np.transpose(A), 256, cmap=cm, vmin=0., vmax=1.)
        plt.colorbar()
        plt.grid(True)
        plt.minorticks_on()
        plt.savefig('out_planes/plano_' + 'phim' + str(j).zfill(1) + str(i).zfill(4), format='png', dpi=300)  # 'pdf')
        print('phim' + str(j).zfill(1) + str(i).zfill(4))
        plt.close('all')


''' vvv PROBLEM vvv '''
for i in range(simulation['ifirst'] + 1, simulation['ifinal']):  # PRE !no prem0000
    A = np.fromfile('data/prem/' + str(i).zfill(4), dtype=np.float32)
    A = A.reshape((nx, ny), order='FORTRAN')
    plt.figure(1, figsize=(Lx, Ly * 1.5))
    plt.axes().set_aspect('equal')
    plt.ylabel(r'$x$')
    plt.xlabel(r'$y$')
    plt.contourf(X, Y, np.transpose(A), 256, cmap=cm)
    plt.colorbar()
    plt.grid(True)
    plt.minorticks_on()
    plt.savefig('out_planes/plano_' + 'prem' + str(i).zfill(4), format='png', dpi=300)  # 'pdf')
    print('prem' + str(i).zfill(4))
    plt.close('all')
''' ^^^ PROBLEM ^^^ '''

X, Z = np.meshgrid(x, z)

for i in range(simulation['ifirst'], simulation['ifinal']):  # TMAP
    A = np.fromfile('data/tmap' + str(i).zfill(4), dtype=np.float32)
    A = A.reshape((nx, nz), order='FORTRAN')
    plt.figure(1, figsize=(Lx, Lz*1.5))
    plt.axes().set_aspect('equal')
    plt.ylabel(r'$x_2$')
    plt.xlabel(r'$x_3$')
    plt.contourf(X, Z, np.transpose(A), 256, cmap=cm, vmin=0., vmax=0.5)
    plt.colorbar()
    plt.grid(True)
    plt.minorticks_on()
    plt.savefig('out_planes/plano_' + 'tmap' + str(i).zfill(4), format='png', dpi=300)  # 'pdf')
    print('tmap' + str(i).zfill(4))
    plt.close('all')

''' vvv PROBLEM vvv '''
for i in range(simulation['ifirst'], simulation['ifinal']):  # DMAP
    A = np.fromfile('data/dmap1' + str(i).zfill(4), dtype=np.float32)
    A = A.reshape((nx, nz), order='FORTRAN')
    plt.figure(1, figsize=(Lx, Lz*1.5))
    plt.axes().set_aspect('equal')
    plt.ylabel(r'$x_2$')
    plt.xlabel(r'$x_3$')
    plt.contourf(X, Z, np.transpose(A), 256, cmap=cm, vmin=0., vmax=0.5)
    plt.colorbar()
    plt.grid(True)
    plt.minorticks_on()
    plt.savefig('out_planes/plano_' + 'dmap1' + str(i).zfill(4), format='png', dpi=300)  # 'pdf')
    print('dmap1' + str(i).zfill(4))
    plt.close('all')
''' ^^^ PROBLEM ^^^ '''
"""
