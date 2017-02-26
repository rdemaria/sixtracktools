# Read binary files  produce by Sixtrack Dump module
#
# author: R. De Maria
#
#Copyright 2017 CERN. This software is distributed under the terms of the GNU
#Lesser General Public License version 2.1, copied verbatim in the file
#``COPYING''.
#
#In applying this licence, CERN does not waive the privileges and immunities
#granted to it by virtue of its status as an Intergovernmental Organization or
#submit itself to any jurisdiction.

import os,gzip

import numpy as np

dump3_t=np.dtype([
        ('dummy'  ,'I'),   #Record size
        ('partid','I'),    #Particle number
        ('turn'  ,'I'),    #Turn number
        ('s'     ,'d'),    #s [m]    -dcum
        ('x'     ,'d'),    #x [mm]   -xv(1,j)
        ('xp'    ,'d'),    #x'[mrad] -yv(1,j)
        ('y'     ,'d'),    #y [mm]   -xv(2,j)
        ('yp'    ,'d'),    #y'[mrad] -yv(2,j)
        ('sigma' ,'d'),    #delay  s - v0 t [mm] - sigmv(j)
        ('deltaE','d'),   #DeltaE/E0 [1] - (ejv(j)-e0)/e0
        ('elemid','I'),    #Element type  - ktrack(i)
        ('energy', 'd'),   #  - ejv(j)
        ('pc'    , 'd'),   #  - ejfv(j)
        ('delta' , 'd'),   #  - dpsv(j)
        ('rpp'   , 'd'),   #P0/P=1/(1+delta) - oidpsv(j) 
        ('rvv'   , 'd'),   #beta0/beta - rvv(j) = (ejv(j)*e0f)/(e0*ejfv(j))
        ('mass0' , 'd'),   # pma
        ('energy0','d'),   # e0
        ('p0c'   , 'd'),   # e0f
        ('dummy2','I')])


def read_dump3(fn):
  if fn.endswith('.gz'):
    fh=gzip.open(fn+'.gz','rb')
    return np.fromstring(fh.read(),dump3_t)
  else:
    return np.fromfile(fn,dump3_t)

class SixDump3(object):
    def __init__(self,filename):
        self.filename=filename
        self.particles=read_dump3(filename)
        self.particles['x']      /=1e3
        self.particles['xp']     /=1e3
        self.particles['y']      /=1e3
        self.particles['yp']     /=1e3
        self.particles['sigma']  /=1e3
        self.particles['mass0']  *=1e6
        self.particles['p0c']    *=1e6
        self.particles['energy0']*=1e6
        self.particles['energy']*=1e6
        self.particles['pc']*=1e6
    px    =property(lambda p: p.xp/p.rpp)
    py    =property(lambda p: p.yp/p.rpp)
    ptau  =property(lambda p: (p.energy-p.energy0)/p.p0c)
    psigma=property(lambda p: p.ptau/p.beta0)
    tau   =property(lambda p: p.sigma*p.beta0)
    mass  =property(lambda p: p.mass0)
    charge=property(lambda p: 1)
    charge0=property(lambda p: 1)
    qratio=property(lambda p: p.charge/p.charge0)
    mratio=property(lambda p: p.mass/p.mass0)
    state =property(lambda p: 0)
    chi   =property(lambda p: p.qratio/p.mratio)
    gamma0=property(lambda p: p.energy0/p.mass0)
    beta0 =property(lambda p: p.p0c/p.energy0)
    gamma=property(lambda p: p.energy/p.mass)
    beta =property(lambda p: p.pc/p.energy)
    def __getattr__(self,k):
        return self.particles[k]
    def __dir__(self):
        return sorted(self.__dict__.keys()+list(self.particles.dtype.names))
    def get_full_beam(self):
        out={}
        names ='partid elemid turn state'.split()
        names+='s x px y py tau ptau sigma psigma delta'.split()
        names+='rpp rvv beta gamma energy pc'.split()
        names+='mass charge mratio qratio chi'.split()
        names+='p0c energy0 mass0 gamma0 beta0 charge0'.split()
        for name in names:
            out[name]=getattr(self,name)
        return out
    def get_minimal_beam(self):
        out={}
        names ='partid elemid turn state'.split()
        names+='s x px y py tau ptau delta'.split()
        names+='mass energy pc'.split()
        names+='mass0 energy0 p0c'.split()
        for name in names:
            out[name]=getattr(self,name)
        return out

