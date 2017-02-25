# Read binary files  produce by Sixtrack Dump module
#
# author: R. De Maria
#
#Copyright 2016 CERN. This software is distributed under the terms of the GNU
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
        ('s'     ,'d'),    #s [m]
        ('x'     ,'d'),    #x [mm]
        ('xp'    ,'d'),    #x'[mrad]
        ('y'     ,'d'),    #y [mm]
        ('yp'    ,'d'),    #y'[mrad]
        ('sig'   ,'d'),    #Path-length sigma=s - v0 t [mm]
        ('delta' ,'d'),    #DeltaE/E0 [1]
        ('ktrack','I'),    #Element type
        ('dummy2'  ,'I')])



def read_dump3(fn):
  if fn.endswith('.gz'):
    fh=gzip.open(fn+'.gz','rb')
    return np.fromstring(fh.read(),dump3_t)
  else:
    return np.fromfile(fn,dump3_t)

class SixDump3(object):
   def __init__(self,filename):
      self.filename=filename
      self._data=read_dump3(filename)


