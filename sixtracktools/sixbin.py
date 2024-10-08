# Read binary files fort.90 fort.89 ... prdouce by Sixtrack
#
# author: R. De Maria
#
# Copyright 2015 CERN. This software is distributed under the terms of the GNU
# Lesser General Public License version 2.1, copied verbatim in the file
#``COPYING''.
#
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.


import struct
import os
import numpy as np
import matplotlib.pyplot as plt



def _read(fh, fmt):
    out = {}
    for line in fmt.splitlines():
        lbl, spec, desc = line.split(None, 2)
        data = fh.read(struct.calcsize(spec))
        obj = struct.unpack(spec, data)
        if len(obj) == 1:
            obj = obj[0]
        out[lbl] = obj
    return out


fmt_head = """\
head1      1I Fortran header
title     80s General title of the run
title2    80s Additional title
date       8s Date
time       8s Time
progname   8s Program name
partfirst  1I First particle in the file
partlast   1I Last particle in the file
parttot    1I Total number of particles
spacecode  1I Code for dimensionality of phase space (1,2,4 are hor., vert. and longitudinal respectively)
turnproj   1I Projected number of turns
qx         1d Horizontal Tune
qy         1d Vertical Tune
qs         1d Longitudinal Tune
closorb    6d Closed Orbit vector
dispvec    6d Dispersion vector
tamatrix   36d Six-dimensional transfer map
mess1     50d 50 additional parameter
mess2      1I ...
"""
"""
seedmax    1d Maximum number of different seeds
seednum    1d Actual seed number
seedstart  1d Starting value of the seed
turnrev    1d Number of turns in the reverse direction (IBM only)
lyapcor1   1d Correction-factor for the Lyapunov (sigma=s - v0 t)
lyapcor2   1d Correction-factor for the Lyapunov (DeltaP/P0)
turnrip    1d Start turn number for ripple prolongation
"""

fmt_part = """\
partnum  1I Particle number
partdist 1d Angular distance in phase space
x        1d x (mm)
xp       1d x'(mrad)
y        1d y (mm)
yp       1d y'(mrad)
sig      1d Path-length sigma=s - v0 t
delta    1d DeltaP/P0
energy   1d Energy (Mev)
"""


def readfile(fn):
    fh = open(fn, 'rb')
    header = _read(fh, fmt_head)
    partfirst = header['partfirst']
    partlast = header['partlast']
    parttot =  header['parttot']
    headers = [header]
    for n in range(1,parttot//2):
        headers.append(_read(fh, fmt_head))
    part = {}
    for ii in range(parttot):
        part[ii]=[]
    while fh.read(4) != b'':  # read(fh,'headpart 1I ...')
        turnnum = struct.unpack('I', fh.read(4))
        # read(fh,fmt_part)
        # read(fh,fmt_part)
        for i in range(partfirst, partlast+1):
            pnum1 = struct.unpack('I', fh.read(4))
            idx = pnum1[0]-1
            orb1 = struct.unpack('8d', fh.read(64))
            part[idx].append(pnum1+orb1)
        fh.read(4)  # read(fh,'headpart 1I ...')
    part={ii: np.array(pp) for ii,pp in part.items()}
    return header, part


class SixBin(object):
    def __init__(self, filename="singletrackfile.dat"):
        self.head, self.part = readfile(filename)

    def show(self):
        plt.show()

    def plot_lossturns(self):
        ii,nt=np.array([ (ii,len(pp)) for ii,pp in self.part.items() ]).T
        plt.figure(num="LossTurns")
        plt.plot(ii,nt)
        plt.xlabel("Particle index")
        plt.ylabel("Number of turns")

    def plot_phasespace(self,ii):
        pp=self.part[ii]
        x=pp[:,2]
        px=pp[:,3]
        y=pp[:,4]
        py=pp[:,5]
        sig=pp[:,6]
        delta=pp[:,7]

        fig,axs=plt.subplots(2,2,num=f"Particle {ii}")
        plt.suptitle(f"Particle {ii}")
        axs[0,0].plot(x,px,'.')
        axs[0,0].set_xlabel("x")
        axs[0,0].set_ylabel("px")

        axs[0,1].plot(y,py,'.')
        axs[0,1].set_xlabel("y")
        axs[0,1].set_ylabel("py")

        axs[1,0].plot(sig,delta,'.')
        axs[1,0].set_xlabel("sig")
        axs[1,0].set_ylabel("delta")

        axs[1,1].plot(delta,x,'.',label='x')
        axs[1,1].plot(delta,y,'.',label='y')
        axs[1,1].set_xlabel("delta")
        axs[1,1].set_ylabel("xy")
        axs[1,1].legend()


    def get_particle(self, part, row, m0=938272046.):
        pdist, x, xp, y, yp, sigma, delta, energy = self.part[part][row].T
        e0 = energy*1e6
        gamma0 = e0/m0
        beta0 = np.sqrt(1-1/gamma0**2)
        tau = sigma/beta0/1e3
        out = dict(x=x/1e3, px=xp*(1+delta)/1e3,
                   y=y/1e3, py=yp*(1+delta)/1e3,
                   delta=delta, e0=e0, tau=tau)
        return out


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        basedir = sys.argv[1]
    else:
        basedir = '.'
    print(basedir)
    head, part = opendir(basedir)

    # pdist,x,xp,y,yp,sigma,delta,energy=part[1].T

    # f=np.linspace(0,1,len(x))
    # tunx=np.fft.fft(x+1j*xp)
    # tuny=np.fft.fft(y+1j*yp)

    # plot(f,abs(tunx),label='qx')
    # plot(f,abs(tuny),label='qy')

# ta=array(head['tamatrix']).reshape(6,6)
# betxI=ta[0,0]**2+ta[0,1]**2
#
# J=array([[ 0.,  1.,  0.,  0.,  0.,  0.],
#         [-1.,  0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  1.,  0.,  0.],
#         [ 0.,  0., -1.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.,  1.],
#         [ 0.,  0.,  0.,  0., -1.,  0.]])
#
# dot(ta.T,dot(J,ta))
