# Read binary files  produce by Sixtrack Dump module
#
# author: R. De Maria
#
# Copyright 2017 CERN. This software is distributed under the terms of the GNU
# Lesser General Public License version 2.1, copied verbatim in the file
#``COPYING''.
#
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

import os
import gzip

import numpy as np


def read_dump_bin(fn, dump_t):
    if fn.endswith('.gz'):
        fh = gzip.open(fn+'.gz', 'rb')
        return np.fromstring(fh.read(), dump_t)
    else:
        return np.fromfile(fn, dump_t)


dump3_t = np.dtype([
    ('dummy', 'I'),  # Record size
    ('partid', 'I'),  # Particle number
    ('turn', 'I'),  # Turn number
    ('s', 'd'),  # s [m]    -dcum
    ('x', 'd'),  # x [mm]   -xv(1,j)
    ('xp', 'd'),  # x'[mrad] -yv(1,j)
    ('y', 'd'),  # y [mm]   -xv(2,j)
    ('yp', 'd'),  # y'[mrad] -yv(2,j)
    ('deltaE', 'd'),  # DeltaE/E0 [1] - (ejv(j)-e0)/e0
    ('sigma', 'd'),  # delay  s - v0 t [mm] - sigmv(j)
    ('ktrack', 'I'),  # ktrack
    ('dummy2', 'I')])


pmass = 0.9376e9


class SixDump3(object):
    def __init__(self, filename, energy0=450e9, mass0=pmass):
        self.filename = filename
        self.particles = read_dump_bin(filename, dump3_t)
        self.particles['x'] /= 1e3
        self.particles['xp'] /= 1e3
        self.particles['y'] /= 1e3
        self.particles['yp'] /= 1e3
        self.particles['sigma'] /= 1e3
        self.particles['deltaE'] *= 1e6
        self.mass0 = mass0
        self.energy0 = energy0
    px = property(lambda p: p.xp/p.rpp)
    py = property(lambda p: p.yp/p.rpp)
    ptau = property(lambda p: (p.energy-p.energy0)/p.p0c)
    psigma = property(lambda p: p.ptau/p.beta0)
    tau = property(lambda p: p.sigma/p.beta0)
    z = property(lambda p: p.beta/p.beta0*p.sigma)
    mass = property(lambda p: p.mass0)
    charge = property(lambda p: 1)
    charge0 = property(lambda p: 1)
    qratio = property(lambda p: p.charge/p.charge0)
    mratio = property(lambda p: p.mass/p.mass0)
    state = property(lambda p: 0)
    chi = property(lambda p: p.qratio/p.mratio)
    gamma0 = property(lambda p: p.energy0/p.mass0)
    beta0 = property(lambda p: p.p0c/p.energy0)
    gamma = property(lambda p: p.energy/p.mass)
    beta = property(lambda p: p.pc/p.energy)

    def __getattr__(self, k):
        return self.particles[k]

    def __dir__(self):
        return sorted(list(self.__dict__.keys())+list(self.particles.dtype.names))

    def get_full_beam(self):
        out = {}
        names = 'partid elemid turn state'.split()
        names += 's x px y py tau ptau sigma psigma delta'.split()
        names += 'rpp rvv beta gamma energy pc'.split()
        names += 'mass charge mratio qratio chi'.split()
        names += 'p0c energy0 mass0 gamma0 beta0 charge0'.split()
        for name in names:
            out[name] = getattr(self, name)
        return out

    def get_minimal_beam(self):
        out = {}
        names = 'partid elemid turn state'.split()
        names += 's x px y py tau ptau delta'.split()
        names += 'mass energy pc'.split()
        names += 'mass0 energy0 p0c'.split()
        for name in names:
            out[name] = getattr(self, name)
        return out


dump101_t = np.dtype([
    ('dummy', 'I'),  # Record size
    ('partid', 'I'),  # Particle number
    ('turn', 'I'),  # Turn number
    ('s', 'd'),  # s [m]    -dcum
    ('x', 'd'),  # x [mm]   -xv(1,j)
    ('xp', 'd'),  # x'[mrad] -yv(1,j)
    ('y', 'd'),  # y [mm]   -xv(2,j)
    ('yp', 'd'),  # y'[mrad] -yv(2,j)
    ('sigma', 'd'),  # delay  s - v0 t [mm] - sigmv(j)
    ('deltaE', 'd'),  # DeltaE/E0 [1] - (ejv(j)-e0)/e0
    ('elemid', 'I'),  # Element type  - ktrack(i)
    ('energy', 'd'),  # - ejv(j)
    ('pc', 'd'),  # - ejfv(j)
    ('delta', 'd'),  # - dpsv(j)
    ('rpp', 'd'),  # P0/P=1/(1+delta) - oidpsv(j)
    ('rvv', 'd'),  # beta0/beta - rvv(j) = (ejv(j)*e0f)/(e0*ejfv(j))
    ('mass', 'd'),  # mass nucm(j) (ex. pma)
    ('chi', 'd'),  # mass to charge ratio mtc(j) m/m0*q0/q?
    ('energy0', 'd'),  # e0
    ('p0c', 'd'),  # e0f
    #('mass0', 'd'),  # mass nucm0 (ex. pma)
    ('dummy2', 'I')])


class SixDump101Abs(object):
    def __init__(self, particles):
        self.particles = particles

    def __getitem__(self, idx):
        return SixDump101Abs(self.particles[idx].copy())

    px = property(lambda p: p.xp/p.rpp)
    py = property(lambda p: p.yp/p.rpp)
    ptau = property(lambda p: (p.energy-p.energy0)/p.p0c)
    psigma = property(lambda p: p.ptau/p.beta0)
    tau = property(lambda p: p.sigma/p.beta0)
    mass = property(lambda p: p.mass0)
    charge = property(lambda p: 1)
    charge0 = property(lambda p: 1)
    qratio = property(lambda p: p.charge/p.charge0)
    mratio = property(lambda p: p.mass/p.mass0)
    state = property(lambda p: 0)
    chi = property(lambda p: p.qratio/p.mratio)
    gamma0 = property(lambda p: p.energy0/p.mass0)
    beta0 = property(lambda p: p.p0c/p.energy0)
    gamma = property(lambda p: p.energy/p.mass)
    beta = property(lambda p: p.pc/p.energy)
    mass0 = property(lambda p: np.sqrt(p.energy0**2-p.p0c**2))

    @property
    def zeta(self):
        raise ValueError('Not anymore supported, use tau')

    def __getattr__(self, k):
        return self.particles[k]

    def __dir__(self):
        return sorted(list(self.__dict__.keys()) +
                      list(self.particles.dtype.names))

    def get_full_beam(self):
        out = {}
        names = 'partid elemid turn state'.split()
        names += 's x px y py tau ptau sigma psigma delta'.split()
        names += 'rpp rvv beta gamma energy pc'.split()
        names += 'mass charge mratio qratio chi'.split()
        names += 'p0c energy0 mass0 gamma0 beta0 charge0'.split()
        for name in names:
            out[name] = getattr(self, name)
        return out

    def get_minimal_beam(self):
        out = {}
        names = 'partid elemid turn state'.split()
        names += 's x px y py tau delta'.split()
        names += 'mass0 p0c'.split()
        for name in names:
            out[name] = getattr(self, name)
        return out


class SixDump101(SixDump101Abs):
    def __init__(self, filename):
        particles = read_dump_bin(filename, dump101_t)
        particles['x'] /= 1e3
        particles['xp'] /= 1e3
        particles['y'] /= 1e3
        particles['yp'] /= 1e3
        particles['sigma'] /= 1e3
        particles['mass'] *= 1e6
        particles['p0c'] *= 1e6
        particles['energy0'] *= 1e6
        particles['energy'] *= 1e6
        particles['pc'] *= 1e6
        SixDump101Abs.__init__(self, particles)
        self.filename = filename
