#! /usr/bin/env python

"""
Plot accretion disk variables
"""

# standard python modules
import argparse
import numpy as np
import matplotlib.pyplot as plt

def blackbody(teff, nu):
  """
  Calculates blackbody spectrum
  """
  h = 6.62607015e-27; # planck constant [cgs]
  kb = 1.380649e-16 # bolztmann constant [cgs]
  c = 2.99792458e10 # speed of light [cgs]
  efact = np.exp(h*nu/(kb*teff))
  bnu = 2.*h*nu**3/(c**2*(efact - 1.)) # [cgs] = [erg s^-1 cm^-2 Hz^-1]
  return bnu

def grcor(abh,r):
  """
  Pythonized version of dispar.f routine used with kerrtrans
  """

  if(abh < 0.):
    signa=-1.
  else:
    signa=1.0

  abh2 = abh * abh
  r1 = 1.0/r
  r12 = np.sqrt(r1)
  r2 = r1*r1
  a2r2 = abh2*r2
  a4r4 = a2r2*a2r2
  a2r3 = abh2*r2*r1
  ar32 = np.sqrt(a2r3)

  A = 1 + a2r2 + 2.*a2r3
  B = 1 + ar32
  C = 1 - 3.*r1 + 2.*ar32
  D = 1 - 2.*r1 + a2r2
  E = 1 + 4.*a2r2 - 4.*a2r3 + 3*a4r4

  # gravity correction  (see Page & Thorne,'73, eq.35) 
  qcor = (1. - 4.*ar32 + 3.*a2r2)/C

  # Minimum radius for last stable circular orbit per unit mass, x0
  z1 = 1.0+(1.0-abh2)**(1.0/3.0)*((1.0+abh)**(1.0/3.0)+(1.0-abh)**(1.0/3.0))
  z2 = np.sqrt(3.0*abh2+z1*z1)
  r0 = 3.0+z2-signa*np.sqrt((3.0-z1)*(3.0+z1+2.0*z2))
  x0 = np.sqrt(r0)

  # roots of x^3 - 3x + 2a = 0
  ca3 = 1.0/3.0 * np.arccos(abh)
  x1 =  2.*np.cos(ca3-np.pi/3.)
  x2 =  2.*np.cos(ca3+np.pi/3.)
  x3 = -2.*np.cos(ca3)

  # FB = '[]' term in eq. (35) of Page&Thorne '73
  x = np.sqrt(r)
  c1 = 3*(x1-abh)*(x1-abh)/(x1*(x1-x2)*(x1-x3))
  c2 = 3*(x2-abh)*(x2-abh)/(x2*(x2-x1)*(x2-x3))
  c3 = 3*(x3-abh)*(x3-abh)/(x3*(x3-x1)*(x3-x2))
  al0 = 1.5*abh * np.log(x/x0)
  al1 = np.log((x-x1)/(x0-x1))
  al2 = np.log((x-x2)/(x0-x2))
  al3 = np.log((x-x3)/(x0-x3))
  fb = (x-x0 - al0 - c1*al1 - c2*al2 - c3*al3)
  Q = fb*(1.0+ar32)*r12/np.sqrt(1.0-3.0*r1+2.0*ar32)

  # temperature correction
  tcor = (Q/B/np.sqrt(C))**0.25

  inds0 = np.where(r < r0)

  #nx = r.shape[0]
  #ny = r.shape[1]

  tcor[inds0] = 0.
  qcor[inds0] = 0.

  return qcor,tcor


def DiskFlux(mass,mdot,radius,abh):
    """
    Returns the flux as a function of radius, mass, accretion rate but without
    relativistic and no-torque correction, which is provided by grcor
    """
    G = 6.67430e-8
    msun = 1.99e33
    kapes = 0.34
    c = 2.99792458e10

    # Minimum radius for last stable circular orbit per unit mass, X0
    abh2 = abh*abh
    signa = 1.0
    z1 = 1.0+(1.0-abh2)**(1.0/3.0)*((1.0+abh)**(1.0/3.0)+(1.0-abh)**(1.0/3.0))
    z2 = np.sqrt(3.0*abh2+z1*z1)
    r0 = 3.0+z2-signa*np.sqrt((3.0-z1)*(3.0+z1+2.0*z2))

    eta = 1.0 - (r0**2 - 2.0*r0 + abh*r0**0.5)/r0/(r0**2 - 3.0*r0 + 2.0*abh*r0**0.5)**0.5

def rms(abh):

    abh2 = abh*abh
    if (abh < 0):
        signa = -1.
    else:
        signa = 1.
    z1 = 1.0+(1.0-abh2)**(1.0/3.0)*((1.0+abh)**(1.0/3.0)+(1.0-abh)**(1.0/3.0))
    z2 = np.sqrt(3.0*abh2+z1*z1)
    r0 = 3.0+z2-signa*np.sqrt((3.0-z1)*(3.0+z1+2.0*z2))
    return r0

def plot_rr():
    """
    plot the R_R factor
    """
    nr = 1000
    rmax = 1000.
    rmin = 1.
    # plot a= 0 first
    abh = 0.
    r = np.logspace(np.log10(rmin),np.log10(rmax),nr)
    # Krolik's R_R is tcor**4
    qcor, tcor = grcor(abh,r)
    plt.plot(r,tcor**4,'k', label='$a_*=0$')

    # plot a = 0.99
    abh = 0.99
    qcor, tcor = grcor(abh,r)
    plt.plot(r,tcor**4,'k',linestyle='dotted', label='$a_*=0.99$')

    # plot newtonian
    fnewt = 1.-np.sqrt(6./r)
    fnewt[(fnewt < 0.)] = 0.
    plt.plot(r,fnewt,'k',linestyle='dashed', label='Newtonian')

    plt.ylabel(r'$R_R$')
    plt.xlabel(r'$r/r_g$')
    plt.xscale('log')
#    plt.savefig("flux_cor.png")
    plt.legend()
    plt.show()

# Main function
def main(**kwargs):

    plot = kwargs.pop('plot')
    if (plot is None):
        # generate all plots
        plot_rr()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot',
        default=None,
        help='specific plot')

    args = parser.parse_args()
    main(**vars(args))

