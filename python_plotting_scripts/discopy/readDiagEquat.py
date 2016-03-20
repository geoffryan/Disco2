import math
from itertools import imap, izip
import h5py as h5
import numpy as np

def readDiagEquat(filename):
    """
    Return the data from a DiagEquat file.
    """

    f = h5.File(filename, "r")
    equat = f['EQUAT'][:,:7][...]
    f.close()

    r = equat[:,0]
    piph = equat[:,1]
    rho = equat[:,2]
    P = equat[:,3]
    vr = equat[:,4]
    vp = equat[:,5]
    q = equat[:,6]

    phi = np.zeros(piph.shape)

    t = float(filename.split("_")[1][:-3])

    #Find Cell Center phi's
    Rs = np.unique(r)
    for R in Rs:
        ind = (r==R)
        my_piph = piph[ind]
        dphi = np.zeros(my_piph.shape)
        dphi[1:] = my_piph[1:] - my_piph[:-1]
        dphi[0] = my_piph[0] - my_piph[-1]
        dphi[dphi < 0] += 2*np.pi
        dphi[dphi > 2*np.pi] -= 2*np.pi

        my_phi = my_piph - 0.5*dphi
        phi[ind] = my_phi[:]
    
    # THESE ARE PLACEHOLDERS.
    w = 0.0
    vz = 0.0
    dV = 0.0

    return t, r, phi, rho, P, vr, vp, vz, w, dV

