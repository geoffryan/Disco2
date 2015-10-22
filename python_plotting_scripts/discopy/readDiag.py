import math
import h5py as h5
import numpy as np

def readDiagEquat(filename):
    """
    Return the data from a DiagEquat file.
    """

    f = h5.File(filename, "r")
    Data = f['EQUAT'][...]
    f.close()
    t = float(filename.split("_")[1][:-3])

    r = Data[:,0]
    phi = Data[:,1]
    rho = Data[:,2]
    P = Data[:,3]
    vr = Data[:,4]
    vp = Data[:,5]
    vz = Data[:,6]
    w = Data[:,5] - Data[:,7]
    dV = Data[:,9]

    return t, r, phi, rho, P, vr, vp, vz, w, dV

