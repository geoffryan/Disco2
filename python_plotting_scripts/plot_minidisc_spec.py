import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

blue = (31.0/255, 119.0/255, 180.0/255)
orange = (255.0/255, 127.0/255, 14.0/255)
green = (44.0/255, 160.0/255, 44.0/255)
red = (214.0/255, 39.0/255, 40.0/255)
purple = (148.0/255, 103.0/255, 189.0/255)

def unpickle(filename):
    f = open(filename, "r")
    data = pickle.load(f)
    f.close()
    return data

if __name__ == "__main__":

    fig, ax = plt.subplots(1,1)

    color=[blue, orange, green]
    marker=['o', '+', '^']
    names = [r'Model 1', r'Model 2', r'Model 3']

    for i, file in enumerate(sys.argv[1:]):
        data = unpickle(file)

        nu = data['nu']
        Fnu = data['Fnu']
        FnuNT = data['FnuNT1']

        ax.plot(1.0e-3*nu, FnuNT, color=color[i], lw=10.0, alpha=0.5)
        ax.plot(1.0e-3*nu, Fnu, color=color[i], marker=marker[i], lw=0.0, 
                mew=2, ms=8, mec=color[i], label=names[i])

    ax.legend(loc="lower left")
    ax.set_ylim(1.0e-13, 1.0e-7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$ $(keV)$", fontsize=18)
    ax.set_ylabel(r"$F_\nu$ $(erg/cm^2 s Hz)$", fontsize=18)

    fig.savefig("spec_all.pdf")


