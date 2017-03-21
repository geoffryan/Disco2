import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Please provide an emission.txt summary file.")
        sys.exit()

    filename = sys.argv[1]

    f = open(filename, "r")
    line = f.readline()
    f.close()

    nu = np.array([float(x) for x in line.split()])
    dat = np.loadtxt(filename, skiprows=1)
    t = dat[:,0]
    Fnu = dat[:,1:]

    fig, ax = plt.subplots(1,1)
    for i in range(len(t)):
        ax.plot(nu, Fnu[i,:])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$ (Hz)")
    ax.set_ylabel(r"$F_\nu \cdot D^2$ (erg/(s Hz))")

    ymax = ax.get_ylim()[1]
    ax.set_ylim([1.0e-10*ymax, ymax])

    fig.savefig("spectrum.pdf")
    plt.close(fig)

    fig2, ax2 = plt.subplots(1,1)
    for i in range(len(nu)):
        ax2.plot(t, Fnu[:,i])
    ax2.set_xscale("linear")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$t$ (s)")
    ax2.set_ylabel(r"$F_\nu \cdot D^2$ (erg/(s Hz))")
    ymax = ax2.get_ylim()[1]
    ax2.set_ylim([1.0e-10*ymax, ymax])
    fig2.savefig("lightcurve.pdf")

