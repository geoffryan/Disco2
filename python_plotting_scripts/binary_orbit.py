import math
import sys
import ode
import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt

def fprime(t, x, *args, **kwargs):
    deriv = np.zeros(x.shape)
    X = x[::4]
    Y = x[1::4]
    VX = x[2::4]
    VY = x[3::4]

    M = args[0]
    q = args[1]
    a = args[2]

    M1 = M / (1.0+q)
    M2 = M / (1.0+1.0/q)
    a1 = a / (1.0+1.0/q)
    a2 = a / (1.0+q)

    w = math.sqrt(M / (a*a*a))

    X1 = a1 * math.cos(w*t)
    Y1 = a1 * math.sin(w*t)
    X2 = -a2 * math.cos(w*t)
    Y2 = -a2 * math.sin(w*t)

    R1 = np.sqrt((X-X1)*(X-X1) + (Y-Y1)*(Y-Y1))
    R2 = np.sqrt((X-X2)*(X-X2) + (Y-Y2)*(Y-Y2))

    AX = -M1/  (R1*R1*R1) * (X-X1) - M2 / (R2*R2*R2) * (X-X2)
    AY = -M1/  (R1*R1*R1) * (Y-Y1) - M2 / (R2*R2*R2) * (Y-Y2)

    deriv[::4] = VX
    deriv[1::4] = VY
    deriv[2::4] = AX
    deriv[3::4] = AY

    return deriv

def evolve_rk(x0, t0, t1, n, step, *args, **kwargs):

    t = np.linspace(t0, t1, num=n+1)
    dt = (t1-t0)/n

    x = np.zeros((n+1,len(x0)))
    x[0,:] = x0

    for i in xrange(n):
        x[i+1] = step(t[i], x[i], fprime, dt, *args, **kwargs)

    return t, x

def plot_trajectory(t, x, M, q, a, rmax=-1.0):

    w = math.sqrt(M/(a*a*a))
    M1 = M/(1.0+q)
    M2 = M/(1.0+1.0/q)
    a1 = a/(1.0+1.0/q)
    a2 = a/(1.0+q)

    coswt = np.cos(w*t)
    sinwt = np.sin(w*t)

    X1 = a1*coswt
    Y1 = a1*sinwt
    X2 = -a2*coswt
    Y2 = -a2*sinwt
    X1com = a1*np.ones(t.shape)
    Y1com = a1*np.zeros(t.shape)
    X2com = -a2*np.ones(t.shape)
    Y2com = -a2*np.zeros(t.shape)

    N = x.shape[1]/4
    
    X = x[:, ::4]
    Y = x[:,1::4]

    Xcom =  X[:,:]*coswt[:,None] + Y[:,:]*sinwt[:,None]
    Ycom = -X[:,:]*sinwt[:,None] + Y[:,:]*coswt[:,None]

    fig, ax = plt.subplots(1,2,figsize=(12,9))

    ax[0].plot(X1, Y1, 'r')
    ax[0].plot(X2, Y2, 'b')
    for i in xrange(N):
        ax[0].plot(X[:,i], Y[:,i], 'k')
    ax[0].set_aspect("equal")

    for i in xrange(N):
        ax[1].plot(Xcom[:,i], Ycom[:,i], 'k')
    ax[1].plot(X1com, Y1com, 'r', ms=10, mew=0, marker='.', ls='')
    ax[1].plot(X2com, Y2com, 'b', ms=10, mew=0, marker='.', ls='')
    ax[1].set_aspect("equal")

    if rmax > 0.0:
        for axis in ax:
            axis.set_xlim([-rmax, rmax])
            axis.set_ylim([-rmax, rmax])

if __name__ == "__main__":
    
    if len(sys.argv) < 4:
        print("\nusage: python bodies.py nb n T <schemes...>\n")
        print("nb: Number of bodies (2 or 3)")
        print("n:  Number of time steps")
        print("T:  Total integration time")
        print("schemes: which numerical schemes to compare")
        print("    - fe:  Forward Euler")
        print("    - rk2: Runge-Kutta Second Order")
        print("    - rk4: Runge-Kutta Fourth Order")
        print("    - s1:  Symplectic First Order")
        print("    - s2:  Symplectic Second Order")
        print("    - s4:  Symplectic Fourth Order")
        print("\nexample: python bodies.py 3 1000 5 fe rk2 rk4\n")
        sys.exit()

    #M = 11.0
    #a = 1000.0
    #q = 0.1

    M = 1.0
    a = 1.0
    q = 0.1
    args = (M, q, a)

    w = math.sqrt(M/(a*a*a))

    t1 = float(sys.argv[1])
    n = int(sys.argv[2])
    rmax = float(sys.argv[3])

    t0 = 0.0

    #r0 = 1256.08
    #r0 = 1.13637

    w0 = w

    R0 = np.array([1.25608, 1.25609])

    x0 = np.zeros(4*len(R0))
    x0[ ::4] = -R0
    x0[1::4] = 0.0
    x0[2::4] = 0.0
    x0[3::4] = -R0*w0

    if "fe" in sys.argv:
        print("Forward Euler...")
        t, x = evolve_rk(x0, t0, t1, n, ode.forward_euler, M, q, a)
        print("   Plotting...")
        plot_trajectory(t, x, M, q, a, rmax)

    if "rk2" in sys.argv:
        print("RK2...")
        t, x = evolve_rk(x0, t0, t1, n, ode.rk2, M, q, a)
        print("   Plotting...")
        plot_trajectory(t, x, M, q, a, rmax)

    if "rk4" in sys.argv:
        print("RK4...")
        t, x = evolve_rk(x0, t0, t1, n, ode.rk4, M, q, a)
        print("   Plotting...")
        plot_trajectory(t, x, M, q, a, rmax)

    plt.show()
