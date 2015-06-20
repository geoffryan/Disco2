import sys
import discopy as dp



if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("makeArchive.py - turn parfile and checkpoints into archives.")
        print("usage: python makeArchive.py parfile checkpoints ...")

    pars = dp.readParfile(sys.argv[1])

    checkpoints = sys.argv[2:]

    g = dp.Grid(pars)

    for f in checkpoints:
        archiveName = f.replace("checkpoint", "archive")
        print("Loading {0:s}".format(f))
        g.loadCheckpoint(f)
        print("Saving {0:s}".format(archiveName))
        g.saveArchive(archiveName)
