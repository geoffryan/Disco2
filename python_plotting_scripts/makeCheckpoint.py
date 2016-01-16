import sys
import discopy as dp

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("makeCheckpoint.py - Make a checkpoint from an archive.")
        print("usage: python makeCheckpoint.py archive numprocs filename")

    archive = sys.argv[1]
    numproc = int(sys.argv[2])
    checkname = sys.argv[3]

    print("Loading grid...")
    g = dp.Grid(archive=archive)
    print("Saving...")
    g.saveCheckpoint(checkname, numProc=numproc, loadBalance="default")

