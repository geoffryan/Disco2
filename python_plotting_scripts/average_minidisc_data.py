import sys
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt

blue = (31.0/255, 119.0/255, 180.0/255)
orange = (255.0/255, 127.0/255, 14.0/255)
green = (44.0/255, 160.0/255, 44.0/255)
red = (214.0/255, 39.0/255, 40.0/255)
purple = (148.0/255, 103.0/255, 189.0/255)

def get_data(filename):

    f = open(filename, "r")
    data = pickle.load(f)
    f.close()

    return data

def save_data(data, filename):

    f = open(filename, "w")
    pickle.dump(data, f, protocol=-1)
    f.close()

def avg_data(datas):

    if len(datas) == 1:
        return data

    T = []

    data0 = datas[0]
    R = data0['R']
    A = data0['A']
    M = data0['M']
    M2 = data0['M2']
    bw = data0['bw']
    ba = data0['ba']
    Mdot = data0['Mdot']
    gam = data0['gam']
    J0 = np.zeros(R.shape)
    Jr = np.zeros(R.shape)
    Tp0 = np.zeros(R.shape)
    Tpr = np.zeros(R.shape)
    T00 = np.zeros(R.shape)
    T0r = np.zeros(R.shape)
    sig = np.zeros(R.shape)
    pi = np.zeros(R.shape)
    vr = np.zeros(R.shape)
    vp = np.zeros(R.shape)
    phiSa = np.zeros((R.shape[0],2))
    phiSb = np.zeros((R.shape[0],2))
    siga = np.zeros((R.shape[0],2))
    sigb = np.zeros((R.shape[0],2))
    pia = np.zeros((R.shape[0],2))
    pib = np.zeros((R.shape[0],2))
    vra = np.zeros((R.shape[0],2))
    vrb = np.zeros((R.shape[0],2))
    vpa = np.zeros((R.shape[0],2))
    vpb = np.zeros((R.shape[0],2))
    psiQa = np.zeros(R.shape)
    psiQb = np.zeros(R.shape)
    dQdma = np.zeros(R.shape)
    dQdmb = np.zeros(R.shape)
    dQdra = np.zeros(R.shape)
    dQdrb = np.zeros(R.shape)
    dCool = np.zeros(R.shape)
    l = np.zeros(R.shape)
    Tmdot = np.zeros(R.shape)
    Tre   = np.zeros(R.shape)
    Text  = np.zeros(R.shape)
    Tcool = np.zeros(R.shape)
    Tpsiq = np.zeros(R.shape)

    n = len(datas)

    cA = np.zeros(R.shape, dtype=np.int)
    cB = np.zeros(R.shape, dtype=np.int)

    iSa = data0['iSa']
    iSb = data0['iSb']

    for data in datas:
        T.append(data['T'])
        J0[:] += data['J0'] / n
        Jr[:] += data['Jr'] / n
        Tp0[:] += data['Tp0'] / n
        Tpr[:] += data['Tpr'] / n
        T00[:] += data['T00'] / n
        T0r[:] += data['T0r'] / n
        sig[:] += data['sig'] / n
        pi[:] += data['pi'] / n
        vr[:] += data['vr'] / n
        vp[:] += data['vp'] / n
        dCool[:] += data['dCool'] / n
        l[:] += data['l'] / n
        Tmdot[:] += data['Tmdot'] / n
        Tre[:] += data['Tre'] / n
        Text[:] += data['Text'] / n
        Tcool[:] += data['Tcool'] / n
        Tpsiq[:] += data['Tpsiq'] / n

        indA = data['iSa'][:,0] >= 0
        indB = data['iSb'][:,0] >= 0
        cA[indA] += 1
        cB[indB] += 1

        iSa[(iSa[:,0]>=0)*indA,:] = data['iSa'][(iSa[:,0]>=0)*indA,:]
        iSb[(iSb[:,0]>=0)*indB,:] = data['iSb'][(iSb[:,0]>=0)*indB,:]

        phiSa[indA,:] += data['phiSa'][indA,:]
        siga[indA,:] += data['siga'][indA,:]
        pia[indA,:] += data['pia'][indA,:]
        vra[indA,:] += data['vra'][indA,:]
        vpa[indA,:] += data['vpa'][indA,:]
        psiQa[indA] += data['psiQa'][indA]
        dQdma[indA] += data['dQdma'][indA]
        dQdra[indA] += data['dQdra'][indA]
        
        phiSb[indB,:] += data['phiSb'][indB,:]
        sigb[indB,:] += data['sigb'][indB,:]
        pib[indB,:] += data['pib'][indB,:]
        vrb[indB,:] += data['vrb'][indB,:]
        vpb[indB,:] += data['vpb'][indB,:]
        psiQb[indB] += data['psiQb'][indB]
        dQdmb[indB] += data['dQdmb'][indB]
        dQdrb[indB] += data['dQdrb'][indB]

    phiSa[cA>0] /= (cA[cA>0])[:,None]
    siga[cA>0] /= (cA[cA>0])[:,None]
    pia[cA>0] /= (cA[cA>0])[:,None]
    vra[cA>0] /= (cA[cA>0])[:,None]
    vpa[cA>0] /= (cA[cA>0])[:,None]
    psiQa[cA>0] /= cA[cA>0]
    dQdma[cA>0] /= cA[cA>0]
    dQdra[cA>0] /= cA[cA>0]
    phiSb[cB>0] /= (cB[cB>0])[:,None]
    sigb[cB>0] /= (cB[cB>0])[:,None]
    pib[cB>0] /= (cB[cB>0])[:,None]
    vrb[cB>0] /= (cB[cB>0])[:,None]
    vpb[cB>0] /= (cB[cB>0])[:,None]
    psiQb[cB>0] /= cB[cB>0]
    dQdmb[cB>0] /= cB[cB>0]
    dQdrb[cB>0] /= cB[cB>0]

    print cA
    print cB

    data = {'T': np.array(T),
            'R': R,
            'A': A,
            'M': M,
            'M2': M2,
            'bw': bw,
            'ba': ba,
            'Mdot': Mdot,
            'gam': gam,
            'sig': sig,
            'pi': pi,
            'vr': vr,
            'vp': vp,
            'J0': J0,
            'Jr': Jr,
            'Tp0': Tp0,
            'Tpr': Tpr,
            'T00': T00,
            'T0r': T0r,
            'dCool': dCool,
            'iSa': iSa,
            'iSb': iSb,
            'phiSa': phiSa,
            'phiSb': phiSb,
            'siga': siga,
            'sigb': sigb,
            'pia': pia,
            'pib': pib,
            'vra': vra,
            'vrb': vrb,
            'vpa': vpa,
            'vpb': vpb,
            'psiQa': psiQa,
            'psiQb': psiQb,
            'dQdma': dQdma,
            'dQdmb': dQdmb,
            'dQdra': dQdra,
            'dQdrb': dQdrb,
            'l': l,
            'Tmdot': Tmdot,
            'Tre': Tre,
            'Text': Text,
            'Tcool': Tcool,
            'Tpsiq': Tpsiq}

    return data


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("usage: python average_minidisc_data.py datfile.dat... filename")
        sys.exit()

    datas = []
    for fname in sys.argv[1:-1]:
        datas.append(get_data(fname))
    data = avg_data(datas)

    save_data(data, sys.argv[-1])

