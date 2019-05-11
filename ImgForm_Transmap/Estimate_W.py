import numpy as np

def estimate_W(T0,Imin,Amin):
    [h,w] = Imin.shape

    Jmin = np.zeros((h, w), dtype=np.float16)
    '''Jmin represents output image of min channel'''
    for i in range(h) :
        for j in range(w) :
            Jmin = (Imin[i,j] - Amin * 255)/max(0.1,T0[i,j]) + Amin

    N = np.std(Imin)/max(0.001,np.std(Jmin))
    if(N < 0.5):
        W0 = 0.9
    elif(N < 1):
        W0 = 0.8
    elif(N < 1.5):
        W0 = 0.7
    elif(N < 2) :
        W0 = 0.5
    else :
        W0 = 0.4
    '''w0 is constant based on N'''
    W = np.zeros((h, w), dtype=np.float)
    '''It contains omega values for each pixel'''

    Ivar = np.zeros((h, w), dtype=np.float)
    '''the matrix containing variance of pixel in local patch of radius 5'''
    for i in range(h):
        for j in range(w):
            if(i<5 and j<5):
                Ivar[i,j]= np.var(Imin[i:i + 6,j:j + 6])
            elif(i<5):
                Ivar[i,j]= np.var(Imin[i:i + 6,j-5:j + 6])
            elif(j<5):
                Ivar[i,j]= np.var(Imin[i-5:i + 6,j:j + 6])
            else:
                Ivar[i,j]= np.var(Imin[i-5:i + 6,j-5:j + 6])

    for i in range(h):
        for j in range(w):
            W[i,j] =(Imin[i,j]*(1-(Ivar[i,j]/np.max(Ivar)))*(1-W0)*W0/Amin)+W0
            '''calculating omega for each pixel'''

    return W

