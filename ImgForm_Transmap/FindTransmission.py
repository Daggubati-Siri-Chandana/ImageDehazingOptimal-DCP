from ImgForm_Transmap.Airlight import *
from ImgForm_Transmap.Estimate_W import *

def Transmission_map(rgb):
    [h, w, k] = rgb.shape
    red = rgb[:, :, 2]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 0]
    Ar = airLight(red)
    Ag = airLight(green)
    Ab = airLight(blue)

    s = 3
    # size of local patch
    mincolor = np.zeros((h + s - 1, w + s - 1), dtype=np.float32)
    '''to find min valu among channels and construct h,w image   minc(Ic/Ac)'''
    minI = np.zeros((h, w), dtype=np.uint8)
    '''minchannel of image'''
    for i in range(h):
        for j in range(w):
            minI[i,j]=min(red[i,j], green[i,j], blue[i,j])
            mincolor[i, j] = min(red[i,j]/Ar, green[i,j]/Ag, blue[i,j]/Ab)

    """To extend the size of image for convolution multiplication
    by copying the top rows to bottom and right colums to left"""
    for i in range(s - 1):
        mincolor[h + i] = mincolor[i]
    for i in range(h + s - 1):
        for j in range(s - 1):
            mincolor[i, w + j] = mincolor[i, j]

    '''To normalise values after division from 0 to 255'''
    minp = np.amin(mincolor)
    maxp = np.amax(mincolor)
    for i in range(h):
        for j in range(w):
            mincolor[i, j] = (mincolor[i, j] - minp) * 255.0 / (maxp - minp)

    minpatch = np.zeros((h, w), dtype=np.uint8)
    '''To find min values in local window this is final transmission map'''
    omega = 0.95
    '''generally omega between 0 and 1 take a arbitray value and 
    find transmission map and use it find exact W for an image'''
    for i in range(h):
        for j in range(w):
            matx = np.array(mincolor[i:i + s, j:j + s])
            matx.resize((s,s))  # at edges we don't get n*n matrix, we will make other values to zero so that the size becomes n*
            minpatch[i, j] = int(255.0 - omega * (matx.min()))  # finding minimum of local patch and subracting from 255

    Amin=min(Ab,Ag,Ar)

    omega = estimate_W(minpatch, minI, Amin)
    for i in range(h):
        for j in range(w):
            matx = np.array(mincolor[i:i + s, j:j + s])
            matx.resize((s,s))  # at edges we don't get n*n matrix, we will make other values to zero so that the size becomes n*
            minpatch[i, j] = int(255.0 - omega[i,j] * (matx.min()))  # finding minimum of local patch and subracting from 255

    return minpatch,[Ab,Ag,Ar]
