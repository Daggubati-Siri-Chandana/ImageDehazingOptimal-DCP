from ImgForm_Transmap.FindTransmission import *
import cv2
import cv2.ximgproc

def optimizedTrans_map(rgb):
    [h,w,k] = rgb.shape
    t,A=Transmission_map(rgb)
    temp = np.zeros((h,w),dtype=np.float32)
    J1 = np.zeros((h,w,k),dtype=np.uint8)


    '''Dehazed image in iteration 1'''
    for c in range(k):
        for i in range(h):
            for j in range(w):
                temp[i,j]=(((rgb[i,j][c] - A[c])/max(t[i,j],0.1)))+A[c]
        '''This is to normalize the image matrix of iteration 1 from 0 - 255'''
        minp=np.amin(temp)
        maxp=np.amax(temp)
        for i in range(h):
            for j in range(w):
                J1[i, j][c] = int(((temp[i, j] - minp) * 255) / (maxp - minp))


    '''S is size of local patch for darkchannel'''
    s = 3
    J1minchannel = np.zeros((h+s-1,w+s-1),dtype=np.float32)
    Iminchannel = np.zeros((h+s-1,w+s-1),dtype=np.float32)
    maxI = np.zeros((h,w),dtype=np.uint8)
    maxJ = np.zeros((h,w),dtype=np.uint8)
    minI = np.zeros((h,w),dtype=np.uint8)
    minJ = np.zeros((h,w),dtype=np.uint8)
    '''min channel/airlight of output image and Input image for optimized transmission'''
    for i in range(h):
        for j in range(w):
            J1minchannel[i, j]=min(J1[i,j][0]/A[0], J1[i,j][1]/A[1], J1[i,j][2]/A[2])
            Iminchannel[i, j] = min(rgb[i, j][0] / A[0], rgb[i, j][1] / A[1], rgb[i, j][2] / A[2])
            minI[i, j] = min(rgb[i, j][0], rgb[i, j][1], rgb[i, j][2])
            minJ[i, j] = min(J1[i, j][0], J1[i, j][1], J1[i, j][2])
            maxI[i, j] = max(rgb[i, j][0], rgb[i, j][1], rgb[i, j][2])
            maxJ[i, j] = max(J1[i, j][0], J1[i, j][1], J1[i, j][2])
    '''to extend the size of image for convolution multiplication'''
    for i in range(s-1):
        J1minchannel[h+i] = J1minchannel[i]
        Iminchannel[h + i] = Iminchannel[i]
    for i in range(h+s-1):
        for j in range(s-1):
            J1minchannel[i,w+j] = J1minchannel[i,j]
            Iminchannel[i,w+j] = Iminchannel[i,j]


    '''default omega value '''
    cmax=(255-np.mean(maxI))*np.std(maxJ)/(np.std(maxI)*(255-np.mean(maxJ)))
    cmin=(255-np.mean(minI))*np.std(minJ)/(np.std(minI)*(255-np.mean(minJ)))
    if(cmax>1.0):
        cmax = cmin
    t2 = np.zeros((h,w),dtype=np.uint8)
    '''optimized transmission map'''
    for i in range(h):
        for j in range(w):
            temp[i,j] = (255.0 -(np.amin(Iminchannel[i:i + s, j:j + s])))/max(1,(np.amin(J1minchannel[i:i + s, j:j + s])/0.95))
    '''This is to normalize the optimized transmission map'''
    minp = np.amin(temp)
    maxp = np.amax(temp)
    for i in range(h):
        for j in range(w):
            t2[i, j] = int(((temp[i, j]- minp) * 255) / (maxp - minp))


    '''t2 is the obtained optimized transmission map with block effect ,so we apply
    guided filtering to smoothen the image and remove block effect'''

    '''destinationimg=cv.ximgproc.guidedFilter(	guidedimg, srcimg, radius=1, eps=0.002[, dst[, dDepth]]	)'''
    t2 = cv2.ximgproc.guidedFilter(rgb,t2,3,5)
    #(np.mean(minI)*np.std(minJ)/(np.std(minI)*np.mean(minJ)))

    return t2,A,J1
