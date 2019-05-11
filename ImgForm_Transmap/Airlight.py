import numpy as np
import cv2

def airLight(img):
    #img=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    l=(np.array(img).flatten().tolist())#flatten image to list
    l.sort(reverse=True)
    n=int(len(l)*0.001)#no.of pixels for 0.1% in image
    avg=0
    for i in range(0,n):
        avg+=l[i]
    return int(avg/n)
