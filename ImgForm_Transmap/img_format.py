from ImgForm_Transmap.FindTransmission import *
import cv2

rgb = cv2.imread("field.jpg")
[h, w, k] = rgb.shape
t,A=Transmission_map(rgb)
t0=0.1

temp = np.zeros((h,w),dtype=np.float32)
color = np.zeros((h,w,k),dtype=np.uint8)
#I(pi)-A/max(t0,t(pi))+A
for c in range(k):
    for i in range(h):
        for j in range(w):
            temp[i,j]=(((rgb[i,j][c] - A)/max(t[i,j],t0)))+A

    minp=np.amin(temp)
    maxp=np.amax(temp)
    for i in range(h):
        for j in range(w):
            if((maxp-minp)!=0) :
                color[i, j][c] = int(((temp[i, j] - minp) * 255) / (maxp - minp))
            else :
                color[i, j][c] = int(((temp[i, j] - minp) * 255))

#print(np.array(t).flatten().tolist())
#print(np.amax(t))
#color_img = cv2.cvtColor(mincolor, cv2.COLOR_GRAY2RGB)
cv2.imshow("original",rgb)
cv2.imshow("dehazed",color)
cv2.imshow("transmission2",t)
cv2.waitKey(0)