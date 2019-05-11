from ImgForm_Transmap.OptimizedTransmission import *
import cv2
from skimage.measure import compare_ssim as ssim
from psnr import *

rgb = cv2.imread("/Users/daggubatisirichandana/Desktop/Project/image.jpg")
[h, w, k] = rgb.shape
t2,A,J= optimizedTrans_map(rgb)
temp = np.zeros((h,w),dtype=np.float32)
J1 = np.zeros((h,w,k),dtype=np.uint8)

'''Dehazed image in iteration 1'''
for c in range(k):
    for i in range(h):
        for j in range(w):
            temp[i,j]=(((rgb[i,j][c] - A[c])/max(t2[i,j],0.1)))+A[c]
    '''This is to normalize the image matrix of iteration 1 from 0 - 255'''
    minp=np.amin(temp)
    maxp=np.amax(temp)
    for i in range(h):
        for j in range(w):
            J1[i, j][c] = int(((temp[i, j] - minp) * 255) / (maxp - minp))



cv2.imshow("input.jpg",rgb)
cv2.imshow("new output.jpg",J1)
cv2.imshow("old output.jpg",J)
ssim_new = ssim(rgb, J1, data_range=np.amax(J1) - np.amin(J1),multichannel=True)
ssim_old = ssim(rgb, J, data_range=np.amax(J) - np.amin(J),multichannel=True)
print("output image ssim = ",ssim_old)
print("optimised output image ssim = ",ssim_new)
print("output image psnr = ",psnr(rgb,J))
print("optimised output image psnr = ",psnr(rgb,J1))


cv2.waitKey(0)



