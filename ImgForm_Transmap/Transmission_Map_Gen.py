from FindTransmission import *
import os
import glob
path='GPR\hazyimage'
os.chdir('C:\Users\y15it823\PycharmProjects\GPR\hazyimage')
for file in glob.glob('*.jpg'):
    rgb = cv2.imread(file)
    img,a=Transmission_map(rgb)
    cv2.imshow(("xyz"+file),img)
cv2.waitKey(0)