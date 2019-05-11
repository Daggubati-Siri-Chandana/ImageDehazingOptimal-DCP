import cv2
import numpy as np
import math

def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref,dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255 / rmse)
