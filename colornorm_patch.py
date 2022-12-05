import openslide as op
import glob
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import deconvolution_based_normalization
cn_img_dir='/home1/ravi/ravi_data/patches_a6s9_200_32f_cn/'
W_target = np.array([[0.5807549,  0.08314027,  0.08213795],[0.71681094,  0.90081588,  0.41999816],[0.38588316,  0.42616716, -0.90380025]])
image_list= glob.glob('/home1/ravi/ravi_data/patches_a6s9_200_32f/*.png')

for i in range(len(image_list)):
    image_path= image_list[i]
    
    img = Image.open(image_path)
    img_rgb=img.convert('RGB')
    img_np=np.array(img_rgb)
    img_cn = deconvolution_based_normalization(img_np, W_target=W_target)
    patch_name=image_path.split('/')[-1].split('.')[0]
    name= os.path.join(cn_img_dir,patch_name)
    plt.imsave(name+'_cn.png',img_cn)
    
print(' color normalization is done') 