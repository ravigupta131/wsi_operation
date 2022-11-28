from sklearn.feature_extraction import image
import numpy as np
import cv2
import os
import glob
import sys


src='/home/ravi/Desktop/TCGA-BRCA_DA/extract/Non-mask/'
# print(src)
dest='/home/ravi/Desktop/TCGA-BRCA_DA/extract/patch_512/'
#dest='/home/Drive3/TCGA/Validation_March/HER2_target/pos/patch^es/'
mpath='/home/ravi/Desktop/TCGA-BRCA_DA/extract/mask'


fold=glob.glob(src+'/*')
print(f'fold is :{fold}')
window=512.0
win=512
threshold=210
slide=1
tot_count=window*window

for f1 in fold:
    
    print(f'f1 is :{f1}')
    fname=f1.split('/')[-1]
    print(f'fname is :{fname}')
    dest_fname=fname.split('.png')[0]
    print(f'dest_fname is :{dest_fname}')
    foldname=f1.split('/')[-1].split('__')[0]
    print(f'fold name is :{foldname}')
    if not os.path.exists(dest+foldname):
        os.makedirs(dest+foldname)
    os.chdir(dest+foldname)
    img=cv2.imread(f1)
    sr=mpath+foldname+'/'+fname
    print(sr)
    mask=cv2.imread(mpath+'/'+fname,0)
        
    row,col= img.shape[:2]
    print(row,col)
      
            
    for i in range(0,row,np.int(win/slide)):
        if (i+win>row):
            rlast_index=row+1
#             print(f'rlast_index:{rlast_index}')
            rfirst_index=row-win
#             print(f'rfirst_index:{rfirst_index}')
        else:
            rlast_index=i+win
#             print(f'else rlast_index:{rlast_index}')
            rfirst_index=i 
#             print(f'else rfirst_index:{rfirst_index}')
        for j in range(0,col,np.int(win/slide)):
            if (j+win>col):
                clast_index=col+1
#                 print(f'clast_index:{clast_index}')
                cfirst_index=col-win
#                 print(f'cfirst_index:{cfirst_index}')
            else:
                clast_index=j+win
#                 print(f'else clast_index:{clast_index}')
                cfirst_index=j
#                 print(f'else cfirst_index:{cfirst_index}')
            mask_patch=mask[rfirst_index:rlast_index,cfirst_index:clast_index]
            patch=img[rfirst_index:rlast_index,cfirst_index:clast_index]
            black_count=np.sum(mask_patch== 0)
            percent_black=black_count*100/tot_count
            if (percent_black<=40.0):
                channel_1=patch[:,:,0]>threshold
                channel_2=patch[:,:,1]>threshold
                channel_3=patch[:,:,2]>threshold
                vfunc=np.vectorize(np.logical_and)
                pixel_and=vfunc(vfunc(channel_1,channel_2),channel_3)
                pixel_and_count=np.count_nonzero(pixel_and)
                ratio_white_pixel=float(pixel_and_count*100/(window*window))
                print(dest_fname)
                if (ratio_white_pixel<40) and patch.shape==(512,512,3):
                    print(dest_fname+str(i)+'_'+str(j)+'.png')
                    cv2.imwrite(dest_fname+'_'+str(i)+'_'+str(j)+'.png',patch)
                
            
    