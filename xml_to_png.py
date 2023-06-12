#This program generates the mask and bounding box extraction from WSI, given the annotations. 
from __future__ import print_function
import cv2
import numpy as np
import os
from xml.dom import minidom
import matplotlib.path as mplPath
import numpy as np
import openslide
import cv2
import time
import sys 
import glob



input_dir = sys.argv[1] 
image_dir = sys.argv[2]
output_dir = sys.argv[3] 

list_xml =  glob.glob(input_dir+"/*.xml")
formatt='.tif'

def get_all_regions(file_name):
    mydoc = minidom.parse(file_name)
    annotations = mydoc.getElementsByTagName('Annotation')
    #print("Length of annoations: ", len(annotations))
    all_anns = []
    all_orgs = []
    all_paths = []
    for annotation in annotations:
        regions = annotation.getElementsByTagName('Region')
    
        
    
    
        all_regions = []
        orgs = []
        paths = []
        for region in regions:
            verticies = region.getElementsByTagName('Vertex')
            xy = []
            for item in verticies:
                xy.append([float(item.getAttribute('X')),float(item.getAttribute('Y'))])
            yield xy
            




if not os.path.exists(output_dir):
       os.makedirs(output_dir)

for item_xml in list_xml:

    fold_name=item_xml.split('/')[-1].split('.xml')[0]
    image_path=image_dir+"/"+fold_name+formatt
    if not(os.path.exists(image_path)):
        print(f"{formatt} image for file {fold_name} does not exist.")

    
    img = cv2.imread(image_path)
    height,width=img.shape[0:2]
    
    mask=np.zeros((height,width,1),np.uint8)

    for all_anns in get_all_regions(item_xml):
    
        all_anns = np.array(all_anns).astype(np.int32)
        cv2.drawContours(mask,[all_anns],-1,255,-1,1)
    print(mask.shape)
    cv2.imwrite(output_dir+'/'+fold_name+'.png',mask)
    

    
