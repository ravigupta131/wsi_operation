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
# import argparse

from multiprocessing import Process
import threading
import glob


# padd = 112
level = 0
openslide_formats = ['.ndpi', '.svs', '.tif', '.vms', '.vmu', '.scn', '.mrxs', '.tiff', 'svslide', 'bif']
#parser=argparse.ArgumentParser()
#parser.add_argument('-c','--case',help='cases')
# results=parser.parse_args()
# cases=results.case

def get_all_regions(file_name):
    mydoc = minidom.parse(file_name)
    annotations = mydoc.getElementsByTagName('Annotation')
    all_anns = []
    all_orgs = []
    all_paths = []
    name=[]
    for annotation in annotations:
        regions = annotation.getElementsByTagName('Region')
        name.append(annotation.getAttribute('Name').encode('utf-8'))
        all_regions = []
        orgs = []
        paths = []
        for region in regions:
            verticies = region.getElementsByTagName('Vertex')
            xy = []
            xy_path = []
            for item in verticies:
            	# print(item.getAttribute('X'))
            	# break
                xy.append(list(map(int,[float(item.getAttribute('X').encode('utf-8')),float(item.getAttribute('Y').encode('utf-8'))])))
                xy_path.append([item.getAttribute('X'),item.getAttribute('Y')])
            all_regions.append(xy)
            ox,oy,wd,ht = cv2.boundingRect(np.asarray(xy))
            orgs.append([ox,oy,wd,ht])
            paths.append(mplPath.Path(xy_path))
        all_anns.append(all_regions)
        all_orgs.append(orgs)
        all_paths.append(paths)
    return all_anns,all_orgs,all_paths,name




def get_annotations_1(format_f,xml_file_path,output_directory,padd,image_path,level=0):
    '''
    format_f = format of file
    padd = padding
    directory = ?
    file_name = name of the xml file
    path = ?
    lrp1 = ?
    
    '''
    ffl = len(format_f)
    ff = format_f
    padd = padd
    directory = output_directory #+ '/'
    all_anns,all_origins,all_paths,name = get_all_regions(xml_file_path)
    if(str(ff) in openslide_formats):
        img = openslide.open_slide(image_path)
    else:
        img = cv2.imread(image_path)

    count = 0
    pathsplt = image_path.split('/')
    #lwp1 = len(pathsplt)
    # subfold_count = lrp1 - lwp1
 #    #print(pathsplt)
 #    #print(subfold_count)
    # while(subfold_count < -1):
    #   if not os.path.exists(directory + pathsplt[subfold_count]):
    #       os.mkdir(directory + pathsplt[subfold_count])
    #   directory = directory + pathsplt[subfold_count] + '/'
 #        subfold_count = subfold_count + 1

    for lyr in range(len(all_anns)):
        all_regions = all_anns[lyr]
      #  if (name[lyr].decode("utf-8")!="IDC"):
      #      print("NO IDC")
      #      continue
        if not os.path.exists(output_directory + str(lyr)):
            os.makedirs(output_directory + str(lyr))
        if(len(all_regions) == 0):
            continue
        dir_path = directory + str(lyr) + '/'
        origins = all_origins[lyr]
        paths = all_paths[lyr]
        n_regions = len(all_regions)
        for region_id in range(n_regions):
            if(len(origins[region_id])==0):
                print("Y")
                continue
            ox,oy,w,h = origins[region_id]
            print("ox and oy: ",w,h)
            print("W and H: ",w,h)
            # print(type(w))
#             if (w>=40000 or h>=40000):
#                 continue
#             else:
            region = all_regions[region_id]
            # mpl_path = paths[tmp1]
            start_x = max(0 , ox - padd)
            start_y = max(0 , oy - padd)
            shifted_region = [[item[0] - start_x, item[1] - start_y] for item in region]
            mask_im = np.zeros((h + 2*padd, w + 2*padd, 3))
            mask_im = cv2.drawContours(mask_im, np.asarray([shifted_region]), -1, (255,255,255), -1, 8)
            mask_im = np.array(mask_im, dtype=np.uint8)
            mask_im = cv2.cvtColor( mask_im, cv2.COLOR_BGR2GRAY)
            img_save_name = dir_path + 'ROI__' + str(count) + '__' + pathsplt[-1][0:-ffl] + '__layer__' + str(lyr) + '__' + str(ox) + '__' + str(oy) + '__' + str(w) + '__' + str(h)
            if(ff in openslide_formats):
                im_object = img.read_region((start_x,start_y),level,(w + 2*padd, h + 2*padd))
                im_object.save(img_save_name + '__.png')
                w1t,h1t = im_object.size 
                # print(im_object.shape)
                # raise valueError()
            else:
                im_patch = img[start_y:start_y + h + 2*padd, start_x:start_x + w + 2*padd, :]
                cv2.imwrite(img_save_name + '__.png', im_patch)
                h1t, w1t, ch1t = im_patch.shape
            mask_im = mask_im[0:h1t, 0:w1t]
            cv2.imwrite(img_save_name + '__mask__.png', mask_im)
            count = count + 1
        return count


list_xml =  glob.glob("/home/ravi/Desktop/TCGA-BRCA_DA/*.xml")
for item_xml in list_xml:
    print(item_xml)
    image_path = item_xml.replace(".xml",".svs")
    print(image_path)
    # img_name=image_path.split('/')[-1].split('.')[0]
    # print(img_name)
    c = get_annotations_1('.svs',item_xml,'/home/ravi/Desktop/TCGA-BRCA_DA/extract/',0,image_path,level=0)
    print("number of ROI: ",c)