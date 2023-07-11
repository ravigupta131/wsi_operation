from pma_python import core
import numpy as np
from shapely.geometry import Polygon
from shapely import wkt
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from math import ceil
from glob import glob
from matplotlib import cm
from typing import Tuple, Optional, List
import json
from mycolorpy import colorlist as mcp
import argparse
import sys
import os
import pathlib
import os

clr = mcp.gen_color(cmap='Spectral_r',n=41)
if not os.path.isdir('./json_qupath'):
    os.mkdir('./json_qupath')

csv_path = sys.argv[1]
if os.path.isdir(csv_path):
    csv_path = glob(csv_path+'/*.csv')
elif os.path.isfile(csv_path):
    csv_path = [csv_path]
else:
    print('csv path does not exists')
    sys.exit()


def build_poly(tx: np.ndarray, ty: np.ndarray, bx: np.ndarray, by: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Counter clock-wise
    px = np.vstack((tx, bx, bx, tx)).T
    py = np.vstack((ty, ty, by, by)).T

    return px, py

hex_map={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,'b':11,'c':12,'d':13,'e':14,'f':15}

def save_annotation_qupath(
        tx: np.ndarray,
        ty: np.ndarray,
        bx: np.ndarray,
        by: np.ndarray,
        values: np.ndarray,
        outpath: str,
        clr: np.ndarray 
) -> None:
    """
    Parameters
    ----------
    tx: array_like
    ty: array_like
    bx: array_like
    by: array_like
    values: array_like
    values_name: array_like
    outpath: str
    cmap: Colormap
    """

    # Check dimensions
    if not all(tx.shape == np.array([ty.shape, bx.shape, by.shape, values.shape])):
        return

    # Build shape and simplify the shapes if True
    polys_x, polys_y = build_poly(tx=tx, ty=ty, bx=bx, by=by)

    # Extract outer shapes
    coords = {}
    colors = []
    clss = []
    for i in range(len(polys_x)):
        c = [16*hex_map[clr[i][1]]+hex_map[clr[i][2]],16*hex_map[clr[i][3]]+hex_map[clr[i][4]],16*hex_map[clr[i][5]]+hex_map[clr[i][6]]]
        color=np.array(c)
        colors.append(color)
        label = values[i]
        clss.append([label])
        coords['poly{}'.format(i)] = {
            "coords": np.vstack((polys_x[i], polys_y[i])).tolist(),
            "class": str(label),
            "color": c
        }

    with open(outpath, 'w') as outfile:
        json.dump(coords, outfile)

def assign_clr(i):
    return clr[i]

def main():
    for csv in csv_path:
        print('create json for',csv)
        df=pd.read_csv(csv,index_col=False)
        TX=np.array(df['dim1'])
        TY=np.array(df['dim2'])
        BX=TX+256
        BY=TY+256
        VALUES=np.array(df['attention'])
        df['bin_VAL'] = 40*df['attention']
        df['bin_VAL'] = df['bin_VAL'].apply(ceil)
        df['CLR'] = df['bin_VAL'].apply(assign_clr)
        json_path = './json_qupath/'+ csv.split('/')[-1][:-3]+'json'
        if os.path.isfile(json_path):
            print('json already exists')
            continue

        try:
            save_annotation_qupath(TX,TY,BX,BY,np.array(df['attention']),json_path,np.array(df['CLR']))
        except Exception as error:
            print("An error occurred:",error,"for file",csv)
            
        
    
if __name__ == "__main__":
    main()