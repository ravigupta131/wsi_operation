import pandas as pd 
import openslide as op
import matplotlib.pyplot as plt
import numpy as np
import glob
import os



#csv_path='/workspace/clam1/CLAM/csv_tmc_pma_annot/CAIB-T00001384OP01B01P0101HE.csv'
annot_csv_files=glob.glob('/workspace/clam1/CLAM/csv_tmc_pma_annot/*', recursive = True)
#annot_csv_files=glob.glob('/workspace/clam1/CLAM/rem_csv/*', recursive=True)
#print(annot_csv_files)

for i in range(len(annot_csv_files)):
    csv_path=annot_csv_files[i]
    p=annot_csv_files[i].split('/')[-1].split('.')[0]+'.svs'
    svs_file_path='/workspace/hpv_tmh/'+p
    save_path=csv_path.split('/')[-1].split('.')[0]+'.csv'
    save_path1= '/workspace/clam1/CLAM/csv_tmc_w_filt_pma_annot/'+save_path
    if os.path.exists(save_path1):
        print('done')
    else:
        print(f'{svs_file_path} is in process')
        #if svs_file_path !='/workspace/hpv_tmh/CAIB-T00001462OP01B01P0101HE.svs':

        wsi= op.OpenSlide(svs_file_path)
        df=pd.read_csv(csv_path)
        data_record = {'dim1':[],'dim2':[]}
        for i in range(len(df)):
            x=df['dim1'][i]
            y=df['dim2'][i]
            #print(f'x is {x} and y is {y}')
            level_zero_img= wsi.read_region((x,y), 0, (256,256)) 
            #converting in rgb
            level_zero_img_rgb=level_zero_img.convert('RGB')
            level_zero_img_np = np.array(level_zero_img_rgb)
            print(level_zero_img_np.mean())
            if level_zero_img_np.mean()<220 and level_zero_img_np.mean()> 10 and  level_zero_img_np.std()>10:
                data_record['dim1'].append(x)
                data_record['dim2'].append(y)
            else:
                print('coords are dropped')
        df_svs = pd.DataFrame(data_record)
        
        #print(save_path1)
        df_svs.to_csv(save_path1,index=False)




'''
svs_file_path ='/workspace/hpv_tmh/CAIB-T00001462OP01B01P0101HE.svs'
csv_path='/workspace/clam1/CLAM/csv_tmc_pma_annot/CAIB-T00001462OP01B01P0101HE.csv'            
wsi= op.OpenSlide(svs_file_path)
df=pd.read_csv(csv_path)
data_record = {'dim1':[],'dim2':[]}
for i in range(len(df)):
    x=df['dim1'][i]
    y=df['dim2'][i]
                    
    level_zero_img= wsi.read_region((x,y), 0, (256,256)) 
                    
    level_zero_img_rgb=level_zero_img.convert('RGB')
    level_zero_img_np = np.array(level_zero_img_rgb)
    print(level_zero_img_np.mean())
    if level_zero_img_np.mean()<220 and level_zero_img_np.mean()> 10 and level_zero_img_np.std()>10:
        data_record['dim1'].append(x)
        data_record['dim2'].append(y)
    else:
        print('coords are dropped')
df_svs = pd.DataFrame(data_record)
        
df_svs.to_csv('gjhggd.csv',index=False)
    
'''
