import argparse
import os
import numpy as np
import time
import pandas as pd

import openslide
from tiatoolbox.wsicore import wsireader

from sub_functions import initialize_df
from sub_functions import create_mask
from sub_functions import create_random_patches
from sub_functions import colornorm_patch
from sub_functions import filter_by_nuclie_count

#---------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='preprocess (mask, patch, filter, colornorm) data')

parser.add_argument('--source_csv', type = str, default = 'test.csv',
					help='path to csv file containing image data')
parser.add_argument('--save_dir', type = str, default= 'results_folder',
					help='directory to save processed data')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--mask_scale', type=int, default=4, 
					help='scale factor for masking i.e mask size is scale*patch_size')
parser.add_argument('--patch_scale', type=int, default=2, 
					help='patch size for random patching is scale*patch_size')
parser.add_argument('--colornorm_scale', type=int, default=2, 
					help='patch size for colornorm is scale*patch_size')
parser.add_argument('--nuclie_scale', type=int, default=2, 
					help='patch size for nuclie filtering is scale*patch_size')

if __name__ == '__main__':
	args = parser.parse_args()

	# creating directories to save results
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	patch_save_dir = os.path.join(args.save_dir, 'sample_patches')
	nucliecount_save_dir = os.path.join(args.save_dir, 'nuclie_filtering')
	colornorm_save_dir = os.path.join(args.save_dir, 'patches')

	directories = {'save_dir': args.save_dir,
				'mask_save_dir' : mask_save_dir,
				'patch_save_dir': patch_save_dir,
                'nucliecount_save_dir':nucliecount_save_dir,
				'colornorm_save_dir':colornorm_save_dir}

	print('source_csv: ', args.source_csv)
	for key, val in directories.items():
		print(key,':', val)
		os.makedirs(val, exist_ok=True)

	# Taking csv input
	slides_info = pd.read_csv(args.source_csv)
	df = initialize_df(slides_info)
	mask = df['process'] == 1
	process_stack = df[mask]
	total = len(process_stack)

	for i in range(total):
		# try:
		# taking input
		df.to_csv(os.path.join(args.save_dir, 'processed_result.csv'), index=False)
		idx = process_stack.index[i]
		slide_id = process_stack.loc[idx, 'slide_id']
		case_id = process_stack.loc[idx, 'case_id']
		full_path = process_stack.loc[idx,'slide_path']

		# slide_id = 'TCGA-05-4382-01Z-00-DX1'
		# full_path = '/home/lung/ravi/preprocessing/TCGA-05-4382-01Z-00-DX1.76b49a4c-dbbb-48b0-b677-6d3037e5ce88.svs'

		print("\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide_id))

		# making slide-level diretories
		os.makedirs(os.path.join(patch_save_dir, slide_id), exist_ok=True)
		os.makedirs(os.path.join(patch_save_dir, slide_id, 'colornorm_patches'), exist_ok=True)

		# checking if color-norm is already done, if yes then skip the wsi
		if os.path.isfile(os.path.join(colornorm_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# opening wsi
		wsi_reader = wsireader.WSIReader.open(input_img= full_path)
		wsi = openslide.open_slide(full_path)
		wsi_dim = wsi.level_dimensions[0]

		# Mask
		time1 = time.time()
		label_bool = create_mask(wsi, slide_id, wsi_dim, args.patch_size, args.mask_scale, mask_save_dir, patch_save_dir)
		time2 = time.time()
		df.loc[idx, 'mask_status'] = 'done'
		df.loc[idx, 'mask_time'] = time2 - time1

		print('mask done ----------')

		# Patch
		patches, num_patches_after_mask = create_random_patches(wsi, slide_id, args.patch_size, args.patch_scale, args.mask_scale, patch_save_dir, label_bool)
		time3 = time.time()
		df.loc[idx, 'patch_status'] = 'done'
		df.loc[idx, 'num_patches_after_mask'] = num_patches_after_mask
		df.loc[idx, 'num_patches_after_patching'] = len(patches)
		df.loc[idx, 'patch_time'] = time3 - time2

		print('patch done ----------')

		# nuclie count
		filtered_patches = filter_by_nuclie_count(wsi_reader, case_id, slide_id, args.patch_size, args.nuclie_scale, nucliecount_save_dir, patches)
		time4 = time.time()
		df.loc[idx, 'nucliecount_status'] = 'done'
		df.loc[idx, 'num_patches_after_filtering'] = len(filtered_patches)
		df.loc[idx, 'nucliecount_time'] = time4 - time3

		print('nuclie filtering done ----------')

		# color normalisation
		colornorm_patch(wsi_reader, slide_id, args.patch_size, args.colornorm_scale, colornorm_save_dir, patch_save_dir, filtered_patches, wsi_dim)
		time5 = time.time()
		df.loc[idx, 'colornorm_status'] = 'done'
		df.loc[idx, 'colornorm_time'] = time5 - time4

		print('color normalization done ----------')

		df.loc[idx, 'process'] = 0
		df.loc[idx, 'total_time_taken'] = time5 - time1
		df.to_csv(os.path.join(args.save_dir, 'processed_result.csv'), index=False)

		# except:
		# 	continue


