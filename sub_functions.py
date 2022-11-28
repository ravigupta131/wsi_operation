'''
Functions for data-preprocessing
'''


import os
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import h5py

# histomicsTK for masking and color normalization
from histomicstk.saliency.tissue_detection import get_slide_thumbnail, get_tissue_mask
from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import deconvolution_based_normalization

# histocartography for nuclie filtering
from histocartography.preprocessing import NucleiExtractor, DeepFeatureExtractor, KNNGraphBuilder
from tiatoolbox.wsicore import wsireader

np.random.seed(0)

#---------------------------------------------------------------------------------------------

def initialize_df(slides):

	total = len(slides)

	# columns to copy from old csv to new
	slide_path = slides.slide_path.values
	slide_id = slides.slide_id.values
	case_id = slides.case_id.values
	dataset = slides.dataset.values
	objective_power = slides.objective_power.values
	slide_dimensions = slides.slide_dimensions.values
	level_count = slides.level_count.values
	level_dimensions = slides.level_dimensions.values
	vendor = slides.vendor.values
	label = slides.label.values

	default_df_dict = 	{'slide_path': slide_path,
						'case_id': case_id,
						'slide_id': slide_id,
						'label': label,
						'dataset': dataset,
						'objective_power': objective_power,
						'slide_dimensions': slide_dimensions,
						'level_count': level_count,
						'level_dimensions': level_dimensions,
						'level_downsamples': level_dimensions,
						'vendor':vendor}

	# adding following columns to input csv
	default_df_dict.update({
		'process': np.full((total), 1, dtype=np.uint8),
		'mask_status': np.full((total), 'tbp'),
		'patch_status': np.full((total), 'tbp'),
		'colornorm_status': np.full((total), 'tbp'),
		'nucliecount_status': np.full((total), 'tbp'),
		'num_patches_after_mask' : np.full((total), 0, dtype = np.int32),
		'num_patches_after_patching' : np.full((total), 0, dtype = np.int32),
		'num_patches_after_filtering' : np.full((total), 0, dtype = np.int32),
		'mask_time' : np.full((total), 0.0, dtype = np.float32),
		'patch_time' : np.full((total), 0.0, dtype = np.float32),
		'nucliecount_time' : np.full((total), 0.0, dtype = np.float32),
		'colornorm_time' : np.full((total), 0.0, dtype = np.float32),
		'total_time_taken' : np.full((total), 0.0, dtype = np.float32)
		})

	temp_copy = pd.DataFrame(default_df_dict) # temporary dataframe w/ default params
	
	# find key in provided df
	# if exist, fill empty fields w/ default values, else, insert the default values as a new column
	for key in default_df_dict.keys(): 
		if key in slides.columns:
			mask = slides[key].isna()
			slides.loc[mask, key] = temp_copy.loc[mask, key]
		else:
			slides.insert(len(slides.columns), key, default_df_dict[key])
	return slides

#---------------------------------------------------------------------------------------------

def initialise_h5(h5_file_path, colornorm_patch, patch_coord, wsi_dim, slide_id, patch_size):
	# opening file in write mode
	file = h5py.File(h5_file_path, "w")
	colornorm_patch = np.array(colornorm_patch)[np.newaxis,...]

	# coordinates dataset
	coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
	coord_dset[:] = patch_coord

	coord_dset.attrs['patch_level'] = 0
	coord_dset.attrs['patch_size'] = patch_size

	# patch dataset
	img_dtype = colornorm_patch.dtype
	img_shape = colornorm_patch.shape
	maxshape = (None,) + img_shape[1:]
	patch_dset = file.create_dataset('imgs', shape=img_shape, maxshape=maxshape, chunks=img_shape, dtype=img_dtype)
	patch_dset[:] = colornorm_patch

	patch_dset.attrs['patch_level'] = 0
	patch_dset.attrs['wsi_name'] = slide_id
	patch_dset.attrs['downsample'] = 1.0
	patch_dset.attrs['level_dim'] = wsi_dim
	patch_dset.attrs['downsampled_level_dim'] = wsi_dim

	file.close()
	return

#---------------------------------------------------------------------------------------------

def save_patch_h5(h5_file_path, colornorm_patch, patch_coord):
	# opening file in append mode
	file = h5py.File(h5_file_path, "a")
	colornorm_patch = np.array(colornorm_patch)[np.newaxis,...]

	# coordinates dataset
	coord_dset = file['coords']
	data_shape = (1, 2)
	coord_dset.resize(len(coord_dset) + data_shape[0], axis=0)
	coord_dset[-data_shape[0]:] = patch_coord

	# patch dataset
	patch_dset = file['imgs']
	data_shape = colornorm_patch.shape
	patch_dset.resize(file['imgs'].shape[0] + data_shape[0], axis=0)
	patch_dset[-data_shape[0]:] = colornorm_patch

	file.close()
	return

#---------------------------------------------------------------------------------------------

def create_mask(wsi, slide_id, wsi_dim, patch_size, mask_scale, mask_save_dir, patch_save_dir):

	deconvolve_first = True
	n_thresholding_steps = 1
	# mask dimentions are (scale/patch_size) times of original image. In our case, its 4/256 = 1/64
	mask_dim = (int(wsi_dim[0]*mask_scale/patch_size), int(wsi_dim[1]*mask_scale/patch_size))

	# getting the image thumbnail using openslides's get_thumbnail function
	wsi_thumbnail_rgb = wsi.get_thumbnail(mask_dim)	# returns a PIL image

	# converting Image object to numpy array
	wsi_thumbnail_array = np.array(wsi_thumbnail_rgb)	

	# getting tissue mask using histomicsTk library get_tissue_mask function
	labeled, mask = get_tissue_mask(wsi_thumbnail_array, deconvolve_first=deconvolve_first, n_thresholding_steps=n_thresholding_steps, sigma=0., min_size=30)

	# converting labeled matrix to 0-1 matrix
	label_bool = labeled
	label_bool[labeled!=0] = 1

	# generating, normalising and converting mask to image object
	wsi_thumbnail_array[label_bool==False] = 0*wsi_thumbnail_array[label_bool==False]
	# wsi_thumbnail_array = wsi_thumbnail_array/max(wsi_thumbnail_array)
	wsi_mask_rgb = Image.fromarray(wsi_thumbnail_array)

	# saving the mask and image thumbnail
	img_path = os.path.join(mask_save_dir, slide_id + '__img.png')
	mask_path = os.path.join(mask_save_dir, slide_id + '__mask.png')
	img_path2 = os.path.join(patch_save_dir, slide_id, slide_id + '__img.png')
	wsi_thumbnail_rgb.save(img_path)
	wsi_mask_rgb.save(mask_path)
	wsi_thumbnail_rgb.save(img_path2)

	return label_bool

#---------------------------------------------------------------------------------------------

def create_random_patches(wsi, slide_id, patch_size, patch_scale, mask_scale, patch_save_dir, label_bool):

	# parameters
	patch_size_on_mask = int(mask_scale*patch_scale)	# (mask_scale/patch_size)*(patch_scale*patch_size)
	patch_size_on_wsi = int(patch_scale*patch_size)	
	area_threshold = 0.9 	# 90%
	max_patches = 500		# max number of patches to store per wsi
	num_patches_png = 50	# number of patches png images to store as samples

	# looping over the mask to filter patches according to area threshold
	start_x = np.min(np.nonzero(label_bool)[0])
	start_y = np.min(np.nonzero(label_bool)[1])
	stop_x = np.max(np.nonzero(label_bool)[0])
	stop_y = np.max(np.nonzero(label_bool)[1])
	step_size_xy = patch_size_on_mask
	patches = []
	for y in range(start_y, stop_y, step_size_xy):
		for x in range(start_x, stop_x, step_size_xy):
			arr = label_bool[x:x+patch_size_on_mask, y:y+patch_size_on_mask]
			if np.sum(arr) > area_threshold*(patch_size_on_mask^2):
				# (x,y) coordinate on numpy array corresponds to (y,x) coordinate in PIL image
				wsi_patch_coord = (y*int(patch_size_on_wsi/patch_size_on_mask), x*int(patch_size_on_wsi/patch_size_on_mask))
				patches.append(wsi_patch_coord)

	# taking random at max max_patches number of patches randomly from patches array 
	total_patches = len(patches)
	print('total patches - ', total_patches)
	index = np.random.choice(total_patches, size=min(total_patches,max_patches), replace=False)
	random_patches = []
	for j in range(len(index)):
		random_patches.append(patches[index[j]])

	# saving png image of num_patches_png number of patches for visualization
	for i in range(min(len(random_patches), num_patches_png)):
		sample = wsi.read_region(random_patches[i], 0, (patch_size_on_wsi,patch_size_on_wsi))
		png_file_name = os.path.join(patch_save_dir, slide_id, slide_id + '__' + str(random_patches[i][0]) + '_' + str(random_patches[i][1]) + '.png')
		sample.save(png_file_name)

	return random_patches, total_patches

#---------------------------------------------------------------------------------------------

def filter_by_nuclie_count(wsi_reader, case_id, slide_id, patch_size, nuclie_scale, nucliecount_save_dir, patches):

	# things to save in csv file
	record = {'case_id':[], 'slide_id':[], 'patch_id':[], 'patch_coord_x': [], 'patch_coord_y': [], 'num_cells':[],'time':[],'valid_patch':[]}

	# filter limits - 2 options - absolute or percentile. choosing absolute
	# upper_limit = 1000
	lower_limit = 50

	# initialising nuclie detector
	nuclei_detector = NucleiExtractor()
	feature_extractor = DeepFeatureExtractor(architecture='resnet34', patch_size=72)

	# filtering patches
	filtered_patches = []
	for i in range(len(patches)):
		print('nuclie filtering patch - ', i+1 , '/', len(patches))
		t_start = time.time()
		sample = wsi_reader.read_region(patches[i], 0, (nuclie_scale*patch_size,nuclie_scale*patch_size))
		_, cell = nuclei_detector.process(sample)
		t_stop = time.time()

		# saving in dictionary
		record['case_id'].append(case_id)
		record['slide_id'].append(slide_id)
		record['patch_id'].append(i)
		record['patch_coord_x'].append(patches[i][0])
		record['patch_coord_y'].append(patches[i][1])
		record['num_cells'].append(len(cell))
		record['time'].append(t_stop-t_start)

		if len(cell) >= lower_limit:
			record['valid_patch'].append(1)
			filtered_patches.append(patches[i])
		else:
			record['valid_patch'].append(0)

	df = pd.DataFrame(record)
	df.to_csv(os.path.join(nucliecount_save_dir, slide_id +'__filtered.csv'),index=False)

	return filtered_patches

#---------------------------------------------------------------------------------------------

def colornorm_patch(wsi_reader, slide_id, patch_size, colornorm_scale, colornorm_save_dir, patch_save_dir, patches, wsi_dim):
	# using histomicTK for color norm with W_target as given in its documentation
	'''
	TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
	for macenco (obtained using rgb_separate_stains_macenko_pca() and reordered such that columns are the order: Hamtoxylin, Eosin, Null
	'''
	W_target = np.array([[0.5807549,  0.08314027,  0.08213795],[0.71681094,  0.90081588,  0.41999816],[0.38588316,  0.42616716, -0.90380025]])

	h5_file_path = os.path.join(colornorm_save_dir, slide_id + '.h5')
	if(len(patches)>0):
		# saving 1st patch to initialise dataset
		print('color normalization patch - ', 1 , '/', len(patches))
	
		sample = wsi_reader.read_region(patches[0], 0, (colornorm_scale*patch_size,colornorm_scale*patch_size))
		png_file_name = os.path.join(patch_save_dir, slide_id, 'colornorm_patches', slide_id + '__' + str(patches[0][0]) + '_' + str(patches[0][1]) + '.png')
		sample_image = Image.fromarray(sample)
		sample_image.save(png_file_name)

		tissue_rgb_normalized = deconvolution_based_normalization(sample, W_target=W_target)
		png_file_name = os.path.join(patch_save_dir, slide_id, 'colornorm_patches', slide_id + '__' + str(patches[0][0]) + '_' + str(patches[0][1]) + '_cn.png')
		colornorm_image = Image.fromarray(tissue_rgb_normalized)
		colornorm_image.save(png_file_name)

		#dividing patches in parts 
		for iy in range(colornorm_scale):
			for ix in range(colornorm_scale):

				# bounding box coordinates of patch
				left = patch_size*ix
				upper = patch_size*iy
				right = left + patch_size
				lower = upper + patch_size

				# data to save in h5 file
				sub_patch_coord = (patches[0][0] + patch_size*ix, patches[0][1] + patch_size*iy)
				sub_patch_img = colornorm_image.crop(box = (left, upper, right, lower))

				# save sub patches of 1 image
				png_file_name = os.path.join(patch_save_dir, slide_id, 'colornorm_patches', slide_id + '__' + str(ix) + str(iy) + '_' + str(sub_patch_coord[0]) + '_' + str(sub_patch_coord[1]) + '_cn.png')
				sub_patch_img.save(png_file_name)

				# initialise h5 file for the first patch
				if iy == 0 and ix == 0:
					initialise_h5(h5_file_path, sub_patch_img, sub_patch_coord, wsi_dim, slide_id, patch_size)
				else:
					save_patch_h5(h5_file_path, sub_patch_img, sub_patch_coord)

		# looping over patches to color normalise them
		for i in range(1,len(patches)):
			print('color normalization patch - ', i+1 , '/', len(patches))

			# take sample, color normalise and convert to image object
			sample = wsi_reader.read_region(patches[i], 0, (colornorm_scale*patch_size,colornorm_scale*patch_size))
			sample_image = Image.fromarray(sample)
			# png_file_name = os.path.join(colornorm_save_dir, slide_id, str(i) + '_' + slide_id + '__' + str(patches[i][0]) + '_' + str(patches[i][1]) + '.png')
			# sample_image.save(png_file_name)

			tissue_rgb_normalized = deconvolution_based_normalization(sample, W_target=W_target)
			colornorm_image = Image.fromarray(tissue_rgb_normalized)
			# png_file_name = os.path.join(colornorm_save_dir, slide_id, str(i) + '_' + slide_id + '__' + str(patches[i][0]) + '_' + str(patches[i][1]) + 'cn.png')
			# colornorm_image.save(png_file_name)

			for iy in range(colornorm_scale):
				for ix in range(colornorm_scale):
					# bounding box coordinates of patch
					left = patch_size*ix
					upper = patch_size*iy
					right = left + patch_size
					lower = upper + patch_size
					# data to save in h5 file
					sub_patch_coord = (patches[0][0] + patch_size*ix, patches[0][1] + patch_size*iy)
					sub_patch_img = colornorm_image.crop(box = (left, upper, right, lower))
					save_patch_h5(h5_file_path, sub_patch_img, sub_patch_coord)
	return 0

#---------------------------------------------------------------------------------------------