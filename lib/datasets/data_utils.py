#!/usr/bin/env python
# encoding: utf-8

import cv2
from datasets.factory import get_imdb
from utils.blob import load_data, hounsfield_unit_window
import numpy as np

def prepare_imdb(images_list, gts_list):
	""" A imdb is a list of dictionaries, each with the following keys: image_path, gt_path, flipped
	"""
	assert len(images_list) == len(gts_list), 'number of images must equal to number of gts'
	imdb = []
	for i in xrange(len(images_list)):
		tmp = dict()
		tmp['image'] = images_list[i]
		tmp['gt'] = gts_list[i]
		imdb.append(tmp)

	return imdb

def prepare_training_imdb(images_list, gts_list):
	""" A imdb is a list of dictionaries, each with the following keys: image_path, gt_path, flipped
	"""
	assert len(images_list) == len(gts_list), 'number of images must equal to number of gts'
	imdb = []
	for i in xrange(len(images_list)):
		tmp = dict()
		tmp['image'] = images_list[i]
		tmp['gt'] = gts_list[i]
		# tmp['flipped'] = False
		imdb.append(tmp)
	# # Add flipped image
	# if cfg.TRAIN.USE_FLIPPED:
	# 	for i in xrange(len(images_list)): 
	# 		tmp = dict()
	# 		tmp['image'] = images_list[i]
	# 		tmp['gt'] = gts_list[i]
	# 		tmp['flipped'] = True
	# 		imdb.append(tmp)

	return imdb

def compute_image_statistics(image_path_list):
	""" Input : A list of image path
	Output: mean_of_images and number_of_images
	"""
	mean_of_images = 0.0
	var_of_images = 0.0
	number_of_images = 0
	for im_path in image_path_list:
		im_metadata = load_data(im_path)
		im = im_metadata['image_data']
		im = hounsfield_unit_window(im, hu_window=[-200,300]) #cfg.TRAIN.HU_WINDOW = [-200, 300] -95.3622154758
		number_of_images += 1
		mean_of_images = mean_of_images + (np.mean(im) - mean_of_images) / number_of_images
		var_of_images = var_of_images + (np.var(im) - var_of_images) / number_of_images
	std_of_images = np.sqrt(var_of_images)
	print 'mean:{}, std:{}'.format(mean_of_images, std_of_images)
	return mean_of_images, std_of_images

def combined_imdb(imdb_names):
	images = []
	gts = []
	for imdb_name in imdb_names.split('+'):
		imdb = get_imdb(imdb_name)
		images.extend(imdb.images_path())
		gts.extend(imdb.gts_path())
		print('Loaded dataset: `{:s}`'.format(imdb_name))

	calculate_pixel_means = 0
	if calculate_pixel_means:
		print('Calculating pixel statistics')
		mean_of_images, std_of_images = compute_image_statistics(images)
	return images, gts


def training_imdb(imdb_names):
	images, gts = combined_imdb(imdb_names)
	imdb = prepare_imdb(images, gts)
	return imdb

def test_imdb(imdb_names):
	images, gts = combined_imdb(imdb_names)
	imdb = prepare_imdb(images, gts)
	return imdb
