#!/usr/bin/env python
# encoding: utf-8


import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from utils.blob import im_list_to_blob, hounsfield_unit_window, normalizer
from lits.config import cfg
import os.path as osp

def _get_image_blob(im):
	'''Converts an image into a network input
	Arguments:
		im (ndarray): a color image in BGR order

	Returns:
		blob (ndarray): a data blob holding an image pyramid
		im_scale_factors (list): list of image scales (relative to im) used
		in the image pyramid
	'''
	im_orig = im.astype(np.float32, copy=True)
	im_orig -= cfg.PIXEL_MEANS
	im_shape = im_orig.shape
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2])

	max_size = cfg.TEST.MAX_SIZE
	test_scales = cfg.TEST.SCALES

	processed_ims = []
	im_scale_factors = []
	for target_size in test_scales:
		im_scale = float(target_size)/float(im_size_min)
		# Prevent the biggest axis from being more than MAX_SIZE
		if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
			im_scale = float(max_size)/float(im_size_max)
		im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
		# append
		if len(im.shape) != 3:
			im = im[..., np.newaxis]
		processed_ims.append(im)
		im_scale_factors.append(im_scale)

	# Create a blob to hold the input images
	blob = im_list_to_blob(processed_ims, im_type='2D')

	return blob

def _get_image_blob_3d(im):
	'''Converts an 3d image into a network input
	Arguments:
		im (ndarray): A 3d numpy array

	Returns:
		blob (ndarray): a data blob holding an image pyramid
		im_scale_factors (list): list of image scales (relative to im) used
		in the image pyramid
	'''
	processed_ims = []
	processed_ims.append(im)
	blob = im_list_to_blob(processed_ims, im_type='3D')

	return blob

def _get_blobs(im):
	''' Get input blob
	'''
	blobs = {'data':None}
	if cfg.SEGMENTATION_MODE == '3D':	
		blobs['data'] = _get_image_blob_3d(im)
	else:
		blobs['data'] = _get_image_blob(im)

	return blobs

def evaluation():
	pass

def dice(im1, im2):
	'''Function to calculate dice for two images
	Input: two images with 0 or 1 elements only (binary)
	Output: dice
	'''
	union = np.sum(im1) + np.sum(im2)
	intersection = np.sum(im1 * im2)

	return 2*intersection/(union+cfg.EPS)

def vis_seg(axes, ims):
	'''Function to display row of images'''
	for i, im in enumerate(ims):
		row = i // axes.shape[1]
		col = i % axes.shape[1]
		axes[row, col].imshow(im, cmap='gray', origin='upper')
	plt.show()
	plt.pause(0.00001)

def test_net(net, imdb, output_dir):
	if cfg.SEGMENTATION_MODE == '3D':
		test_net_3d(net, imdb, output_dir)
	else:
		test_net_2d(net, imdb, output_dir)

def test_net_2d(net, imdb, output_dir):
	''''''
	plt.ion() # turn on interactive mode
	fig, axes = plt.subplots(3, 3)
	for i in xrange(len(imdb)):
		''' Load images in 3 channel
		and Load gts in 1 channel
		'''
		im_path = imdb[i]['image']
		gt_path = imdb[i]['gt']
		im_path = osp.join(cfg.DATA_DIR,'lits/Training_Batch','Images/volume-100/volume-100_slice_551.jpg')
		gt_path = osp.join(cfg.DATA_DIR,'lits/Training_Batch','Segmentation/segmentation-100/segmentation-100_slice_551.jpg')
		if not osp.exists(im_path):
			print 'Path does not exist: {}'.format(im_path)
			continue
		if not osp.exists(gt_path):
			print 'Path does not exist: {}'.format(gt_path)
			pass
		
		im = cv2.imread(im_path)
		gt = cv2.imread(gt_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		''' Transform im to float type
		'''
		im = im.astype(np.float32, copy=False)
		""" Apply Hounsfield Unit Window  
		The Hounsfield unit values will be windowed in the range
		HU_WINDOW to exclude irrelevant organs and objects.
		"""
		# HU_WINDOW = [-100, 400]
		# im = hounsfield_unit_window(im, HU_WINDOW)
		''' Map src_range value to dst_range value
		'''
		im = normalizer(im, src_range=[0., 255.], dst_range=[0., 1.])
		''' Prepare input blobs
		'''
		blobs = _get_blobs(im)
		input_data = blobs['data']
		# reshape network inputs and feed input to network data layer
		net.blobs['data'].reshape(input_data.shape[0],input_data.shape[1],input_data.shape[2],input_data.shape[3])
		net.blobs['data'].data[...] = blobs['data']
		# do forward
		blobs_out = net.forward()
		# print net.blobs.items()
		# Gets the network output score
		score = blobs_out['score']
		# Gets target map
		outmap = np.argmax(score[0], axis=0)
		#print outmap.shape, im.shape
		''' visulize outmap
		'''
		vis_seg(axes, [im, gt, score[0,0], score[0,1], score[0,2], outmap==0, outmap==1, outmap==2])
		# exit()

# def test_net_3darray(net, im, HU_WINDOW, ):
# 	""" Apply Hounsfield Unit Window
# 	The Hounsfield unit values will be windowed in the range
# 	HU_WINDOW to exclude irrelevant organs and objects.
# 	"""
# 	im = hounsfield_unit_window(im, HU_WINDOW)
# 	""" Map src_range value to dst_range value
# 	"""
# 	im = normalizer(im, src_range, dst_range)
# 	""" Due to the limited GPU memory,
# 	use overlapped sliding windows strategy to crop sub-volumes
# 	then used the average of the probability maps of these sub-volumes to get the whole volume prediction
# 	"""
# 	chunk_shape=[]
# 	stride=[]
# 	do_forward(net, im, chunk_shape, stride)


def do_forward(net, image, chunk_shape, stride):
	""" Due to the limited GPU memory,
	use overlapped sliding windows strategy to crop sub-volumes
	then used the average of the probability maps of these sub-volumes to get the whole volume prediction
	"""
	# count is for counting the times of each position has overlapped
	count_overlap = np.zeros(image.shape, dtype=np.float_)
	prediction_map = np.zeros(image.shape, dtype=np.float_)
	chunks_index = split2chunks(image.shape, chunk_shape, stride)
	for ind_chunk in chunks_index:
		im_chunk = image[ind_chunk[0][0]:ind_chunk[0][1], ind_chunk[1][0]:ind_chunk[1][1], ind_chunk[2][0]:ind_chunk[2][1]]
		blobs = _get_blobs(im_chunk)
		# reshape network inputs
		net.blobs['data'].reshape(*(blobs['data'].shape))
		net.blobs['data'].data[...] = blobs['data']
		# do forward
		blobs_out = net.forward()
		# get network output prob
		prob = blobs_out['u0d_score']
		# get the target map
		outmap = np.argmax(prob[0], axis=0)
		# stitch the chunk
		prediction_map[ind_chunk[0][0]:ind_chunk[0][1], ind_chunk[1][0]:ind_chunk[1][1], ind_chunk[2][0]:ind_chunk[2][1]] += outmap
		count_overlap[ind_chunk[0][0]:ind_chunk[0][1], ind_chunk[1][0]:ind_chunk[1][1], ind_chunk[2][0]:ind_chunk[2][1]] += 1
     
	# get the final results
	prediction_map = prediction_map / count_overlap

	return prediction_map


def test_net_3d(net, imdb, output_dir):
	plt.ion() # turn on interactive mode
	fig, axes = plt.subplots(2, 2)
	for i in xrange(len(imdb)):
		# imdb preprocessing to imput blob
		im_path = imdb[i]['image']
		gt_path = imdb[i]['gt']
		im_path = osp.join(cfg.DATA_DIR,'lits/Training_Batch_256_256','volume-0.nii')
		gt_path = osp.join(cfg.DATA_DIR,'lits/Training_Batch_256_256','segmentation-0.nii')
		print im_path, gt_path
		
		if not osp.exists(im_path):
			print 'Path does not exist: {}'.format(im_path)
			continue
		if not osp.exists(gt_path):
			print 'Path does not exist: {}'.format(gt_path)
			pass

		im = nib.load(im_path).get_data()
		gt = nib.load(gt_path).get_data()
		im = im.astype(np.float32, copy=False)
		""" Apply Hounsfield Unit Window  
		The Hounsfield unit values will be windowed in the range
		HU_WINDOW to exclude irrelevant organs and objects.
		"""
		HU_WINDOW = [-100, 400]
		im = hounsfield_unit_window(im, HU_WINDOW)
		''' Map src_range value to dst_range value
		'''
		src_range = HU_WINDOW
		dst_range = [0, 1]
		im = normalizer(im, src_range, dst_range)
		chunk_shape=[232, 232, 8]
		stride=[232, 232, 8]
		prediction_map = do_forward(net, im, chunk_shape, stride)
		for slices in xrange(prediction_map.shape[2]):
			vis_seg(axes, [im[:,:,slices], gt[:,:,slices], prediction_map[:,:,slices]])
		exit()

def split2chunks(image_shape, chunk_shape, stride):
	''' Extract chunks from image based on chunk_shape and stride,
	return a list of chunks' indexes
	Each stores a chunks' index refer to the original image position, in the form of:
	[[row_start,row_end], [colume_start, colume_end], [page_start, page_end]]
	
	Each chunk data could obtained from image data in a following way:
	chunk_data = image[row_start:row_end, colume_start:colume_end, page_start:page_end]
	'''
	chunks_index = []
	split_shape = np.int_(np.ceil((np.array(image_shape) - np.array(chunk_shape))/np.float_(np.array(stride)))) + 1
	for r in xrange(split_shape[0]):
		# chunk index of row
		r_s, r_e = r*stride[0], r*stride[0]+chunk_shape[0]
		if r_e > image_shape[0]:
			r_s, r_e = image_shape[0]-chunk_shape[0], image_shape[0]
		for c in xrange(split_shape[1]):
			# chunk index of column
			c_s, c_e = c*stride[1], c*stride[1]+chunk_shape[1]
			if c_e > image_shape[1]:
				c_s, c_e = image_shape[1]-chunk_shape[1], image_shape[1]
			for p in xrange(split_shape[2]):
				# chunk index of page
				p_s, p_e = p*stride[2], p*stride[2]+chunk_shape[2]
				if p_e > image_shape[2]:
					p_s, p_e = image_shape[2]-chunk_shape[2], image_shape[2]
				# store the chunk index
				chunk_index = [[r_s, r_e], [c_s, c_e], [p_s, p_e]]
				chunks_index.append(chunk_index)

	return chunks_index