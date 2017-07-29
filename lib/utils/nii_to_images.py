#!/usr/bin/env python
# encoding: utf-8


import os
import errno
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os.path as osp
import nibabel as nib

class nifti_slice_extractor(object):
	"""docstring for nii_to_images"""
	def __init__(self, nifti_dir):
		super(nifti_slice_extractor, self).__init__()
		self._nifti_dir = nifti_dir
		assert osp.exists(nifti_dir), 'nifti_dir not exist'

		self._images_outdir = osp.join(self._nifti_dir, 'Images')
		self._gt_outdir = osp.join(self._nifti_dir, 'Segmentation')
		self._imagesets_dir = osp.join(self._nifti_dir, 'ImageSets')
		self._mkdir_p(self._images_outdir)
		self._mkdir_p(self._gt_outdir)
		self._mkdir_p(self._imagesets_dir)

		self._ext = '.jpg'
		# for visualize the label
		self._contrast_enhancing = 100

		self._create_image_file_list()
		self._create_gt_file_list()

	def _mkdir_p(self, path):
		try:
			os.makedirs(path)
		except OSError as exc:
			if exc.errno == errno.EEXIST and os.path.isdir(path):
				pass
			else:
				raise

	def _create_image_file_list(self):
		self._volume_list = [f for f in os.listdir(self._nifti_dir) if (f.startswith('volume-') and f.endswith('.nii'))]
		# sort based on index
		self._volume_list.sort(key=lambda name: int(name.split('-')[1].split('.')[0]))

	def _create_gt_file_list(self):
		self._segmentaion_list = [f for f in os.listdir(self._nifti_dir) if (f.startswith('segmentation-') and f.endswith('.nii'))]
		# sort based on index
		self._segmentaion_list.sort(key=lambda name: int(name.split('-')[1].split('.')[0]))

	def hounsfield_unit_value(self):
		# load nifti file
		index = 1
		max_value = 0
		min_value = 0
		for f in self._volume_list:
			im_path = osp.join(self._nifti_dir, f)
			gt_path = osp.join(self._nifti_dir, 'segmentation-'+f.split('-')[1])
			image_data = nib.load(im_path).get_data()
			image_data = image_data.astype(dtype=np.float32)
			gt_data = nib.load(gt_path).get_data()
			gt_data[gt_data < 1.5] = 0
			gt_data[gt_data > 1.5] = 1
			image_masked = np.multiply(image_data, gt_data)
			image_masked_max = np.max(image_masked)
			image_masked_min = np.min(image_masked)
			print image_masked_min, image_masked_max, im_path
			max_value = max_value if max_value > image_masked_max else image_masked_max
			min_value = min_value if min_value < image_masked_min else image_masked_min
			print min_value, max_value
		print '{} / {} : {}'.format(index, len(self._volume_list), f)
		index += 1
		print max_value, min_value

	def extract_image_slice_2_numpyarray(self):
		# load nifti file
		index = 1
		for f in self._volume_list:
			fpath = osp.join(self._nifti_dir, f)
			# create each nifti file output dir
			image_slice_dir = osp.join(self._images_outdir, osp.splitext(f)[0])
			self._mkdir_p(image_slice_dir)
			# read nifti
			image_data = nib.load(fpath).get_data()
			image_data = image_data.astype(dtype=np.int16)
			print image_data.shape, image_data.dtype
			for i in xrange(image_data.shape[2]):
				# extract a slice with index i
				image_slice = image_data[:,:,i]
				# plt.imshow(image_slice, cmap="gray",origin="lower")
				# plt.show()
				# save slice
				image_slice_path = osp.join(image_slice_dir, osp.split(image_slice_dir)[1] + '_slice_' + str(i) + '.npy')
				# print image_slice_path
				np.save(image_slice_path, image_slice)
				# image_slice = np.load(image_slice_path)
				# plt.imshow(image_slice, cmap="gray",origin="lower")
				# plt.show()
			print '{} / {} : {}'.format(index, len(self._volume_list), f)
			index += 1

	def extract_gt_slice_2_numpyarray(self):
		''' load nifti file '''
		# save image_slice_name, gt_slice_name, label1count, label2count in image_gt_match_list
		imageset_path = osp.join(self._imagesets_dir, 'image_gt_match_list.txt')
		fid = open(imageset_path, 'w')
		fid.write('# image_slice_name #' + ' ' + '# gt_slice_name #' + ' ' + '# label1count #' + ' ' + '# label2count #' + '\n')

		index = 1
		for f in self._segmentaion_list:
			fpath = osp.join(self._nifti_dir, f)
			# create output dir to save extracted slices for each nifti file
			gt_slice_dir = osp.join(self._gt_outdir, osp.splitext(f)[0])
			self._mkdir_p(gt_slice_dir)
			# read nifti
			gt_data = nib.load(fpath).get_data()
			gt_data = gt_data.astype(np.uint8)
			print gt_data.shape, gt_data.dtype
			for i in xrange(gt_data.shape[2]):
				# extract a slice with index i
				gt_slice = gt_data[:,:,i]
				# plt.imshow(gt_slice, cmap="gray",origin="lower")
				# plt.show()
				# save slice
				gt_slice_path = osp.join(gt_slice_dir, osp.split(gt_slice_dir)[1] + '_slice_' + str(i) + '.npy')
				# print gt_slice_path
				np.save(gt_slice_path, gt_slice)

				# corresponding image slice
				# check whether slice contains specific label value
				gt_slice_name = osp.split(gt_slice_path)[1]
				image_slice_name = 'volume-' + gt_slice_name.split('-')[1]
				label1count = len(gt_slice[gt_slice==1])
				label2count = len(gt_slice[gt_slice==2])
				# print gt_slice_name, image_slice_name, label1count, label2count
				
				fid.write('{} {} {} {}\n'.format(image_slice_name, gt_slice_name, label1count, label2count))
				# gt_slice = np.load(gt_slice_path)
				# plt.imshow(gt_slice, cmap="gray",origin="lower")
				# plt.show()
			print '{} / {} : {}'.format(index, len(self._segmentaion_list), f)
			index += 1
		fid.close()

	def extract_image_slice(self):
		# load nifti file
		index = 1
		for f in self._volume_list:
			fpath = osp.join(self._nifti_dir, f)
			# create each nifti file output dir
			image_slice_dir = osp.join(self._images_outdir, osp.splitext(f)[0])
			self._mkdir_p(image_slice_dir)
			# read nifti
			nifti = sitk.ReadImage(fpath)
			# Threshold the image, set below 0 as 0, above 255 as 255
			nifti = sitk.Threshold(nifti, 0, 9999, 0)
			nifti = sitk.Threshold(nifti, 0, 255, 255)
			nifti = sitk.Cast(nifti, sitk.sitkUInt8)
			for i in xrange(nifti.GetSize()[2]):
			# extract a slice with index i
				image_slice = sitk.Extract(nifti, [nifti.GetSize()[0], nifti.GetSize()[1], 0], [0, 0, i])
				# plt.imshow(sitk.GetArrayFromImage(image_slice), cmap="gray", origin="lower")
				# plt.show()
				# save extracted slice
				image_slice_path = osp.join(image_slice_dir, osp.split(image_slice_dir)[1] + '_slice_' + str(i) + self._ext)
				# horizontal flip back to normal view image FlipAxes = [False,True]
				image_slice = sitk.Flip(image_slice, [False,True])
				sitk.WriteImage(image_slice, image_slice_path)

			print '{} / {} : {}'.format(index, len(self._volume_list), f)
			index += 1

	def extract_gt_slice(self):
		''' load nifti file '''
		# save image_slice_name gt_slice_name label_number
		imageset_path = osp.join(self._imagesets_dir, 'image_gt_match_list.txt')
		fid = open(imageset_path, 'w')
		fid.write('# image_slice_name #' + ' ' + '# gt_slice_name #' + ' ' + '# label1count #' + ' ' + '# label2count #' + '\n')

		index = 1
		for f in self._segmentaion_list:
			fpath = osp.join(self._nifti_dir, f)
			# create output dir to save extracted slices for each nifti file
			gt_slice_dir = osp.join(self._gt_outdir, osp.splitext(f)[0])
			self._mkdir_p(gt_slice_dir)

			# nifti = sitk.Cast(sitk.ReadImage(fpath), sitk.sitkUInt8)
			nifti = sitk.ReadImage(fpath)
			# for visualize the label
			nifti = sitk.Multiply(nifti, self._contrast_enhancing)
			nifti = sitk.Cast(nifti, sitk.sitkUInt8)
			for i in xrange(nifti.GetSize()[2]):
				# Extract a slice with index i
				gt_slice = sitk.Extract(nifti, [nifti.GetSize()[0], nifti.GetSize()[1], 0], [0, 0, i])
				# plt.imshow(sitk.GetArrayFromImage(gt_slice), cmap="gray", origin="lower")
				# plt.show()
				# save extracted slice
				gt_slice_path = osp.join(gt_slice_dir, osp.split(gt_slice_dir)[1] + '_slice_' + str(i) + self._ext)
				# horizontal flip back to normal view image FlipAxes = [False,True]
				gt_slice = sitk.Flip(gt_slice, [False,True])
				# sitk.WriteImage(gt_slice, gt_slice_path)

				# corresponding image slice
				gt_slice_name = osp.split(gt_slice_path)[1]
				image_slice_name = 'volume-' + gt_slice_name.split('-')[1]

				# check whether slice contains specific label value
				gt_array = sitk.GetArrayFromImage(gt_slice)
				label1count = len(gt_array[gt_array==int(1*self._contrast_enhancing)])
				label2count = len(gt_array[gt_array==int(2*self._contrast_enhancing)])
				fid.write('{} {} {} {}\n'.format(image_slice_name, gt_slice_name, label1count, label2count))

			print '{} / {} : {}'.format(index, len(self._segmentaion_list), f)
			index += 1

		fid.close()

nifti_dir = '/home/zlp/data/lpzhang/LiverTumorSegmentation/Training_Batch'
extr = nifti_slice_extractor(nifti_dir)
# extr.extract_gt_slice()
#extr.extract_image_slice()
# extr.extract_image_slice_2_numpyarray()
extr.extract_gt_slice_2_numpyarray()
# extr.hounsfield_unit_value()