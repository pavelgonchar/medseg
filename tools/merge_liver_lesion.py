import _init_paths
from utils.blob import load_data, save_data
import os
import numpy as np

def load_segmentations_test(folder, prefix, suffix):
	files = [f for f in os.listdir(folder) if (f.startswith(prefix) and f.endswith(suffix))]
	files.sort(key=lambda name: int(name.split('-')[2].split('.')[0]))
	return files

def combine_liver_liver(infolder1, infolder2, outfolder):
	livers1 = load_segmentations_test(infolder1, prefix='test-segmentation-', suffix='.nii')
	livers2 = load_segmentations_test(infolder2, prefix='test-segmentation-', suffix='.nii')
	#print livers1, livers2
	assert len(livers1)==len(livers2), 'liver1 number must equal to livers2 number'
	print 'Total number of livers1: {}\n'.format(len(livers1))
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)
	
	for ind in range(len(livers1)):
		liver1_f = livers1[ind]
		liver2_f = livers2[ind]
		liver1_index = os.path.splitext(liver1_f)[0].split('-')[2]
		liver2_index = os.path.splitext(liver2_f)[0].split('-')[2]
		assert liver1_index == liver2_index, 'index mismatch'
		liver1_path = os.path.join(infolder1, liver1_f)
		liver2_path = os.path.join(infolder2, liver2_f)
		print 'liver1_path: {}'.format(liver1_path)
		print 'liver2_path: {}'.format(liver2_path)
		# load Image
		liver1_metadata = load_data(liver1_path)
		liver2_metadata = load_data(liver2_path)
		assert liver1_metadata is not None, 'liver1 open failed'
		assert liver2_metadata is not None, 'liver2 open failed'
		liver1_data = liver1_metadata['image_data']
		liver2_data = liver2_metadata['image_data']
		###
		# keep both liver1_data's liver label and liver2_data's liver label
		###
		print liver1_data.dtype, liver2_data.dtype
		print np.sum(liver1_data == 0), np.sum(liver1_data == 1), np.sum(liver1_data == 2)
		print np.sum(liver2_data ==0), np.sum(liver2_data == 1), np.sum(liver2_data==2)
		liver1_data += liver2_data
		liver1_data[liver1_data > 0] = 1
		print np.sum(liver1_data==0), np.sum(liver1_data == 1), np.sum(liver1_data==2)
		assert np.sum(liver1_data == 0)==np.sum(liver1_data < 1), 'liver1_data == 0, error'
		assert np.sum(liver1_data == 1)==np.sum((liver1_data > 0)&(liver1_data < 2)), 'liver1_data == 1, error'
		assert np.sum(liver1_data == 2)==np.sum(liver1_data > 1), 'liver1_data == 2, error'
		print liver1_data.dtype
		### save merge results
		outpath = os.path.join(outfolder, liver1_f)
		print 'Output file will save to: {}\n'.format(outpath)
		save_data(liver1_data, outpath)
	print '=== DONE ==='


def merge_liver_lesion(infolder1, infolder2, outfolder):
	livers = load_segmentations_test(infolder1, prefix='test-segmentation-', suffix='.nii')
	lesions  = load_segmentations_test(infolder2, prefix='test-segmentation-', suffix='.nii')
	#print livers, lesions
	assert len(livers)==len(lesions), 'liver number must equal to lesions number'
	print 'Total number of livers: {}\n'.format(len(livers))
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)
	
	for ind in range(len(livers)):
		liver_f = livers[ind]
		lesion_f = lesions[ind]
		liver_index = os.path.splitext(liver_f)[0].split('-')[2]
		lesion_index = os.path.splitext(lesion_f)[0].split('-')[2]
		assert liver_index == lesion_index, 'index mismatch'
		liver_path = os.path.join(infolder1, liver_f)
		lesion_path = os.path.join(infolder2, lesion_f)
		print 'liver_path: {}'.format(liver_path)
		print 'lesion_path: {}'.format(lesion_path)
		# load Image
		liver_metadata = load_data(liver_path)
		lesion_metadata = load_data(lesion_path)
		assert liver_metadata is not None, 'liver open failed'
		assert lesion_metadata is not None, 'lesion open failed'
		liver_data = liver_metadata['image_data']
		lesion_data = lesion_metadata['image_data']
		###
		# keep liver_data's liver label and lesion_data's lesion label
		###
		print liver_data.dtype, lesion_data.dtype
		# print np.sum(liver_data == 0), np.sum(liver_data == 1), np.sum(liver_data == 2)
		# print np.sum(lesion_data ==0), np.sum(lesion_data == 1), np.sum(lesion_data==2)
		# get lesion only from lesion_data, set others to zeros
		lesion_data[lesion_data < 2] = 0
		# move lesion to liver_data, lesion will overrie
		liver_data += lesion_data
		liver_data[liver_data > 1] = 2
		# print np.sum(liver_data==0), np.sum(liver_data == 1), np.sum(liver_data==2)
		print liver_data.dtype
		### save merge results
		outpath = os.path.join(outfolder, liver_f)
		print 'Output file will save to: {}\n'.format(outpath)
		save_data(liver_data, outpath)
		#print 'Output saved to: {}\n'.format(foutpath)
	print '=== DONE ==='

def model_combination(infolders, outfolder):
	models = []
	for infolder in infolders:
		# load each models' images
		model = load_segmentations_test(infolder, prefix='test-segmentation-', suffix='.nii')
		if models:
			assert len(models[len(models)-1])==len(model), 'model mismatch'
		print 'model has {} images\n'.format(len(model))
		models.append(model)
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)

	# iterate images
	for ind_image in xrange(len(models[0])):
		multi_images_d = []
		multi_images_f = []
		# iterate load models
		for ind_model in xrange(len(models)):
			model = models[ind_model]
			image_f = model[ind_image]
			# image_path
			image_path = os.path.join(infolders[ind_model], image_f)
			print 'Model {} image_path: {}'.format(ind_model, image_path)
			# load image data
			image_metadata = load_data(image_path)
			assert image_metadata is not None, 'image open failed'
			image_data = image_metadata['image_data']
			multi_images_d.append(image_data)
			multi_images_f.append(image_f)

		# iterate check models' images
		for ind_f in xrange(1, len(multi_images_f)):
			assert multi_images_f[0] == multi_images_f[ind_f], 'index mismatch'
			assert multi_images_d[0].shape == multi_images_d[ind_f].shape, 'image shape mismatch'

		### iterate combine models
		out_image = np.zeros(multi_images_d[0].shape, dtype=multi_images_d[0].dtype)
		for ind_model in xrange(len(multi_images_d)):
			image_data = multi_images_d[ind_model]
			print image_data.dtype, np.sum(image_data == 0), np.sum(image_data == 1), np.sum(image_data == 2)
			image_data[image_data>1] = 10
			print image_data.dtype, np.sum(image_data == 0), np.sum(image_data == 1), np.sum(image_data == 10)
			out_image += image_data

		### process the label
		out_image[out_image<1] = 0
		out_image[(out_image>=1)&(out_image<10)] = 1
		out_image[out_image>=10] = 2
		print out_image.dtype, np.sum(out_image == 0), np.sum(out_image == 1), np.sum(out_image == 2)
		### save merge results
		outpath = os.path.join(outfolder, multi_images_f[0])
		print 'Output file will save to: {}\n'.format(outpath)
		save_data(out_image, outpath)
	print '=== DONE ==='

if __name__ == "__main__":
	# task_type = 'combine_liver_liver'
	task_type = 'merge_liver_lesion'
	# task_type = 'model_combination'
	assert task_type in ('combine_liver_liver', 'merge_liver_lesion', 'model_combination'), 'task type error'
	print task_type
	if task_type == 'combine_liver_liver':
		''' 
		infolder1: 2 classes for bg and liver
		infolder2: 2 classes for bg and liver
		'''
		root_folder = '/home/zlp/dev/medseg'
		infolder1 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_liver_iter_280000_LCC')
		infolder2 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_multiloss_iter_200000_LCC')
		outfolder = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_liver_200000LCC_280000LCC_CB')
		combine_liver_liver(infolder1, infolder2, outfolder)
		
	elif task_type == 'merge_liver_lesion':
		''' 
		infolder1: 2 classes for bg and liver
		infolder2: 3 classes for bg, liver and lesion
		'''
		root_folder = '/home/zlp/dev/medseg'
		# infolder1 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_multiloss_iter_200000_LCC')
		# infolder2 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c3/lits_Test_Batch_trainval_3D/unet_2d_bn_c3_zoom_refine_iter_160000_LCC')
		# outfolder = os.path.join(root_folder, 'output/unet/unet_2d_bn_c3/lits_Test_Batch_trainval_3D/unet_2d_bn_c3_zoom_refine_iter_160000_LCC_merge')
		# infolder1 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_liver_iter_280000_LCC')
		# infolder2 = os.path.join(root_folder, 'output/unet/unet_2d_bn_weigted_c3/lits_Test_Batch_trainval_3D/unet_2d_bn_weigted_c3_zoom_iter_220000_LCC')
		# outfolder = os.path.join(root_folder, 'output/unet/unet_2d_bn_weigted_c3/lits_Test_Batch_trainval_3D/unet_2d_bn_weigted_c3_zoom_iter_220000_LCC_merge')
		infolder1 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_liver_iter_280000_LCC')
		infolder2 = os.path.join(root_folder, 'output/uvnet/uvnet_2d_bn_modified_weigted_c3/lits_Test_Batch_trainval_3D/lesion_250000_LCC_280000LCC_300000LCC_MC_LCC')
		outfolder = os.path.join(root_folder, 'output/uvnet/uvnet_2d_bn_modified_weigted_c3/lits_Test_Batch_trainval_3D/liver_280000LCC_lesion_250000_LCC_280000LCC_300000LCC_MC_LCC_MG')
		merge_liver_lesion(infolder1, infolder2, outfolder)

	elif task_type == 'model_combination':
		"""infolder list"""
		root_folder = '/home/zlp/dev/medseg/output'
		files = ['uvnet/uvnet_2d_bn_modified_weigted_c3/lits_Test_Batch_trainval_3D/uvnet_2d_bn_modified_weigted_c3_1.1.7_refined_iter_280000_LCC', 
				 'uvnet/uvnet_2d_bn_modified_weigted_c3/lits_Test_Batch_trainval_3D/uvnet_2d_bn_modified_weigted_c3_1.1.7_refined_iter_300000_LCC',
				 'uvnet/uvnet_2d_bn_weigted_c3/lits_Test_Batch_trainval_3D/uvnet_2d_bn_weigted_1.1.7_iter_250000_LCC']
		outfolder = os.path.join(root_folder,'uvnet/uvnet_2d_bn_modified_weigted_c3/lits_Test_Batch_trainval_3D/lesion_250000_LCC_280000LCC_300000LCC_MC')
		infolders =[os.path.join(root_folder,f) for f in files]
		model_combination(infolders, outfolder)
	else:
		print 'error'
	
	