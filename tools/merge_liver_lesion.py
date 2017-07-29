import _init_paths
from utils.blob import load_data, save_data
import os
import numpy as np

def load_segmentations_test(folder, prefix, suffix):
	files = [f for f in os.listdir(folder) if (f.startswith(prefix) and f.endswith(suffix))]
	files.sort(key=lambda name: int(name.split('-')[2].split('.')[0]))
	return files



def process_prob_data(infolder1, infolder2, outfolder):
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

if __name__ == "__main__":
	root_folder = '/home/zlp/dev/medseg/output'
	''' 
	infolder1: 2 classes for bg and liver
	infolder2: 3 classes for bg, liver and lesion
	'''
	# infolder1 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_multiloss_iter_200000_LCC')
	# infolder2 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c3/lits_Test_Batch_trainval_3D/unet_2d_bn_c3_zoom_refine_iter_160000_LCC')
	# outfolder = os.path.join(root_folder, 'output/unet/unet_2d_bn_c3/lits_Test_Batch_trainval_3D/unet_2d_bn_c3_zoom_refine_iter_160000_LCC_merge')
	# infolder1 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_liver_iter_280000_LCC')
	# infolder2 = os.path.join(root_folder, 'output/unet/unet_2d_bn_weigted_c3/lits_Test_Batch_trainval_3D/unet_2d_bn_weigted_c3_zoom_iter_220000_LCC')
	# outfolder = os.path.join(root_folder, 'output/unet/unet_2d_bn_weigted_c3/lits_Test_Batch_trainval_3D/unet_2d_bn_weigted_c3_zoom_iter_220000_LCC_merge')
	infolder1 = os.path.join(root_folder, 'unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_liver_200000LCC_280000LCC_CB_LCC')
	infolder2 = os.path.join(root_folder, 'uvnet/uvnet_2d_bn_weigted_c3/lits_Test_Batch_trainval_3D/uvnet_2d_bn_weigted_1.1.7_iter_250000_LCC')
	outfolder = os.path.join(root_folder, 'uvnet/uvnet_2d_bn_weigted_c3/lits_Test_Batch_trainval_3D/uvnet_2d_bn_weigted_1.1.7_iter_250000_LCC_MG')
	process_prob_data(infolder1, infolder2, outfolder)