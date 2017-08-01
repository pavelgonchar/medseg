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


if __name__ == "__main__":
	root_folder = '/home/zlp/dev/medseg'
	''' 
	infolder1: 2 classes for bg and liver
	infolder2: 2 classes for bg and liver
	'''
	infolder1 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_liver_iter_280000_LCC')
	infolder2 = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_multiloss_iter_200000_LCC')
	outfolder = os.path.join(root_folder, 'output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D/unet_2d_bn_c2_liver_200000LCC_280000LCC_CB')
	combine_liver_liver(infolder1, infolder2, outfolder)