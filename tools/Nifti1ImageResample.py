import numpy as np
import os
import os.path as osp
import scipy.ndimage
import nibabel as nib
'''
Resampling
A common method of dealing with this is resampling the full dataset to
a certain isotropic resolution. If we choose to resample everything to 
1mm1mm1mm pixels we can use 3D convnets without worrying about learning 
zoom/slice thickness invariance.
'''
def resample(src_image, src_spacing, dst_spacing):
	src_spacing = np.array(src_spacing, dtype=np.float32)
	dst_spacing = np.array(dst_spacing, dtype=np.float32)
	src_image = src_image.astype(np.float32, copy=False)
	# Determine current pixel spacing
	resize_factor = src_spacing / dst_spacing
	new_real_shape = src_image.shape * resize_factor
	new_shape = np.round(new_real_shape)
	# new_shape[0] = src_image.shape[0]
	# new_shape[1] = src_image.shape[1]
	# new_shape = np.array([256., 256., 64.])
	# new_shape[2] = src_image.shape[2]
	real_resize_factor = new_shape / src_image.shape
	dst_spacing = src_spacing / real_resize_factor
	# print new_shape, dst_spacing
	# exit()

	dst_image = scipy.ndimage.interpolation.zoom(src_image, real_resize_factor, mode='constant')

	return dst_image, dst_spacing

def nifti1_image_resample(src_path, dst_path, dst_spacing):
	src_img = nib.load(src_path)
	src_spacing = src_img.header.get_zooms()
	src_data = src_img.get_data()

	dst_data, new_spacing = resample(src_data, src_spacing, dst_spacing)
	
	new_spacing =  np.append(new_spacing, 1)
	dst_img = nib.Nifti1Image(dst_data, affine=np.diag(new_spacing))

	# print dst_img

	nib.save(dst_img, dst_path)

def nifti1_label_resample(src_path, dst_path, dst_spacing):
	src_img = nib.load(src_path)
	src_spacing = src_img.header.get_zooms()
	src_data = src_img.get_data()

	dst_data, new_spacing = resample(src_data, src_spacing, dst_spacing)

	dst_data[dst_data<0.5] = 0
	dst_data[(dst_data>=0.5)&(dst_data<1.5)] = 1
	dst_data[dst_data>=1.5] = 2
	dst_data = dst_data.astype(np.uint8, copy=False)
	
	new_spacing =  np.append(new_spacing, 1)
	dst_img = nib.Nifti1Image(dst_data, affine=np.diag(new_spacing))

	# print dst_img

	nib.save(dst_img, dst_path)

def load_files_from_folder(folder, prefix, suffix):
	''' Load files from file_folder startswith prefix and endswith suffix
	Return a list of files name
	'''
	files = [f for f in os.listdir(folder) if (f.startswith(prefix) and f.endswith(suffix))]
	return files

def load_volumes(folder):
	files = load_files_from_folder(folder, prefix='volume-', suffix='.nii')
	files.sort(key=lambda name: int(name.split('-')[1].split('.')[0]))
	return files

def load_segmentations(folder):
	files = load_files_from_folder(folder, prefix='segmentation-', suffix='.nii')
	files.sort(key=lambda name: int(name.split('-')[1].split('.')[0]))
	return files

volume_folder = '/home/zlp/dev/medseg/data/lits/Training_Batch'
segmentation_folder = '/home/zlp/dev/medseg/data/lits/Training_Batch'
volumes = load_volumes(volume_folder)
segmentations = load_segmentations(segmentation_folder)

output_folder = '/home/zlp/dev/medseg/data/lits/Training_Batch_isotropic'
spacing = [1,1,1]

for vol in volumes:
	src_path = osp.join(volume_folder, vol)
	dst_path = osp.join(output_folder, vol)
	print src_path, dst_path
	if osp.exists(dst_path):
		continue

	nifti1_image_resample(src_path, dst_path, spacing)
print 'image_resample done'

for seg in segmentations:
	src_path = osp.join(segmentation_folder, seg)
	dst_path = osp.join(output_folder, seg)
	print src_path, dst_path
	if osp.exists(dst_path):
		continue
	
	nifti1_label_resample(src_path, dst_path, spacing)
print 'label_resample done'