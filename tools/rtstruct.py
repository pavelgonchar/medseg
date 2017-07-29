import os, sys, glob
import numpy as np
import dicom
import matplotlib.pyplot as plt
# %matplotlib inline
from skimage.draw import polygon
import SimpleITK as sitk
import nibabel as nib
import scipy.ndimage

def get_pixels_hu(slices):
	image = np.stack([s.pixel_array for s in slices], axis=-1)
	# Convert to int16 (from sometimes int16),
	# should be possible as values should always be low enough (<32k)
	image = image.astype(np.int16)
	# Set outside-of-scan pixels to 0
	# The intercept is usually -1024, so air is approximately 0
	image[image == -2000] = 0
	# Convert to Hounsfield units (HU)
	for slice_number in range(len(slices)):
		intercept = slices[slice_number].RescaleIntercept
		slope = slices[slice_number].RescaleSlope
		# print slope, intercept
		if slope != 1:
			image[:,:,slice_number] = slope * image[:,:,slice_number].astype(np.float64)
			image[:,:,slice_number] = image[:,:,slice_number].astype(np.int16)

		image[:,:,slice_number] += np.int16(intercept)

	return np.array(image, dtype=np.int16)

def read_structure(structure):
	contours = []
	for i in range(len(structure.ROIContourSequence)):
		contour = {}
		contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
		# print contour['color']
		contour['number'] = structure.ROIContourSequence[i].RefdROINumber
		# print contour['number']
		contour['name'] = structure.StructureSetROISequence[i].ROIName
		# print contour['name']
		assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
		contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
		contours.append(contour)
		# print contour['number'], contour['name']
	return contours

def get_mask(contours, slices, image, scale):
	z = [round(s.ImagePositionPatient[2],1) for s in slices]
	# x,y,z --> col, row, page
	pos_r = slices[0].ImagePositionPatient[1]
	spacing_r = slices[0].PixelSpacing[1] / scale[1]
	pos_c = slices[0].ImagePositionPatient[0]
	spacing_c = slices[0].PixelSpacing[0] / scale[0]

	label = np.zeros_like(image, dtype=np.uint8)
	# since contour numbers are inconsistant cross different patients, we manually design the label mapping as follow
	label_mapping_dict = dict()
	label_mapping_dict['Esophagus'] = 1
	label_mapping_dict['Heart'] = 2
	label_mapping_dict['SpinalCord'] = 3
	label_mapping_dict['Lung_L'] = 4
	label_mapping_dict['Lung_R'] = 5
	for con in contours:
		name = con['name']
		num = label_mapping_dict[name]
		# print name, num
		# num = int(con['number'])
		for c in con['contours']:
			nodes = np.array(c).reshape((-1, 3))
			# check contours in the same current slice
			assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
			z_index = z.index(round(nodes[0, 2],1))
			r = (nodes[:, 1] - pos_r) / spacing_r
			c = (nodes[:, 0] - pos_c) / spacing_c
			rr, cc = polygon(r, c)
			# row, col, page
			label[rr, cc, z_index] = num

	colors = tuple(np.array([con['color'] for con in contours]) / 255.0)
	return label, colors

def ndimage_zoom(image, dst_shape, mode='constant'):
	''' Resize image to dst_shape
	Return resized image
	mode ('constant, nearest, reflect or wrap).
	'''
	src_image = image.astype(np.float32, copy=True)
	resize_factor = np.array(dst_shape, dtype=np.float) / src_image.shape
	dst_image = scipy.ndimage.interpolation.zoom(src_image, resize_factor, mode=mode)
	return dst_image

def ims_vis(axes, ims):
	'''Function to display row of images'''
	for i, im in enumerate(ims):
		axes[i].imshow(im, cmap='gray')
	plt.show()
	plt.pause(0.00001)

train_data_path = "/home/zlp/data/lpzhang/LCTSC/DOI"
output_dir = "/home/zlp/data/lpzhang/LCTSC/NII256"
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
train_patients = [os.path.join(train_data_path, name) for name in os.listdir(train_data_path) if os.path.isdir(os.path.join(train_data_path, name))]

for patient in train_patients:
	print 'current patient: ', patient
	for subdir, dirs, files in os.walk(patient):
		dcms = glob.glob(os.path.join(subdir, "*.dcm"))
		if len(dcms) == 1:
			# read RTSTRUCT
			structure = dicom.read_file(os.path.join(subdir, files[0]))
			#contours = read_structure(structure)
		elif len(dcms) > 1:
			# Read the slices from the dicom file
			slices = [dicom.read_file(dcm) for dcm in dcms]
			print slices[0].SOPInstanceUID
			print slices[0].StudyInstanceUID
			print slices[0].SeriesInstanceUID
			print slices[0].InstanceNumber
			print slices[0].ImagePositionPatient[2]
			print slices[1].SOPInstanceUID
			print slices[1].StudyInstanceUID
			print slices[1].SeriesInstanceUID
			print slices[1].InstanceNumber
			print slices[1].ImagePositionPatient[2]
			exit()
			# Sort the dicom slices in their respective order
			slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
			# Get the pixel values for all the slices
			# image = np.stack([s.pixel_array for s in slices], axis=-1)
			# Get the pixel values for all the slices with HU values
			image = get_pixels_hu(slices)

	# Resize image to 512x512xZ to 256x256xZ
	dst_shape = [256,256,image.shape[2]]
	image_zoomed = ndimage_zoom(image, dst_shape)
	image_zoomed = np.array(image_zoomed, dtype=np.int16)
	# Get mask
	zoom_scale = np.array(dst_shape, np.float32)/np.array(image.shape)
	label_zoomed, colors = get_mask(contours, slices, image_zoomed, zoom_scale)
	# label_zoomed = ndimage_zoom(label, dst_shape, mode='nearest')
	# label_zoomed[label_zoomed<0.5] = 0
	# label_zoomed[(label_zoomed>=0.5)&(label_zoomed<1.5)] = 1
	# label_zoomed[(label_zoomed>=1.5)&(label_zoomed<2.5)] = 2
	# label_zoomed[(label_zoomed>=2.5)&(label_zoomed<3.5)] = 3
	# label_zoomed[(label_zoomed>=3.5)&(label_zoomed<4.5)] = 4
	# label_zoomed[(label_zoomed>=4.5)] = 5
	# label_zoomed = np.array(label_zoomed, dtype=np.uint8)
	print image_zoomed.shape, image_zoomed.dtype
	print label_zoomed.shape, label_zoomed.dtype

	# Save
	image_path = os.path.join(output_dir, os.path.split(patient)[1]+'.nii')
	gt_path = os.path.join(output_dir, os.path.split(patient)[1]+'_gt.nii')
	nii_spacing =  [slices[0].PixelSpacing[0]/zoom_scale[0],slices[0].PixelSpacing[1]/zoom_scale[1], slices[0].SliceThickness/zoom_scale[2], 1]
	print nii_spacing
	# nii_image = nib.Nifti1Image(image, affine=np.diag(nii_spacing))
	nii_image = nib.Nifti1Image(image_zoomed, affine=np.diag(nii_spacing))
	nib.save(nii_image, image_path)
	# nii_label = nib.Nifti1Image(label, affine=np.diag(nii_spacing))
	nii_label = nib.Nifti1Image(label_zoomed, affine=np.diag(nii_spacing))
	nib.save(nii_label, gt_path)
print 'done'

	# Plot to check slices, for example 50 to 59
	# plt.figure(figsize=(15, 15))
	# for i in range(9):
	# 	plt.subplot(3, 3, i + 1)
	# 	plt.imshow(image[..., i + 50], cmap="gray")
	# 	plt.contour(label[..., i + 50], levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors=colors)
	# 	plt.axis('off')
	# plt.show()
	# exit()
