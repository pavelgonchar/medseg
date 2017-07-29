import SimpleITK as sitk
import os
import numpy as np

def load_segmentations_test(folder, prefix, suffix):
	files = [f for f in os.listdir(folder) if (f.startswith(prefix) and f.endswith(suffix))]
	files.sort(key=lambda name: int(name.split('-')[2].split('.')[0]))
	return files

def largest_connected_component_mask(image):
	"""
	Fill Holes
	Dilation
	Get connected label
	Get the largest connected label
	"""
	# Fill holes
	fillFilter = sitk.BinaryFillholeImageFilter()
	image_filtered = fillFilter.Execute(image)

	# # Connect nearby volumes by image closing
	# closeFilter = sitk.BinaryMorphologicalClosingImageFilter()
	# closeFilter.SetForegroundValue(1)
	# closeFilter.SetKernelRadius(1)
	# closeFilter.SetKernelType(1)
	# image_filtered = closeFilter.Execute(image_filtered)

	# Connect nearby volumes by image dilation
	dilateFilter = sitk.BinaryDilateImageFilter()
	dilateFilter.SetForegroundValue(1)
	dilateFilter.SetKernelRadius(1)
	dilateFilter.SetKernelType(1)
	image_filtered = dilateFilter.Execute(image_filtered)

	# Get connected label
	ccFilter = sitk.ConnectedComponentImageFilter()
	image_filtered = ccFilter.Execute(image_filtered)

	# Get the largest connected label
	statFilter = sitk.LabelStatisticsImageFilter()
	statFilter.Execute(image_filtered, image_filtered)

	labels = statFilter.GetLabels()[1:]
	maxind = np.argmax(np.array([statFilter.GetCount(l) for l in labels]))
	target_label = labels[maxind]
	print 'labels:{}, maxind:{}, target_label:{}'.format(labels, maxind, target_label)

	# Get the label by binary thresholding
	thresholdFilter = sitk.BinaryThresholdImageFilter()
	thresholdFilter.SetOutsideValue (0)
	thresholdFilter.SetInsideValue (1)
	thresholdFilter.SetLowerThreshold(target_label)
	thresholdFilter.SetUpperThreshold(target_label + 0.01)
	image_filtered = thresholdFilter.Execute(image_filtered)

	# smoothFilter = sitk.BinaryMorphologicalOpeningImageFilter()
	# smoothFilter.SetKernelRadius (2)
	# smoothFilter.SetKernelType(1)
	# image_filtered = smoothFilter.Execute(image_filtered)
	return image_filtered

def largest_connected_component(infolder, outfolder):
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)
	print 'Will save output in folder: {}'.format(outfolder)
	files  = load_segmentations_test(infolder, prefix='test-segmentation-', suffix='.nii')
	print files
	print 'Total number of files: {}\n'.format(len(files))
	
	for f in files:
		finpath = os.path.join(infolder, f)
		foutpath = os.path.join(outfolder, f)
		print 'input file: {}'.format(finpath)
		# Read Image
		reader = sitk.ImageFileReader()
		reader.SetFileName(finpath)
		image = reader.Execute()
		print 'image: ', image.GetPixelIDTypeAsString()
		assert image.GetPixelID() == 1, 'type error'
		# Get largest connected component mask
		mask = largest_connected_component_mask(image)
		print "mask: ", mask.GetPixelIDTypeAsString()
		# Apply largest connected component mask
		mask_filter = sitk.MaskImageFilter()
		image_masked = mask_filter.Execute(image, mask)
		# Write Results
		print "image_masked: ", image_masked.GetPixelIDTypeAsString()
		writer = sitk.ImageFileWriter()
		writer.SetFileName(foutpath)
		writer.Execute(image_masked)
		print 'Output saved to: {}\n'.format(foutpath)
	print '=== DONE ==='

if __name__ == "__main__":
	# root_folder = '/home/zlp/dev/medseg/output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D'
	# infolder = os.path.join(root_folder, 'unet_2d_bn_c2_liver_iter_280000')
	# outfolder = os.path.join(root_folder, 'unet_2d_bn_c2_liver_iter_280000_LCC')
	# root_folder = '/home/zlp/dev/medseg/output/unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D'
	# infolder = os.path.join(root_folder, 'unet_2d_bn_c2_liver_200000LCC_280000LCC_CB')
	# outfolder = os.path.join(root_folder, 'unet_2d_bn_c2_liver_200000LCC_280000LCC_CB_LCC')
	# root_folder = '/home/zlp/dev/medseg/output/uvnet/uvnet_2d_bn_weigted_c3/lits_Test_Batch_trainval_3D'
	# infolder = os.path.join(root_folder, 'uvnet_2d_bn_weigted_1.1.7_iter_250000')
	# outfolder = os.path.join(root_folder, 'uvnet_2d_bn_weigted_1.1.7_iter_250000_LCC')
	root_folder = '/home/zlp/dev/medseg/output/uvnet/uvnet_2d_bn_weigted_c3/lits_Test_Batch_trainval_3D'
	infolder = os.path.join(root_folder, 'uvnet_2d_bn_weigted_1.1.7_iter_250000_LCC_MG')
	outfolder = os.path.join(root_folder, 'uvnet_2d_bn_weigted_1.1.7_iter_250000_LCC_MG_LCC')
	largest_connected_component(infolder, outfolder)