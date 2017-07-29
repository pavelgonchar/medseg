#!/usr/bin/env python
# encoding: utf-8


"""Compute minibatch blobs for training a network."""

import numpy as np
from utils.blob import load_data
from lits.config import cfg
from utils.blob import im_list_to_blob, seg_list_to_blob, prep_weight_for_blob, prepare_data_unit, ims_vis

def get_minibatch(imdb):
    """Given a imdb, construct a minibatch sampled from it."""
    num_images = len(imdb)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), 'num_images ({}) must divide BATCH_SIZE ({})'.format(num_images, cfg.TRAIN.BATCH_SIZE)

    patch_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    if (patch_per_image > 1):
        assert(cfg.TRAIN.USE_CROPPED,'When patch_per_image greater than 1, TRAIN.USE_CROPPED must be ON!')

    # Sample random scales to use for each image in this batch
    random_scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES), size=num_images)
    random_rotation_inds = np.random.randint(0, high=len(cfg.TRAIN.ROTATIONS), size=num_images)

    # Get the input blobs, formatted for caffe
    blobs = get_image_blob(imdb, patch_per_image, scale_inds=random_scale_inds, rotation_inds=random_rotation_inds)
    return blobs

def get_image_blob(imdb, patch_per_image, scale_inds, rotation_inds):
    """
    Builds an input blob from imdb at
    the specified scales, flipped, rotated, cropped
    """
    processed_im_patchs = []
    processed_gt_patchs = []
    for ind_imdb in xrange(len(imdb)):
        """  Get Item Infos
        """
        # im_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/Images/volume-100/volume-100_slice_551.jpg'
        # gt_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/Segmentation/segmentation-100/segmentation-100_slice_551.jpg'
        # im_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/volume-0.nii'
        # gt_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/segmentation-0.nii'
        im_path = imdb[ind_imdb]['image']
        gt_path = imdb[ind_imdb]['gt']
        scale = cfg.TRAIN.SCALES[scale_inds[ind_imdb]]
        rotation = cfg.TRAIN.ROTATIONS[rotation_inds[ind_imdb]]
        """ Prepare each data unit
        """
        im_prepared, gt_prepared = prepare_data_unit(im_path=im_path, gt_path=gt_path, pixel_means=cfg.TRAIN.PIXEL_MEANS,
            scale=scale, max_size=cfg.TRAIN.MAX_SIZE, cropped_size=cfg.TRAIN.CROPPED_SIZE, patch_per_image=patch_per_image,
            hu_window=cfg.TRAIN.HU_WINDOW, data_range=cfg.TRAIN.DATA_RANGE, rotation=rotation, use_flip=cfg.TRAIN.USE_FLIP)
        processed_im_patchs.extend(im_prepared)
        processed_gt_patchs.extend(gt_prepared)
    """ Create a blob to hold the input images, gts and class_weights
    Axis order will become: (batch elem, channel, height, width, depth)
    """
    im_blob = im_list_to_blob(processed_im_patchs, im_type=cfg.TRAIN.SEGMENTATION_MODE)
    gt_blob = seg_list_to_blob(processed_gt_patchs, im_type=cfg.TRAIN.SEGMENTATION_MODE)
    weight_blob = prep_weight_for_blob(gt_blob, cfg.TRAIN.CLASS_WEIGHT)
    print im_blob.dtype, gt_blob.dtype, weight_blob.dtype
    print im_blob.shape, gt_blob.shape, weight_blob.shape
    # ims_vis_list = []
    # for batch in xrange(im_blob.shape[0]):
    #    ims_vis_list.extend([im_blob[batch,0,:,:], gt_blob[batch,0,:,:], weight_blob[batch,0,:,:]])
    # ims_vis(ims_vis_list)
    # ims_vis([im_blob[0, 0, :, :, im_blob.shape[4]//2], gt_blob[0, 0, :, :, gt_blob.shape[4]//2], weight_blob[0, 0, :, :, gt_blob.shape[4]//2]])
    """ Return Blobs
    """
    blobs = {'data': im_blob, 'label': gt_blob, 'label_weight': weight_blob}
    return blobs

# def get_image_blob(imdb, patch_per_image, scale_inds, rotation_inds, im_type='2D'):
#     # if im_type == '3D':
#     #     im_blob, gt_blob, weight_blob = _get_image_blob_3d(imdb, patch_per_image, scale_inds)
#     # else:
#     #     im_blob, gt_blob, weight_blob = _get_image_blob(imdb, patch_per_image, scale_inds)
#     im_blob, gt_blob, weight_blob = _get_image_blob(imdb, patch_per_image, scale_inds)
#     blobs = {'data': im_blob, 'label': gt_blob, 'label_weight': weight_blob}

#     return blobs

# def _get_image_blob(imdb, patch_per_image, scale_inds):
#     """Builds an input blob from imdb at
#         the specified scales, flipped, rotated, cropped
#     """
#     processed_im_patchs = []
#     processed_gt_patchs = []
#     for i in xrange(len(imdb)):
#         # ''' Load an image in grayscale
#         # '''
#         # im = cv2.imread(imdb[i]['image'], cv2.CV_LOAD_IMAGE_GRAYSCALE)
#         # gt = cv2.imread(imdb[i]['gt'], cv2.CV_LOAD_IMAGE_GRAYSCALE)
#         im = cv2.imread(imdb[i]['image'])
#         gt = cv2.imread(imdb[i]['gt'])
#         '''Transform im to float type
#         '''
#         im = im.astype(np.float32, copy=False)
#         gt = gt.astype(np.float32, copy=False)
#         ''' Transform 100 to 1, 200 to 2
#         '''
#         gt[gt < 50] = 0.0
#         gt[(gt>=50)&(gt<150)] = 1.0
#         gt[gt>=150] = 2.0
#         ''' Image processing:
#         Rotation
#         Mean substraction
#         Resize
#         '''
#         ''' Rotation must be performed before the mean substraction
#         '''
#         if cfg.TRAIN.USE_ROTATED and (np.mod(np.random.randint(4),3)==0):
#             angle = cfg.TRAIN.ROTATIONS[np.random.randint(len(cfg.TRAIN.ROTATIONS))]
#             rotated_images = rotation_2d_images([im, gt], angle)
#             im = rotated_images[0]
#             gt = rotated_images[1]
#         ''' Load image as height x width x channel
#         '''
#         if len(im.shape) != 3:
#             im = im[..., np.newaxis]
#             gt = gt[..., np.newaxis]
#         ''' Map src_range value to dst_range value
#         '''
#         im = normalizer(im, src_range=[0., 255.], dst_range=[0., 1.])
#         im = prep_im_for_blob(im, pixel_means=0.01, target_size=cfg.TRAIN.SCALES[scale_inds[i]], max_size=cfg.TRAIN.MAX_SIZE, im_type='2D')
#         gt = prep_seg_for_blob(gt, pixel_means=0, target_size=cfg.TRAIN.SCALES[scale_inds[i]], max_size=cfg.TRAIN.MAX_SIZE, im_type='2D')
#         ''' Horizontal flip
#         '''
#         if imdb[i]['flipped']:
#             im = im[:, ::-1, :]
#             gt = gt[:, ::-1, :]
#         ''' Random crop patch from image
#         '''
#         if cfg.TRAIN.USE_CROPPED:
#             cropped_patches = random_crop_patches([im, gt], cfg.TRAIN.CROPPED_SIZE, patch_per_image)
#             processed_im_patchs.extend(cropped_patches[0])
#             processed_gt_patchs.extend(cropped_patches[1])
#         else:
#             processed_im_patchs.append(im)
#             processed_gt_patchs.append(gt)

#     ''' Create a blob to hold the input blobs
#     Axis order will become: (batch elem, channel, height, width)
#     '''
#     im_blob = im_list_to_blob(processed_im_patchs, im_type='2D')
#     gt_blob = seg_list_to_blob(processed_gt_patchs, im_type='2D')
#     # label should have (batch elem, 1, height, width)
#     gt_blob = gt_blob[:,0,:,:]
#     gt_blob = gt_blob[:,np.newaxis,:,:]
#     weight_blob = prep_weight_for_blob(gt_blob, cfg.TRAIN.CLASS_WEIGHT)
#     # print im_blob.shape, gt_blob.shape, weight_blob.shape
#     #ims_vis_list = []
#     #for batch in xrange(im_blob.shape[0]):
#     #    ims_vis_list.extend([im_blob[batch,0,:,:], gt_blob[batch,0,:,:], weight_blob[batch,0,:,:]])
#     #ims_vis(ims_vis_list)
#     return im_blob, gt_blob, weight_blob

# def _get_image_blob_3d(imdb, patch_per_image, scale_inds):
#     """Builds an input blob from imdb at
#         the specified scales, flipped, rotated, cropped
#     """
#     processed_im_patchs = []
#     processed_gt_patchs = []
#     for i in xrange(len(imdb)):
#         ''' Load data (height, width, depth)
#         '''
#         im = nib.load(imdb[i]['image']).get_data()
#         gt = nib.load(imdb[i]['gt']).get_data()
#         '''Transform im to float type
#         '''
#         im = im.astype(np.float32, copy=False)
#         gt = gt.astype(np.float32, copy=False)

#         # Transform label 1 to 1, 2 to 2
#         gt[gt<0.5] = 0.0
#         gt[(gt>=0.5)&(gt<1.5)] = 1.0
#         gt[gt>=1.5] = 2.0
#         # gt[gt>=1.5] = 1.0

#         ''' Trim volume to only contain gt, remove irrelavant background
#         min_size set to cfg.TRAIN.CROPPED_SIZE
#         '''
#         row_col_page_b, new_size = trim_volume(gt, min_size=cfg.TRAIN.CROPPED_SIZE, pad=[0,0,8])
#         row_col_page_e = row_col_page_b + new_size
#         # im = im[row_col_page_b[0]:row_col_page_e[0], row_col_page_b[1]:row_col_page_e[1], row_col_page_b[2]:row_col_page_e[2]]
#         # gt = gt[row_col_page_b[0]:row_col_page_e[0], row_col_page_b[1]:row_col_page_e[1], row_col_page_b[2]:row_col_page_e[2]]
#         im = im[:,:,row_col_page_b[2]:row_col_page_e[2]]
#         gt = gt[:,:,row_col_page_b[2]:row_col_page_e[2]]
#         # ims_vis([im[:,:,im.shape[2]//2], gt[:,:,gt.shape[2]//2]])
#         ''' Image Processing
#         Rotation
#         Mean substraction
#         Resize
#         '''
#         '''Rotation must be performed before the mean substraction
#         ''' 
#         if cfg.TRAIN.USE_ROTATED and (np.mod(np.random.randint(4),3)==0):
#             pass
#         """ Apply Hounsfield Unit Window  
#         The Hounsfield unit values will be windowed in the range
#         HU_WINDOW to exclude irrelevant organs and objects.
#         """
#         HU_WINDOW = [-100, 400]
#         im = hounsfield_unit_window(im, HU_WINDOW)
#         # ims_vis([im[:,:,im.shape[2]//2], gt[:,:,gt.shape[2]//2]])
#         ''' Map src_range value to dst_range value
#         '''
#         src_range = HU_WINDOW
#         dst_range = [0, 1]
#         im = normalizer(im, src_range, dst_range)
#         # ims_vis([im[:,:,im.shape[2]//2], gt[:,:,gt.shape[2]//2]])
        
#         im = prep_im_for_blob(im, pixel_means=0, target_size=cfg.TRAIN.SCALES[scale_inds[i]], max_size=cfg.TRAIN.MAX_SIZE, im_type='3D')
#         gt = prep_seg_for_blob(gt, pixel_means=0, target_size=cfg.TRAIN.SCALES[scale_inds[i]], max_size=cfg.TRAIN.MAX_SIZE, im_type='3D')
#         # ims_vis([im[:,:,im.shape[2]//2], gt[:,:,gt.shape[2]//2]])
#         '''Horizontal flip
#         '''
#         if imdb[i]['flipped']:
#             im = im[:,::-1,:]
#             gt = gt[:,::-1,:]
#         # ims_vis([im[:,:,im.shape[2]//2], gt[:,:,gt.shape[2]//2]])
#         ''' Random Crop patch from image
#         '''
#         if cfg.TRAIN.USE_CROPPED:
#             cropped_patches = random_crop_patches([im, gt], cfg.TRAIN.CROPPED_SIZE, patch_per_image)
#             processed_im_patchs.extend(cropped_patches[0])
#             processed_gt_patchs.extend(cropped_patches[1])
#         else:
#             processed_im_patchs.append(im)
#             processed_gt_patchs.append(gt)
#         # if cfg.TRAIN.USE_CROPPED:
#         #     diff_size = np.array(im.shape) - np.array(cfg.TRAIN.CROPPED_SIZE)
#         #     assert(np.min(diff_size)>=0), 'CROPPED_SIZE must not greater than image size'
#         #     rows = np.random.randint(diff_size[0] + 1, size=patch_per_image)
#         #     cols = np.random.randint(diff_size[1] + 1, size=patch_per_image)
#         #     pages = np.random.randint(diff_size[2] + 1, size=patch_per_image)
#         #     for j in xrange(patch_per_image):
#         #         im_patch = im[rows[j]:rows[j]+cfg.TRAIN.CROPPED_SIZE[0], cols[j]:cols[j]+cfg.TRAIN.CROPPED_SIZE[1], pages[j]:pages[j]+cfg.TRAIN.CROPPED_SIZE[2]]
#         #         gt_patch = gt[rows[j]:rows[j]+cfg.TRAIN.CROPPED_SIZE[0], cols[j]:cols[j]+cfg.TRAIN.CROPPED_SIZE[1], pages[j]:pages[j]+cfg.TRAIN.CROPPED_SIZE[2]]
#         #         processed_im_patchs.append(im_patch)
#         #         processed_gt_patchs.append(gt_patch)
#         #         # ims_vis([im_patch[:,:,im_patch.shape[2]//2], gt_patch[:,:,gt_patch.shape[2]//2]])
#         # else:
#         #     processed_im_patchs.append(im)
#         #     processed_gt_patchs.append(gt)
#     ''' Create a blob to hold the input images, gts and class_weights
#     Axis order will become: (batch elem, channel, height, width, depth)
#     '''
#     im_blob = im_list_to_blob(processed_im_patchs, im_type='3D')
#     gt_blob = seg_list_to_blob(processed_gt_patchs, im_type='3D')
#     weight_blob = prep_weight_for_blob(gt_blob, cfg.TRAIN.CLASS_WEIGHT)
#     # print im_blob.dtype, gt_blob.dtype, weight_blob.dtype
#     # print im_blob.shape, gt_blob.shape, weight_blob.shape
#     # ims_vis([im_blob[0, 0, :, :, im_blob.shape[4]//2], gt_blob[0, 0, :, :, gt_blob.shape[4]//2], weight_blob[0, 0, :, :, gt_blob.shape[4]//2]])
#     return im_blob, gt_blob, weight_blob


