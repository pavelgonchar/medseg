#!/usr/bin/env python
# encoding: utf-8

"""Blob helper functions."""
import numpy as np
import cv2
import nibabel as nib
from scipy.ndimage.interpolation import zoom, rotate
import matplotlib.pyplot as plt
import os.path as osp


def im_list_to_blob(ims, im_type='2D'):
    """Convert a list of images into a network input has follow axis order:
    (batch elem, channel, height, width) for 2d images or
    (batch elem, channel, height, width, depth) for 3d images

    Assumes images are already prepared (means subtracted, BGR order, ...).
    Input list has axis order (height, width, channel) for 2d images,
    and (height, width, depth) for 3d images.
    """
    assert im_type in ('2D', '3D'), 'image type must be either 2D or 3D'
    blob = _im_list_to_blob(ims, Dtype=np.float32)
    if im_type == '3D':
        # Add newaxis(channel) to blob and blob will become 5 dimensions
        # Move channels (axis 0) to axis 1
        # Axis order will become: (batch elem, channel, height, width, depth) for 3d images
        blob = blob[np.newaxis, ...]
        channel_swap = (1, 0, 2, 3, 4)
        blob = blob.transpose(channel_swap)
    else:
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width) for 2d images
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
    return blob

def seg_list_to_blob(ims, im_type='2D'):
    """Convert a list of images into a network input has follow axis order:
    (batch elem, channel, height, width) for 2d images or
    (batch elem, channel, height, width, depth) for 3d images

    Assumes images are already prepared (means subtracted, BGR order, ...).
    Input list has axis order (height, width, channel) for 2d images,
    and (height, width, depth) for 3d images.
    """
    assert im_type in ('2D', '3D'), 'image type must be either 2D or 3D'
    blob = _im_list_to_blob(ims, Dtype=np.uint8)
    if im_type == '3D':
        # Add newaxis(channel) to blob and blob will become 5 dimensions
        # Move channels (axis 0) to axis 1
        # Axis order will become: (batch elem, channel, height, width, depth) for 3d images
        blob = blob[np.newaxis, ...]
        channel_swap = (1, 0, 2, 3, 4)
        blob = blob.transpose(channel_swap)
    else:
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width) for 2d images
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
    return blob

def _im_list_to_blob(ims, Dtype=np.float32):
    """Convert a list of images(BGR) into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    assert len(max_shape) == 3, 'image must have 3 dimensions'
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], max_shape[2]), dtype=Dtype)
    for i in xrange(num_images):
        im = ims[i].astype(dtype=Dtype, copy=False)
        blob[i, 0:im.shape[0], 0:im.shape[1], 0:im.shape[2]] = im
    return blob

def scale_image(im, target_size, max_size, im_type='2D'):
    """ Scale an image for use in a blob.
    """
    assert im_type in ('2D', '3D'), 'image type must be either 2D or 3D'
    if im_type == '3D':
        # im, im_scale = ndimage_zoom(im, target_size)
        pass
    else:
        im, im_scale = image_zoom(im, target_size, max_size)

    return im

def image_zoom(im, target_size, max_size):
    """ Scale an image for use in a blob.
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR) 
    # For input im has shape height x width x 1, im shape will become height x width after resize
    # so need to recover its shape
    if len(im_shape) != len(im.shape):
        im = im[..., np.newaxis]
    return im, im_scale

def ndimage_zoom(image, target_size):
    ''' Resize image to target_size
    Return resized image
    '''
    src_image = image.astype(np.float32, copy=True)
    resize_factor = np.array(target_size, dtype=np.float) / src_image.shape
    dst_image = zoom(src_image, resize_factor)
    return dst_image, resize_factor

def prep_weight_for_blob(blob, class_weight):
    """ Input blob axis order is: 
    (batch elem, channel, height, width) for 2d image
    (batch elem, channel, height, width, depth) for 3d image
    Output weight_blob tackle the imbalance classes.
    """
    weight_blob = blob.astype(np.float32, copy=True)
    weight_blob[blob<0.5] = class_weight[0]
    weight_blob[(blob>=0.5)&(blob<1.5)] = class_weight[1]
    weight_blob[blob>=1.5] = class_weight[2]

    return weight_blob

def hounsfield_unit_window(im, hu_window):
    """ The Hounsfield unit values will be windowed in the range
    [min_bound, max_bound] to exclude irrelevant organs and objects.
    """
    hu_window = np.array(hu_window, np.float32)

    im[im<hu_window[0]] = hu_window[0]
    im[im>hu_window[1]] = hu_window[1]
    return im

def normalizer(im, src_range, dst_range=[0.0, 1.0]):
    # Normalize src_range to dst_range
    src_range = np.array(src_range, np.float32)
    dst_range = np.array(dst_range, np.float32)
    
    im = (im - src_range[0]) / (src_range[1] - src_range[0])
    im = im * (dst_range[1] - dst_range[0]) + dst_range[0]
    im[im<dst_range[0]] = dst_range[0]
    im[im>dst_range[1]] = dst_range[1]
    return im

def trim_volume(im, min_size, pad=[0,0,0]):
    """Volume will contain all the segs and remove irrelavant background
    Return: the row_col_page_b index and the trimmed image size
    """
    orig_size = np.array(im.shape)
    min_size = np.array(min_size)
    pad = np.array(pad)
    assert(np.min(orig_size - min_size) >= 0), 'orig_size {} must greater than min_size {}'.format(orig_size, min_size)
    # Find index interval that contain all the segs
    nonzero_row = np.flatnonzero((np.sum(im, axis=(1,2))))
    nonzero_col = np.flatnonzero((np.sum(im, axis=(0,2))))
    nonzero_page = np.flatnonzero((np.sum(im, axis=(0,1))))
    if nonzero_row.size == 0 or nonzero_col.size == 0 or nonzero_page.size == 0:
        return np.array([0, 0, 0]), min_size

    # Volume range [row_col_page_b, row_col_page_e] will contain all the segs, include row_col_page_e
    row_col_page_b = np.array([nonzero_row[0], nonzero_col[0], nonzero_page[0]])
    row_col_page_e = np.array([nonzero_row[-1], nonzero_col[-1], nonzero_page[-1]])
    # Volume range [row_col_page_b:row_col_page_e] will contain all the segs, not include row_col_page_e
    row_col_page_e += 1
    new_size = row_col_page_e - row_col_page_b
    # add padding to seg
    row_col_page_b -= pad
    new_size += 2*pad
    # Pad diff_size/2 to both side such that new_size will be not smaller than min_size
    diff_size = min_size - new_size
    diff_size = np.ceil(diff_size/2.0).astype(dtype=int)
    index_diff = (diff_size>0)
    row_col_page_b[index_diff] -= diff_size[index_diff]
    new_size[index_diff] = min_size[index_diff]
    # Prevent new_size exceed orig_size
    new_size = np.minimum(new_size, orig_size)
    # Prevent row_col_page_b out of bound
    row_col_page_b[row_col_page_b<0] = 0
    # Prevent row_col_page_e out of bound
    row_col_page_e = row_col_page_b + new_size
    index_out_of_bound = ((row_col_page_e - orig_size) > 0)
    row_col_page_b[index_out_of_bound] = (orig_size - new_size)[index_out_of_bound]

    return row_col_page_b, new_size

def trim_volume_square(im, min_size, pad=[0,0,0]):
    # """Volume will contain all the segs and remove irrelavant background
    # Return: the row_col_page_b index and the trimmed image size
    # """
    orig_size = np.array(im.shape)
    row_col_page_b, new_size = trim_volume(im, min_size, pad=pad)

    # Get Center
    row_col_page_center = row_col_page_b + new_size/2.0
    # Update size
    new_size[0:2] = np.max(new_size[0:2])
    # Update row_col_page_b
    row_col_page_b[0:2] = np.round(row_col_page_center[0:2] - new_size[0:2]/2.0).astype(dtype=int)
    # Prevent row_col_page_b out of bound
    row_col_page_b[row_col_page_b<0] = 0
    # Prevent row_col_page_e out of bound
    row_col_page_e = row_col_page_b + new_size
    index_out_of_bound = ((row_col_page_e - orig_size) > 0)
    row_col_page_b[index_out_of_bound] = (orig_size - new_size)[index_out_of_bound]

    return row_col_page_b, new_size

def random_crop_patches(images, patch_size, number_of_patches):
    ''' Random Crop Patches from a list of images.
    Each image in images has number_of_patches number of patches.
    Same level patches have same random crop index for all images at this level.
    Return a list of cropped patches for each images.
    '''
    ref_image = images[0]
    # Check all image in images have same size.
    for image in images:
        assert np.count_nonzero(np.array(ref_image.shape)-np.array(image.shape))==0, 'images size must equal'
    # Check patch size not greater than ref_image size
    diff_size = np.array(ref_image.shape) - np.array(patch_size)
    assert (np.min(diff_size) >= 0), 'patch size {} must not greater than image size {}'.format(patch_size, ref_image.shape)
    # Random row, column, page index
    rows = np.random.randint(diff_size[0] + 1, size=number_of_patches)
    cols = np.random.randint(diff_size[1] + 1, size=number_of_patches)
    pages = np.random.randint(diff_size[2] + 1, size=number_of_patches)

    images_patches = []
    for image in images:
        patches = []
        for i in xrange(number_of_patches):
            patch = image[rows[i]:rows[i]+patch_size[0], cols[i]:cols[i]+patch_size[1], pages[i]:pages[i]+patch_size[2]]
            patches.append(patch)
        images_patches.append(patches)

    return images_patches


def rotation_2d_images(images, angle):
    """As in mathematics, row:0, column:0 is the top-left element of the matrix.
    In a perfect world we would choose the coordinate system of points/images to be (origin in top-left):
    0/0 ---column--->                0/0 ---x--->
     |                                |
    row                               y
     |                                | 
     v                                v
    """
    ref_image = images[0]
    for image in images:
        assert np.count_nonzero(np.array(ref_image.shape)-np.array(image.shape))==0, 'images size must equal'

    rows, cols = ref_image.shape[0:2]
    # center(point), angle, scale
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle , scale=1.0)
    # dsize: size of the output image.
    rotated_images = []
    for image in images:
        rotated_image = cv2.warpAffine(image, M, (cols, rows))
        rotated_images.append(rotated_image)

    return rotated_images

def load_data(image_path, flags=1):
    """ flags: 1 --> cv2.IMREAD_COLOR
        flags: 0 --> cv2.IMREAD_GRAYSCALE
        flags:-1 --> cv2.IMREAD_UNCHANGED
    """
    if not osp.exists(image_path):
        print 'Path does not exist: {}'.format(image_path)
        return None
    ext = osp.splitext(image_path)[1]
    assert ext in ('.jpg', '.png', '.nii', '.npy'), '{} not supported'.format(ext)
    image_header = None
    if ext in ('.jpg', '.png'):
        ''' Load image as height x width x channel
        '''
        image_data = cv2.imread(image_path, flags)
        if len(image_data.shape) != 3:
            image_data = image_data[..., np.newaxis]
        image_type = '2D'
    elif ext in ('.npy'):
        image_data = np.load(image_path)
        if len(image_data.shape) != 3:
            image_data = image_data[..., np.newaxis]
        image_type = '2D'
    elif ext in ('.nii'):
        image = nib.load(image_path)
        image_data = image.get_data()
        image_header = image.header
        image_type = '3D'
    else:
        print('{} not supported'.format(ext))
    # construct metadata
    metadata = {}
    metadata['image_data'] = image_data
    metadata['image_type'] = image_type
    metadata['image_header'] = image_header

    return metadata

def save_data(image, output_path):
    ext = osp.splitext(output_path)[1]
    assert ext in ('.jpg', '.png', '.nii', '.npy'), '{} not supported'.format(ext)
    if ext in ('.jpg', '.png'):
        cv2.imwrite(output_path, image)
    elif ext in ('.npy'):
        np.save(output_path, image)
    elif ext in ('.nii'):
        fake_spacing =  [1,1,5,1]
        nii_image = nib.Nifti1Image(image, affine=np.diag(fake_spacing))
        nib.save(nii_image, output_path)
    else:
        print('{} not supported'.format(ext))

def split2chunks(image_shape, chunk_shape, stride):
    ''' Extract chunks from image based on chunk_shape and stride,
    return a list of chunks' indexes
    Each stores a chunks' index refer to the original image position, in the form of:
    [[row_start,row_end], [colume_start, colume_end], [page_start, page_end]]

    Each chunk data could obtained from image data in a following way:
    chunk_data = image[row_start:row_end, colume_start:colume_end, page_start:page_end]
    '''
    chunks_index = []
    split_shape = np.int_(np.ceil((np.array(image_shape) - np.array(chunk_shape))/np.float32(np.array(stride)))) + 1
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

def prepare_data_unit(im_path, gt_path, pixel_statistics,
    target_size, max_size, trim_params, class_params, adjacnet, bg, cropped_size, patch_per_image,
    hu_window=[0,255], data_range=[0,1], rotation=0, use_flip=False):
    """ Processing single unit data, include:
    Load image metadata, Trim Volume to remove irrelavant background
    Data Augmentation: Rotation, Flip
    Apply Hounsfield Unit Window, Mean subtractiong and Normalization
    Scale and Crop
    Return processed im_patch and gt_patch
    """
    """ Load image in grayscale if image type is 2D
    """
    im_metadata = load_data(im_path, flags=0)
    gt_metadata = load_data(gt_path, flags=0)
    assert ((im_metadata is not None) and (gt_metadata is not None)), 'im open failed'
    # parse metadata
    im = im_metadata['image_data']
    gt = gt_metadata['image_data']
    im_type = im_metadata['image_type']
    """ ADJACENT
    Load Adjacent Slices if image type is 2D
    """
    # Load adjacent im slices
    if adjacnet and im_type=="2D":
        adj_im_data = prepare_adj_data(im_path)
        im_merged = np.zeros((im.shape[0], im.shape[1], 5), dtype=np.float32)
        im_merged[:,:,0] = adj_im_data[0]
        im_merged[:,:,1] = adj_im_data[1]
        im_merged[:,:,2] = im[:,:,0]
        im_merged[:,:,3] = adj_im_data[2]
        im_merged[:,:,4] = adj_im_data[3]
        im = im_merged
    im = im.astype(np.float32, copy=False)
    # Load adjacent gt slices
    if adjacnet and im_type=="2D":
        adj_gt_data = prepare_adj_data(gt_path)
        gt_merged = np.zeros((gt.shape[0], gt.shape[1], 5), dtype=np.float32)
        gt_merged[:,:,0] = adj_gt_data[0]
        gt_merged[:,:,1] = adj_gt_data[1]
        gt_merged[:,:,2] = gt[:,:,0]
        gt_merged[:,:,3] = adj_gt_data[2]
        gt_merged[:,:,4] = adj_gt_data[3]
        gt = gt_merged
    gt = gt.astype(np.float32, copy=False)
    ### Clean the bg of im
    if bg.CLEAN:
        assert gt is not None, 'gt must be not None'
        if bg.BRIGHT:
            fg = np.zeros(im.shape, dtype=np.float)
            fg[gt>=0.5] = im[gt>=0.5]
            im[gt<0.5] = np.amax(fg)
        else:
            bg = np.zeros(im.shape, dtype=np.float)
            bg[gt<0.5] = im[gt<0.5]
            im[gt<0.5] = np.amin(bg)
    ### class params
    assert class_params.NUMBER in (2,3), 'class number error'
    assert len(class_params.SPLIT)==(class_params.NUMBER-1), 'class split error'
    if class_params.NUMBER == 2:
        gt[gt<class_params.SPLIT[0]] = 0.0
        gt[gt>=class_params.SPLIT[0]] = 1.0
    else:
        gt[gt<class_params.SPLIT[0]] = 0.0
        gt[(gt>=class_params.SPLIT[0])&(gt<class_params.SPLIT[1])] = 1.0
        gt[gt>=class_params.SPLIT[1]] = 2.0
    """ Trim volume is to obtain volume that only contain gt, remove irrelavant background
    Note: min_size set to cropped_size
    """
    row_col_page_b, new_size = trim_volume_square(gt, min_size=trim_params.MINSIZE, pad=trim_params.PAD)
    row_col_page_e = row_col_page_b + new_size
    # print row_col_page_b, new_size
    if im_type=="2D":
        im = im[row_col_page_b[0]:row_col_page_e[0], row_col_page_b[1]:row_col_page_e[1], :]
        gt = gt[row_col_page_b[0]:row_col_page_e[0], row_col_page_b[1]:row_col_page_e[1], :]
    else:
        im = im[row_col_page_b[0]:row_col_page_e[0], row_col_page_b[1]:row_col_page_e[1], row_col_page_b[2]:row_col_page_e[2]]
        gt = gt[row_col_page_b[0]:row_col_page_e[0], row_col_page_b[1]:row_col_page_e[1], row_col_page_b[2]:row_col_page_e[2]]
    """ Rotation
    """
    if (rotation != 0) and (np.random.randint(2) == 0):
        im = rotate(im, rotation, reshape=False, cval=np.min(im))
        gt = rotate(gt, rotation, reshape=False, cval=0)
    """ Horizontal Flip
    """
    if use_flip and (np.random.randint(2) == 0):
        im = im[:,::-1,:]
        gt = gt[:,::-1,:]
    """ Apply Hounsfield Unit Window
    The Hounsfield unit values will be windowed in the range
    HU_WINDOW to exclude irrelevant organs and objects.
    """
    im = hounsfield_unit_window(im, hu_window=hu_window)
    """ Zero-center and Normalization
    """
    im -= pixel_statistics[0]
    im /= pixel_statistics[1]
    #im = normalizer(im, src_range=hu_window, dst_range=data_range) 
    """ Scale image
    """
    if adjacnet and im_type=="2D":
        im_scaled = np.zeros((target_size, target_size, im.shape[2]), dtype=np.float32)
        gt_scaled = np.zeros((target_size, target_size, gt.shape[2]), dtype=np.float32)
        for ind in xrange(im.shape[2]):
            im_scaled[:,:,ind] = scale_image(im[:,:,ind], target_size=target_size, max_size=max_size, im_type=im_type)
            gt_scaled[:,:,ind] = scale_image(gt[:,:,ind], target_size=target_size, max_size=max_size, im_type=im_type)
        im = im_scaled
        gt = gt_scaled
    else:
        im = scale_image(im, target_size=target_size, max_size=max_size, im_type=im_type)
        gt = scale_image(gt, target_size=target_size, max_size=max_size, im_type=im_type)
    """ Random Crop patch from image
    """
    cropped_patches = random_crop_patches([im, gt], cropped_size, patch_per_image)

    return cropped_patches[0], cropped_patches[1]

def prepare_adj_data(im_path):
    '''Load adjacent slice
    '''
    ### Get adjacent slices path
    im_dir, im_file = osp.split(im_path)
    im_name, im_ext = osp.splitext(im_file)
    im_prefix, im_index = im_name.split('_slice_')
    adj_l2_im_path = osp.join(im_dir, '{}_slice_{}{}'.format(im_prefix, int(im_index) - 2, im_ext))
    adj_l1_im_path = osp.join(im_dir, '{}_slice_{}{}'.format(im_prefix, int(im_index) - 1, im_ext))
    adj_r1_im_path = osp.join(im_dir, '{}_slice_{}{}'.format(im_prefix, int(im_index) + 1, im_ext))
    adj_r2_im_path = osp.join(im_dir, '{}_slice_{}{}'.format(im_prefix, int(im_index) + 2, im_ext))
    ''' Check adjacent im slices
    '''
    #if l1 not exists then l1 and l2 both set to im_path
    #if l1 exists and l2 is not, then set l2 to l1_path
    if not osp.exists(adj_l1_im_path):
        adj_l1_im_path = im_path
        adj_l2_im_path = im_path
    elif not osp.exists(adj_l2_im_path):
        adj_l2_im_path = adj_l1_im_path
    #if r1 not exists then r1 and r2 both set to im_path
    #if r1 exists and r2 is not, then set r2 to r1_path
    if not osp.exists(adj_r1_im_path):
        adj_r1_im_path = im_path
        adj_r2_im_path = im_path
    elif not osp.exists(adj_r2_im_path):
        adj_r2_im_path = adj_r1_im_path
    ''' Load adjacent im slices
    '''
    adj_l2_im_metadata = load_data(adj_l2_im_path, flags=0)
    adj_l1_im_metadata = load_data(adj_l1_im_path, flags=0)
    adj_r1_im_metadata = load_data(adj_r1_im_path, flags=0)
    adj_r2_im_metadata = load_data(adj_r2_im_path, flags=0)
    adj_l2_im_data = adj_l2_im_metadata['image_data'][:,:,0]
    adj_l1_im_data = adj_l1_im_metadata['image_data'][:,:,0]
    adj_r1_im_data = adj_r1_im_metadata['image_data'][:,:,0]
    adj_r2_im_data = adj_r2_im_metadata['image_data'][:,:,0]

    return (adj_l2_im_data, adj_l1_im_data, adj_r1_im_data, adj_r2_im_data)

def ims_vis(ims):
    '''Function to display row of images'''
    fig, axes = plt.subplots(1, len(ims))
    for i, im in enumerate(ims):
        axes[i].imshow(im, cmap='gray', origin='upper')
    plt.show()

def vis_seg(axes, ims):
    '''Function to display row of images'''
    if len(axes.shape) < 2:
        axes = axes[np.newaxis,:]
    for i, im in enumerate(ims):
        row = i // axes.shape[1]
        col = i % axes.shape[1]
        axes[row, col].imshow(im, cmap='gray', origin='upper')
    plt.show()
    plt.pause(0.00001)
