#!/usr/bin/env python
# encoding: utf-8


import os
import os.path as osp
import numpy as np
# import nibabel as nib
from lits.config import cfg

class litsdb3d(object):
    def __init__(self, image_set, training_batch):
        self._name = 'lits_' + training_batch + '_' + image_set
        self._image_set = image_set

        # cfg.DATA_DIR, 'lits', training
        self._data_path = osp.join(cfg.DATA_DIR, 'lits', training_batch)
        self._image_folder = self._data_path
        self._gt_folder = self._data_path
        
        assert osp.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)
        assert osp.exists(self._image_folder), 'Path does not exist: {}'.format(self._image_folder)
        assert osp.exists(self._gt_folder), 'Path does not exist: {}'.format(self._gt_folder)

        # load images' path, gts' path
        if self._name.split('_')[1] == 'Test':
            self._images_path, self._gts_path = self.load_path_test()
        else:
            self._images_path, self._gts_path = self.load_path()

    def load_path_test(self):
        """
        Load the Images' Path.
        """
        # self._gt_folder = None
        self._gt_folder = osp.join(cfg.OUTPUT_DIR, 'unet/unet_2d_bn_c2/lits_Test_Batch_trainval_3D', 'unet_2d_bn_c2_liver_200000LCC_280000LCC_CB_LCC')

        images_path = []
        gts_path = []
        volumes = self.load_volumes_test(folder=self._image_folder)
        if self._gt_folder is None:
            segmentations = volumes
            self._gt_folder = self._image_folder
        else:
            segmentations = self.load_segmentations_test(folder=self._gt_folder)

        for i in xrange(len(volumes)):
            im_path = osp.join(self._image_folder, volumes[i])
            gt_path = osp.join(self._gt_folder, segmentations[i])
            if osp.splitext(osp.split(im_path)[1])[0].split('-')[2] == '40':
                continue
            if gt_path is not None:
                im_index = osp.splitext(osp.split(im_path)[1])[0].split('-')[2]
                gt_index = osp.splitext(osp.split(gt_path)[1])[0].split('-')[2]
                assert im_index == gt_index, 'im_index:{} and gt_index:{} mismatch'.format(im_index, gt_index)
            images_path.append(im_path)
            gts_path.append(gt_path)

        assert len(images_path)==len(gts_path), 'the number of images must equal to the number of segmentations'

        return images_path, gts_path

    def load_path(self):
        """
        Load the Images' Path.
        """
        images_path = []
        gts_path = []
        volumes = self.load_volumes(folder=self._image_folder)
        segmentations = self.load_segmentations(folder=self._gt_folder)
        for i in xrange(len(volumes)):
            im_path = osp.join(self._image_folder, volumes[i])
            gt_path = osp.join(self._gt_folder, segmentations[i])
            if gt_path is not None:
                im_index = osp.splitext(osp.split(im_path)[1])[0].split('-')[1]
                gt_index = osp.splitext(osp.split(gt_path)[1])[0].split('-')[1]
                assert im_index == gt_index, 'im_index:{} and gt_index:{} mismatch'.format(im_index, gt_index)
            images_path.append(im_path)
            gts_path.append(gt_path)

        # Divide dataset to the training set and validation set
        training_set_num = 100
        if self._image_set == 'val':
            images_path = images_path[training_set_num:]
            gts_path = gts_path[training_set_num:]
        else:
            images_path = images_path[0:training_set_num]
            gts_path = gts_path[0:training_set_num]

        assert len(images_path)==len(gts_path), 'the number of images must equal to the number of segmentations'

        return images_path, gts_path

    def load_volumes(self, folder):
        files = self.load_files_from_folder(folder, prefix='volume-', suffix='.nii')
        files.sort(key=lambda name: int(name.split('-')[1].split('.')[0]))
        return files

    def load_segmentations(self, folder):
        files = self.load_files_from_folder(folder, prefix='segmentation-', suffix='.nii')
        files.sort(key=lambda name: int(name.split('-')[1].split('.')[0]))
        return files
    def load_volumes_test(self, folder):
        files = self.load_files_from_folder(folder, prefix='test-volume-', suffix='.nii')
        files.sort(key=lambda name: int(name.split('-')[2].split('.')[0]))
        return files
    def load_segmentations_test(self, folder):
        files = self.load_files_from_folder(folder, prefix='test-segmentation-', suffix='.nii')
        files.sort(key=lambda name: int(name.split('-')[2].split('.')[0]))
        return files

    def load_files_from_folder(self, folder, prefix, suffix):
        ''' Load files from file_folder startswith prefix and endswith suffix
        Return a list of files name
        '''
        files = [f for f in os.listdir(folder) if (f.startswith(prefix) and f.endswith(suffix))]
        
        return files

    def name(self):
        return self._name

    def images_path(self):
        return self._images_path

    def gts_path(self):
        return self._gts_path