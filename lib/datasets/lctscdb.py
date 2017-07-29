#!/usr/bin/env python
# encoding: utf-8


import os
import os.path as osp
import numpy as np
# import nibabel as nib
from lits.config import cfg

class lctscdb(object):
    def __init__(self, image_set, training_batch):
        self._name = 'lctsc_' + training_batch + '_' + image_set
        self._image_set = image_set
        ''' 
        1 Esophagus
        2 Heart
        3 SpinalCord
        4 Lung_L
        5 Lung_R
        '''

        # cfg.DATA_DIR, 'lits', training
        self._data_path = osp.join(cfg.DATA_DIR, 'lctsc_'+training_batch)
        #self._data_path = osp.join(cfg.DATA_DIR, 'lctsc')
        self._image_folder = self._data_path
        self._gt_folder = self._data_path

        assert osp.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)
        assert osp.exists(self._image_folder), 'Path does not exist: {}'.format(self._image_folder)
        assert osp.exists(self._gt_folder), 'Path does not exist: {}'.format(self._gt_folder)

        # load images' path, gts' path
        self._images_path, self._gts_path = self.load_path()

    def load_path(self):
        """
        Load the Images' Path.
        """
        images_path = []
        gts_path = []
        # volumes = self.load_volumes(folder=self._image_folder)
        segmentations = self.load_segmentations(folder=self._gt_folder)

        for i in xrange(len(segmentations)):
            im_name = segmentations[i].split('_gt')[0] + '.nii'
            gt_name = segmentations[i]

            im_path = osp.join(self._image_folder, im_name)
            gt_path = osp.join(self._gt_folder, gt_name)
            print im_path, gt_path
            # gt = nib.load(gt_path)
            # gt = gt.get_data()
            # gt_nonzero = np.flatnonzero((np.sum(gt, axis=(0,1))))
            # if gt_nonzero.size == 0:
            #     print gt_path
            #     continue
            # else:
            #     print (gt_nonzero[-1] - gt_nonzero[0])
            images_path.append(im_path)
            gts_path.append(gt_path)

        # # Divide dataset to the training set and validation set
        # training_set_num = 100
        # if self._image_set == 'val':
        #     images_path = images_path[training_set_num:]
        #     gts_path = gts_path[training_set_num:]
        # else:
        #     images_path = images_path[0:training_set_num]
        #     gts_path = gts_path[0:training_set_num]

        return images_path, gts_path

    # def load_volumes(self, folder):
    #     files = self.load_files_from_folder(folder, prefix='LCTSC-Train-S', suffix='.nii')
    #     files.sort(key=lambda name: int(name.split('-')[1].split('.')[0]))
    #     return files

    def load_segmentations(self, folder):
        files = self.load_files_from_folder(folder, prefix='LCTSC-Train-S', suffix='_gt.nii')
        # files.sort(key=lambda name: int(name.split('-')[1].split('.')[0]))
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