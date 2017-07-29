#!/usr/bin/env python
# encoding: utf-8


import os
import os.path as osp
import numpy as np
from lits.config import cfg


class litsdb(object):
    def __init__(self, image_set, training_batch):
        self._name = 'lits_' + training_batch + '_' + image_set
        self._image_set = image_set

        # cfg.DATA_DIR, 'lits', training
        self._data_path = osp.join(cfg.DATA_DIR, 'lits', training_batch)

        self._image_folder = osp.join(self._data_path, 'Images')
        self._gt_folder = osp.join(self._data_path, 'Segmentation')
        self._image_set_folder = osp.join(self._data_path, 'ImageSets')
        assert osp.exists(self._data_path), 'Path does not exist: {}'.format(self._data_path)
        assert osp.exists(self._image_folder), 'Path does not exist: {}'.format(self._image_folder)
        assert osp.exists(self._gt_folder), 'Path does not exist: {}'.format(self._gt_folder)

        # load image_path, gt_path, label1count, label2count
        self._image_paths, self._gt_paths,self._label1count, self._label2count = self._load_image_paths()
        self.pos_neg(0.1)
        #self.pos_neg(1)

    def _load_image_paths(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        self._image_set_file = osp.join(self._image_set_folder, self._image_set + '.txt')
        assert osp.exists(self._image_set_file), 'Path does not exist: {}'.format(self._image_set_file)

        image_paths, gt_paths, label1count, label2count = self._load_image_paths_from_imagesets()
        print 'Load image set index from imageset file: {}'.format(self._image_set)

        return image_paths, gt_paths, label1count, label2count

    # Change _load_image_set_index to fit other dataset
    def _load_image_paths_from_imagesets(self):
        """
        Load the indexes listed from imagesets
        imagesets format: image_slice_name gt_slice_name label1count label2count
        """
        image_paths = list()
        gt_paths = list()
        label1count = list()
        label2count = list()
        with open(self._image_set_file, 'r') as f:
            for line in f.readlines():
                if line.startswith('#') or len(line)==0:
                    continue

                line = line.strip('\n')
                line = line.strip().split(" ")
                # if 'volume-83' == line[0].split("_slice_")[0]:
                #     print line[0].split("_slice_")[0]
                #     continue

                image_paths.append(osp.join(self._image_folder, line[0].split("_slice_")[0], line[0]))
                if self._image_set == 'test':
                    gt_paths.append(None)
                    label1count.append(None)
                    label2count.append(None)
                else:
                    gt_paths.append(osp.join(self._gt_folder, line[1].split("_slice_")[0], line[1]))
                    label1count.append(float(line[2]))
                    label2count.append(float(line[3]))

        return image_paths, gt_paths, label1count, label2count

    def name(self):
        return self._name

    def images_path(self):
        return self._image_paths

    def gts_path(self):
        return self._gt_paths

    def pos_neg(self, neg_pos_ratio=None):
        '''separate the postive and negative samples'''
        ind_pos_1 = np.array(self._label1count) > 0.5
        ind_pos_2 = np.array(self._label2count) > 4
        ind_neg = np.array(self._label1count) < 0.5
        num_pos_1 = np.sum(ind_pos_1)
        num_pos_2 = np.sum(ind_pos_2)
        num_neg = np.sum(ind_neg)
        if neg_pos_ratio is not None:
            # num_neg = np.min([int(np.ceil(num_pos_1 * neg_pos_ratio)), num_neg])
            num_neg = np.min([int(np.ceil(num_pos_2 * neg_pos_ratio)), num_neg])

        img_pos_1 = np.array(self._image_paths)[ind_pos_1]
        img_pos_2 = np.array(self._image_paths)[ind_pos_2]
        img_neg = np.array(self._image_paths)[ind_neg]
        gt_pos_1 = np.array(self._gt_paths)[ind_pos_1]
        gt_pos_2 = np.array(self._gt_paths)[ind_pos_2]
        gt_neg = np.array(self._gt_paths)[ind_neg]

        # random select negtive samples
        perm = np.random.permutation(np.arange(len(img_neg)))[0:num_neg]
        img_neg = img_neg[perm]
        gt_neg = gt_neg[perm]

        image_paths = list()
        gt_paths = list()
        # image_paths.extend(img_pos_1)
        image_paths.extend(img_pos_2)
        image_paths.extend(img_neg)
        # gt_paths.extend(gt_pos_1)
        gt_paths.extend(gt_pos_2)
        gt_paths.extend(gt_neg)

        # Augment label2 samples
        augment_pos_2_times = 0
        if augment_pos_2_times > 0:
            print 'Before Augment label2, image size = {}'.format(len(image_paths))
            for t in xrange(augment_pos_2_times):
                image_paths.extend(img_pos_2)
                gt_paths.extend(gt_pos_2)
            print 'After Augment label2 {} times, image size = {}'.format(augment_pos_2_times, len(image_paths))

        # Update 
        self._image_paths = image_paths
        self._gt_paths = gt_paths
        
        # Calculate the Pixel Percentage
        percentage_pos_1 = np.sum(np.array(self._label1count)[ind_pos_1] / (512.0*512.0))
        percentage_pos_2 = np.sum(np.array(self._label2count)[ind_pos_2] / (512.0*512.0))
        if augment_pos_2_times > 0:
            percentage_pos_1 += np.sum(np.array(self._label1count)[pos_2_ind] / (512.0*512.0)) * augment_pos_2_times
            percentage_pos_2 += np.sum(np.array(self._label1count)[pos_2_ind] / (512.0*512.0)) * augment_pos_2_times

        percentage_pos_1 /= len(self._image_paths)
        percentage_pos_2 /= len(self._image_paths)
        percentage_neg = 1 - percentage_pos_1 - percentage_pos_2
        # Calculate the class weight
        weight_pos_1 = 1.0 / (percentage_pos_1 + cfg.EPS)
        weight_pos_2 = 1.0 / (percentage_pos_2 + cfg.EPS)
        weight_neg = 1.0 / (percentage_neg + cfg.EPS)
        # normalize to have sum = 1
        normalizer = weight_pos_1 + weight_pos_2 + weight_neg
        weight_pos_1 /= normalizer
        weight_pos_2 /= normalizer
        weight_neg /= normalizer
        # Update CLASS_WEIGHT
        use_online_weight = 0
        if use_online_weight:
            cfg.TRAIN.CLASS_WEIGHT = (weight_neg, weight_pos_1, weight_pos_2)

        print 'Label0:{}, Label1:{}, Label2:{}'.format(weight_neg, weight_pos_1, weight_pos_2)
