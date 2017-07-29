#!/usr/bin/env python
# encoding: utf-8

"""Train a CNN network."""

import _init_paths
import caffe
import argparse
import pprint
import numpy as np
import sys
import os.path as osp
from easydict import EasyDict as edict
from lits.train import train_net
from lits.config import cfg, get_output_dir, set_solver_prototxt
from datasets.factory import get_imdb
from datasets.litsdb import prepare_training_imdb, compute_image_mean
from datasets.litsdb3d import litsdb3d


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=0, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_training_imdb(imdb_names):
    images = list()
    gts = list()
    for name in imdb_names.split('+'):
        imdb = get_imdb(name)
        images.extend(imdb.images_path())
        gts.extend(imdb.gts_path())
        print('Loaded dataset: `{:s}`'.format(name))

    if cfg.CALCULATE_PIXEL_MEANS:
        cfg.PIXEL_MEANS = compute_image_mean(images)
    trdb = prepare_training_imdb(images, gts)

    assert len(trdb) > 0, 'The size of Dataset must be > 0: {}'.format(len(trdb))
    print('Training data size: {:d}'.format(len(trdb)))
    print('Sample looks like: {:s}'.format(trdb[len(trdb)-1]))
    
    return trdb

if __name__ == '__main__':
    ''' 
    Overwrite the default
    '''
    cfg.CAFFE_MODE = 'GPU'
    cfg.SEGMENTATION_MODE = '2D'
    cfg.TRAIN.USE_PREFETCH = True
    cfg.TRAIN.CLASS_WEIGHT = [1, 2, 3]
    cfg.TRAIN.MAX_ITER = 15000
    cfg.TRAIN.SNAPSHOT_ITERS = 5000
    cfg.TRAIN.CROPPED_SIZE = (321, 321, 3)
    cfg.TRAIN.USE_CROPPED = True
    cfg.TRAIN.USE_ROTATED = False
    cfg.TRAIN.IMS_PER_BATCH = 30
    cfg.TRAIN.BATCH_SIZE = 30
    cfg.TRAIN.PLOT_INTERVAL = 100
     # cfg.EXP_DIR = 'fcn32s_C2'
    cfg.EXP_DIR = 'deeplab/deeplab_largefov_C3'
    # pretrained_model
    # cfg.TRAIN.PRETRAINED_MODEL = '/home/zlp/dev/deeplab-public-ver2/deeplab_model/DeepLab-largeFOV/train2_iter_8000.caffemodel'
    cfg.TRAIN.PRETRAINED_MODEL = '/home/zlp/dev/deeplab-public-ver2/deeplab_model/DEEPLABV2_VGG-16/train2_iter_20000.caffemodel'
    ''' 
    SOLVER PARAMETER Setting
    '''
    SOLVER_PARAMETER = edict()
    # SOLVER_PARAMETER.NET = osp.abspath(osp.join(cfg.MODELS_DIR, 'UNet/train.prototxt'))
    # SOLVER_PARAMETER.NET = osp.abspath(osp.join(cfg.MODELS_DIR, 'UNet3DBN_C3/train.prototxt'))
    SOLVER_PARAMETER.NET = osp.abspath(osp.join(cfg.MODELS_DIR, '{}{}'.format(cfg.EXP_DIR,'/train.prototxt')))

    SOLVER_PARAMETER.BASE_LR = 0.001
    SOLVER_PARAMETER.MOMENTUM = 0.99
    SOLVER_PARAMETER.WEIGHT_DECAY = 0.0005

    SOLVER_PARAMETER.LR_POLICY = "step"
    SOLVER_PARAMETER.STEPSIZE = cfg.TRAIN.SNAPSHOT_ITERS
    SOLVER_PARAMETER.GAMMA = 0.1
    SOLVER_PARAMETER.DISPLAY_INTERVAL = cfg.TRAIN.PLOT_INTERVAL
    SOLVER_PARAMETER.SNAPSHOT = 0  #We disable standard caffe solver snapshotting and implement our own snapshot
    SOLVER_PARAMETER.SNAPSHOT_PREFIX = "{}{}".format('deeplab_largefov_C3','')
    '''
    Dataset
    '''
    IMDB_NAME = 'lits_{}_{}_{}'.format('Training_Batch', 'trainval', cfg.SEGMENTATION_MODE)
    #IMDB_NAME = 'lits_{}_{}_{}'.format('Training_Batch_256_256', 'train', cfg.SEGMENTATION_MODE)
    ''' 
    Args Overwrite the setting
    '''
    args = parse_args()
    print('\n=================================================================')
    print('Called with args:')
    print(args)
    print('====================\n')
    cfg.GPU_ID = args.gpu_id
    if args.max_iters > 0:
        cfg.TRAIN.MAX_ITER = args.max_iters
    if args.pretrained_model:
        cfg.TRAIN.PRETRAINED_MODEL = args.pretrained_model
    if args.imdb_name:
        IMDB_NAME = args.imdb_name
    if args.solver:
        cfg.TRAIN.SOLVER = args.solver
    else:
        # Write a temporary solver text file because pycaffe is stupid
        cfg.TRAIN.SOLVER = osp.join(osp.split(SOLVER_PARAMETER.NET)[0], 'solver.prototxt')
        set_solver_prototxt(SOLVER_PARAMETER, cfg.TRAIN.SOLVER)
    '''
    Set up caffe
    '''
    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)
    if cfg.CAFFE_MODE == 'GPU':
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)
    else:
        caffe.set_mode_cpu()
    '''
    Set up training dataset
    '''
    trdb = combined_training_imdb(IMDB_NAME)
    print('====================\n')
    print('Using config:')
    pprint.pprint(cfg)

    # set up caffe model output directory
    output_dir = get_output_dir(IMDB_NAME)
    print('Output will be saved to `{:s}`'.format(output_dir))
    print('=================================================================\n')

    # set up solver
    train_net(cfg.TRAIN.SOLVER, trdb, output_dir, pretrained_model=cfg.TRAIN.PRETRAINED_MODEL, max_iters=cfg.TRAIN.MAX_ITER)
