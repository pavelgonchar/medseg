#!/usr/bin/env python
# encoding: utf-8


"""
CNN config system.
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
# __C.PIXEL_MEANS = 0

# For reproducibility
__C.RNG_SEED = None

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Model directory
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models'))

# Output Directory
__C.OUTPUT_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'output'))

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'


# Default GPU device id
__C.GPU_ID = 0

# Default Segmentation '2D' or '3D'
__C.SEGMENTATION_MODE = None


#
# Training options
#

__C.TRAIN = edict()

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (512,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 720

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIP = False

# rotations when use clockwise-rotated images during training
__C.TRAIN.ROTATIONS = (0,)

# Crop Size (height, width, depth)
__C.TRAIN.CROPPED_SIZE = (512, 512, 512)

# Class weight to tackle imbalance data
__C.TRAIN.CLASS_WEIGHT = (1,1,1)

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = None

# Maximum Iterations
__C.TRAIN.MAX_ITER = None

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

# Plot Curve interval
__C.TRAIN.DISPLAY_INTERVAL = 100

# PRETRAINED_MODEL
__C.TRAIN.PRETRAINED_MODEL = None

# Solver file
__C.TRAIN.SOLVER = None


#
# Testing options
#

__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (512,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 512


def get_output_dir(imdb_name, caffemodel_name=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a caffemodel name
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.OUTPUT_DIR, __C.EXP_DIR, imdb_name))
    if caffemodel_name is not None:
        outdir = osp.join(outdir, caffemodel_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def set_solver_prototxt(solver_para, wpath):
    '''
    SOLVER_PARAMETER = edict()
    SOLVER_PARAMETER.NET = osp.abspath(osp.join(cfg.MODELS_DIR, cfg.TRAIN.NETWORK))
    SOLVER_PARAMETER.TEST_ITER = 100
    SOLVER_PARAMETER.TEST_INTERVAL = 2000000 #We disable TEST during training

    SOLVER_PARAMETER.BASE_LR = 0.001
    SOLVER_PARAMETER.MOMENTUM = 0.99
    SOLVER_PARAMETER.WEIGHT_DECAY = 0.0005

    SOLVER_PARAMETER.LR_POLICY = "step"
    SOLVER_PARAMETER.STEPSIZE = 20000
    SOLVER_PARAMETER.GAMMA = 0.1
    SOLVER_PARAMETER.DISPLAY_INTERVAL = 100
    SOLVER_PARAMETER.MAX_ITER = 100000
    SOLVER_PARAMETER.SNAPSHOT = 0  #We disable standard caffe solver snapshotting and implement our own snapshot
    SOLVER_PARAMETER.SNAPSHOT_PREFIX = "VGG_train_c2_bn_1_10"
    SOLVER_PARAMETER.SOLVER_MODE = cfg.CAFFE_MODE
    '''
    with open(wpath, 'w') as f:
        f.write('# The train/test net protocol buffer definition\n')
        # f.write('net: "{}" \n'.format(solver_para.NET))
        f.write('train_net: "{}" \n'.format(solver_para.NET))

        # f.write('# test_iter specifies how many forward passes the test should carry out.\n')
        # f.write('# In the case of MNIST, we have test batch size 100 and 100 test iterations,\n')
        # f.write('# covering the full 10,000 testing images.\n')
        # f.write('test_iter: {} \n'.format(solver_para.TEST_ITER))

        # f.write('# Carry out testing every 100 training iterations.\n')
        # f.write('test_interval: {} \n'.format(solver_para.TEST_INTERVAL))

        f.write('# The base learning rate, momentum and the weight decay of the network.\n')
        f.write('base_lr: {} \n'.format(solver_para.BASE_LR))
        f.write('momentum: {} \n'.format(solver_para.MOMENTUM))
        f.write('weight_decay: {} \n'.format(solver_para.WEIGHT_DECAY))

        f.write('# The learning rate policy \n')
        f.write('lr_policy: "{}" \n'.format(solver_para.LR_POLICY))
        f.write('stepsize: {} \n'.format(solver_para.STEPSIZE))
        f.write('gamma: {} \n'.format(solver_para.GAMMA))

        f.write('# Display every 100 iterations\n')
        f.write('display: {} \n'.format(solver_para.DISPLAY_INTERVAL))

        # f.write('# The maximum number of iterations\n')
        # f.write('max_iter: {} \n'.format(solver_para.MAX_ITER))

        f.write('# snapshot intermediate results\n')
        f.write('# We disable standard caffe solver snapshotting and implement our own snapshot \n')
        f.write('snapshot: {} \n'.format(solver_para.SNAPSHOT))
        f.write('snapshot_prefix: "{}" \n'.format(solver_para.SNAPSHOT_PREFIX))
        
        # f.write('# solver mode: CPU or GPU\n')
        # f.write('solver_mode: {} \n'.format(solver_para.SOLVER_MODE))
        
    f.close()


