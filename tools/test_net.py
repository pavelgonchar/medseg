#!/usr/bin/env python
# encoding: utf-8


"""Test a CNN network on an image database."""

import _init_paths
import caffe
import argparse
import pprint
import time, os, sys
import os.path as osp

from lits.test import test_net
from lits.config import cfg, get_output_dir
from datasets.factory import get_imdb
from datasets.litsdb import prepare_test_imdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        #sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('=================================================================\n')
    print('Called with args:')
    print(args)

    cfg.EXP_DIR = 'deeplab/deeplab_largefov_C3'
    prototxt = 'deeplab/deeplab_largefov_C3/test.prototxt'
    caffemodel = 'lits_Training_Batch_trainval_2D/deeplab_largefov_C3_iter_15000.caffemodel'
    imdb_name = 'lits_Training_Batch_test_2D'
    # Overwrite the default
    cfg.CAFFE_MODE = 'GPU'
    cfg.SEGMENTATION_MODE = '2D'
    cfg.PIXEL_MEANS = 0.01
    cfg.TEST.SCALES = (512,)

    
    cfg.GPU_ID = args.gpu_id
    if not args.prototxt:
        args.prototxt = osp.abspath(osp.join(cfg.MODELS_DIR, prototxt))

    if not args.caffemodel:
        args.caffemodel = osp.join(cfg.OUTPUT_DIR, cfg.EXP_DIR, caffemodel)

    if not args.imdb_name:
        args.imdb_name = imdb_name

    print('Using config:')
    pprint.pprint(cfg)

    # setup caffe test
    if cfg.CAFFE_MODE == 'GPU':
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    else:
        caffe.set_mode_cpu()

    # prepare imdb list
    print('Prepare TEST Dataset:')
    imdb = get_imdb(args.imdb_name)
    print('Loaded dataset `{:s}`'.format(args.imdb_name))
    tedb = prepare_test_imdb(imdb.images_path(), imdb.gts_path())
    assert len(tedb) > 0, 'The size of Dataset must be > 0: {}'.format(len(tedb))
    print('Test data size: {:d}'.format(len(tedb)))
    print('Sample looks like: {:s}'.format(tedb[len(tedb)-1]))
    #exit(0)

    print('Load caffe Net:')
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    # setup output dir
    output_dir = get_output_dir(args.imdb_name, net)
    print('Output will be saved to `{:s}`'.format(output_dir))
    print('=================================================================\n')

    test_net(net, tedb, output_dir)