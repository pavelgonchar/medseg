#!/usr/bin/env python
# encoding: utf-8
'''
Train a CNN network.
or 
Test a CNN network
'''

import _init_paths
import argparse
import pprint
import numpy as np
import sys
import os.path as osp
from easydict import EasyDict as edict
from lits.config import cfg, set_solver_prototxt
import CaffeWrapper as CW
from lits.EvalWrapper import eval
import os

def parse_args():
	'''
	Parse input arguments
	'''
	# create the top-level parser
	parser = argparse.ArgumentParser(prog='PROG')
	subparsers = parser.add_subparsers(title='Network Option',
						description='Train or Test',
						help='additional help',
						dest='subparser_name')

	# create the parser for the "train" command
	parse_train = subparsers.add_parser('train', help='train help')
	parse_train.add_argument('--gpu', dest='gpu_id',
						help='GPU device id to use [0]',
						default=0, type=int)
	parse_train.add_argument('--solver', dest='solver',
						help='solver prototxt',
						default=None, type=str)
	parse_train.add_argument('--iters', dest='max_iters',
						help='number of iterations to train',
						default=0, type=int)
	parse_train.add_argument('--weights', dest='pretrained_model',
						help='initialize with pretrained model weights',
						default=None, type=str)
	parse_train.add_argument('--imdb', dest='imdb_name',
						help='dataset to train on',
						default=None, type=str)
	parse_train.add_argument('--rand', dest='randomize',
						help='randomize (do not use a fixed seed)',
						action='store_true')

	# create the parser for the "test" command
	parse_test = subparsers.add_parser('test', help='test help')
	parse_test.add_argument('--gpu', dest='gpu_id',
						help='GPU device id to use [0]',
						default=0, type=int)
	parse_test.add_argument('--def', dest='prototxt',
						help='prototxt file defining the network',
						default=None, type=str)
	parse_test.add_argument('--net', dest='caffemodel',
						help='model to test',
						default=None, type=str)
	parse_test.add_argument('--imdb', dest='imdb_name',
						help='dataset to test',
						default=None, type=str)

	# create the parser for the "eval" command
	parse_eval = subparsers.add_parser('eval', help='eval help')
	parse_eval.add_argument('--gpu', dest='gpu_id',
						help='GPU device id to use [0]',
						default=0, type=int)

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	
	args = parser.parse_args()
	return args

###### ###### ###### ###### ###### ######
''' Overwrite the Default
''' 
# GPU_ID == -1 stands for only CPU
cfg.GPU_ID = 0
cfg.PID = os.getpid()
''' Liver
'''
# 1
# cfg.EXP_DIR = 'unet/unet_2d_bn_c2'
# snapshot_prefix = 'unet_2d_bn_c2_multiloss'
# 2
# cfg.EXP_DIR = 'unet/unet_2d_bn_c2'
# snapshot_prefix = 'unet_2d_bn_c2_liver'
# 3
# cfg.EXP_DIR = 'uvnet/uvnet_2d_bn_incept2_weigted_c2'
# snapshot_prefix = 'uvnet_2d_bn_incept2_liver_c2'
# 4
# cfg.EXP_DIR = 'uvnet/uvnet_2d_bn_incept2_weigted_c2'
# snapshot_prefix = 'uvnet_2d_bn_incept2_liver_c2_416full'

''' Lesion
'''
# 1 (1.1.5)
# cfg.EXP_DIR = 'unet/unet_2d_bn_weigted_c3'
# snapshot_prefix = 'unet_2d_bn_weigted_c3_zoom'
# 2
# cfg.EXP_DIR = 'unet/unet_2d_bn_weigted_c3'
# snapshot_prefix = 'unet_2d_bn_weigted_c3_1.1.10'
# 3
# cfg.EXP_DIR = 'uvnet/uvnet_2d_bn_c3'
# snapshot_prefix = 'uvnet_2d_bn_c3_zoom_refined'
# 4
# cfg.EXP_DIR = 'uvnet/uvnet_2d_bn_weigted_c3'
# snapshot_prefix = 'uvnet_2d_bn_weigted_1.1.7'
# 5
# cfg.EXP_DIR = 'uvnet/uvnet_2d_bn_modified_weigted_c3'
# snapshot_prefix = 'uvnet_2d_bn_modified_weigted_c3_1.1.7'
# cfg.EXP_DIR = 'uvnet/uvnet_2d_bn_modified_weigted_c3'
# snapshot_prefix = 'uvnet_2d_bn_modified_weigted_c3_1.1.7_refined'
# 6
# cfg.EXP_DIR = 'uvnet/uvnet_2d_bn_original_weigted_c3'
# snapshot_prefix = 'uvnet_2d_bn_original_weigted_c3_1.1.7_refined'
# 7
cfg.EXP_DIR = 'uvnet/uvnet_2d_bn_incept_weigted_c3'
snapshot_prefix = 'uvnet_2d_bn_incept_weigted_c3_1.1.10'
# 8
# cfg.EXP_DIR = 'uvnet/uvnet_2d_bn_incept2_weigted_c3'
# snapshot_prefix = 'uvnet_2d_bn_incept2_weigted_c3_1.1.10'
###### ###### ###### ###### ###### ######
''' 
Train
'''
cfg.TRAIN.SEGMENTATION_MODE = '2D'
cfg.TRAIN.HU_WINDOW = [-200, 300]
cfg.TRAIN.DATA_RANGE= [0, 1]
cfg.TRAIN.PIXEL_STATISTICS =(-93.59, 131.86)
cfg.TRAIN.SCALES = (416,)
cfg.TRAIN.USE_FLIP = True
cfg.TRAIN.ROTATIONS = (0,1,2,3) # Number of times the array is rotated by 90 degrees

cfg.TRAIN.CROPPED_SIZE = (416,416,5)

cfg.TRAIN.ADJACENT = True
cfg.TRAIN.DEBUG = False

cfg.TRAIN.BG = edict()
cfg.TRAIN.BG.CLEAN = False
cfg.TRAIN.BG.BRIGHT = False

cfg.TRAIN.TRIM = edict()
cfg.TRAIN.TRIM.MINSIZE = [64,64,5]
cfg.TRAIN.TRIM.PAD = [32, 32, 0]

cfg.TRAIN.CLASS = edict()
cfg.TRAIN.CLASS.USE_WEIGHT = True
cfg.TRAIN.CLASS.WEIGHT = [1, 1, 10]
cfg.TRAIN.CLASS.NUMBER = 3
cfg.TRAIN.CLASS.SPLIT = (0.5, 1.5)
#cfg.TRAIN.CLASS.NUMBER = 2
#cfg.TRAIN.CLASS.SPLIT = (0.5,)

cfg.TRAIN.IMS_PER_BATCH = 2
cfg.TRAIN.BATCH_SIZE = 2
cfg.TRAIN.SNAPSHOT_ITERS = 20000
cfg.TRAIN.MAX_ITER = 300000
cfg.TRAIN.USE_PREFETCH = False
cfg.TRAIN.DISPLAY_INTERVAL = 100
cfg.TRAIN.SOLVER = None
cfg.TRAIN.PROTOTXT = osp.abspath(osp.join(cfg.MODELS_DIR, cfg.EXP_DIR, '{}'.format('train.prototxt')))
cfg.TRAIN.PRETRAINED_MODEL = '{}'.format('/home/zlp/dev/medseg/output/uvnet/uvnet_2d_bn_incept_weigted_c3/lits_Training_Batch_trainval_2D/uvnet_2d_bn_incept_weigted_c3_1.1.10_iter_80000.caffemodel')
cfg.TRAIN.IMDB_NAME = 'lits_Training_Batch_trainval_2D'
cfg.TRAIN.NUM_PROCESS = 6 #the number of threads to do data augmentation
###### ###### ###### ###### ###### ######
''' SOLVER PARAMETER Setting
'''
SOLVER_PARAMETER = edict()
SOLVER_PARAMETER.NET = cfg.TRAIN.PROTOTXT
SOLVER_PARAMETER.BASE_LR = 0.0001
SOLVER_PARAMETER.MOMENTUM = 0.99
SOLVER_PARAMETER.WEIGHT_DECAY = 0.0005
SOLVER_PARAMETER.LR_POLICY = "step"
SOLVER_PARAMETER.STEPSIZE = 100000
SOLVER_PARAMETER.GAMMA = 0.1
SOLVER_PARAMETER.DISPLAY_INTERVAL = cfg.TRAIN.DISPLAY_INTERVAL
SOLVER_PARAMETER.SNAPSHOT = 0  #We disable standard caffe solver snapshotting and implement our own snapshot
SOLVER_PARAMETER.SNAPSHOT_PREFIX = "{}".format(snapshot_prefix)
###### ###### ###### ###### ###### ######
''' 
Test
'''
cfg.TEST.SEGMENTATION_MODE = '2D'
cfg.TEST.DEBUG = False
cfg.TEST.FAST_INFERENCE = True # IF IMAGES HAVE GTS, THEN PERFORM FAST INFERENCE
cfg.TEST.APPLY_MASK = True
cfg.TEST.ADJACENT = True
cfg.TEST.CLASS_NUM = 3
cfg.TEST.TRIM = edict()
cfg.TEST.TRIM.MINSIZE = [64, 64, 1]
cfg.TEST.TRIM.PAD = [32, 32, 0]

cfg.TEST.BG = edict()
cfg.TEST.BG.CLEAN = False
cfg.TEST.BG.BRIGHT = False

cfg.TEST.HU_WINDOW = [-200, 300]
cfg.TEST.DATA_RANGE= [0, 1]
cfg.TEST.PIXEL_STATISTICS = (-93.59, 131.86)
cfg.TEST.SCALES = (416,) # for zoom
cfg.TEST.CHUNK_SHAPE = (416,416,1)
cfg.TEST.STRIDE = (400,400,1)
cfg.TEST.MAX_SIZE = 720
cfg.TEST.PROTOTXT = osp.abspath(osp.join(cfg.MODELS_DIR, cfg.EXP_DIR, '{}'.format('test.prototxt')))
cfg.TEST.CAFFEMODEL = osp.join(cfg.OUTPUT_DIR, cfg.EXP_DIR, 'lits_Training_Batch_trainval_2D', '{}_iter_{}.caffemodel'.format(snapshot_prefix, 120000))
cfg.TEST.IMDB_NAME = 'lits_Training_Batch_val_3D'
# cfg.TEST.IMDB_NAME = 'lits_Test_Batch_trainval_3D'
cfg.TEST.NUM_PROCESS = 6#the number of threads to do data augmentation
cfg.TEST.MODE = 'TESTEVAL' # EVAL OR TEST OR TESTEVAL

'''
Evaluation
'''
cfg.EVAL = edict()
cfg.EVAL.GT_DIR = '{}'.format('/home/zlp/dev/medseg/data/lits/Training_Batch')
cfg.EVAL.PRED_DIR = '{}'.format('/home/zlp/dev/medseg/output/uvnet/uvnet_2d_bn_modified_weigted_c3/lits_Training_Batch_val_3D/uvnet_2d_bn_modified_weigted_c3_1.1.7_refined_iter_280000/label')
cfg.EVAL.OUT_DIR = '{}'.format('/home/zlp/dev/medseg/output/uvnet/uvnet_2d_bn_modified_weigted_c3/lits_Training_Batch_val_3D/uvnet_2d_bn_modified_weigted_c3_1.1.7_refined_iter_280000')
cfg.EVAL.NUM_PROCESS = 12 #the number of threads to do data augmentation


if __name__ == '__main__':
	
	# sys.argv.extend(['train', '--gpu=0'])
	# sys.argv.extend(['test', '--gpu=0'])
	# sys.argv.extend(['eval', '--gpu=0'])
	args = parse_args()
	print('Called with args:')
	print(args)

	if args.subparser_name  == 'train':
		cfg.GPU_ID = args.gpu_id
		#cfg.GPU_ID = 1
		if args.max_iters > 0:
			cfg.TRAIN.MAX_ITER = args.max_iters
		if args.pretrained_model is not None:
			cfg.TRAIN.PRETRAINED_MODEL = args.pretrained_model
		if args.imdb_name is not None:
			cfg.TRAIN.IMDB_NAME = args.imdb_name
		if args.solver is not None:
			cfg.TRAIN.SOLVER = args.solver
		if not args.randomize:
			# if randomize in off then fix the random seeds (numpy and caffe) for reproducibility
			cfg.RNG_SEED = 777

		# Write a solver text file if is not provided
		if cfg.TRAIN.SOLVER is None:
			cfg.TRAIN.SOLVER = osp.join(osp.split(SOLVER_PARAMETER.NET)[0], 'solver.prototxt')
			set_solver_prototxt(SOLVER_PARAMETER, cfg.TRAIN.SOLVER)
		# delete cfg['TEST'] for better view
		del cfg['TEST']
		del cfg['EVAL']

	elif args.subparser_name == 'test':
		cfg.GPU_ID = args.gpu_id
		if args.prototxt is not None:
			cfg.TEST.PROTOTXT = args.prototxt
		if args.caffemodel is not None:
			cfg.TEST.CAFFEMODEL = args.caffemodel
		if args.imdb_name is not None:
			cfg.TEST.IMDB_NAME = args.imdb_name
		# delete cfg['TRAIN'] for better view
		del cfg['TRAIN']
		del cfg['EVAL']

	elif args.subparser_name == 'eval':
		cfg.GPU_ID = args.gpu_id
		del cfg['TRAIN']
		del cfg['TEST']

	else:
		print 'subparser_name error'
		exit()

	print('====================\n')
	print('Using config:')
	pprint.pprint(cfg)

	'''
	CaffeWrapper
	'''
	caffe_wrapper = CW.CaffeWrapper()
	if args.subparser_name  == 'train':
		caffe_wrapper.train()
	elif args.subparser_name  == 'test':
		caffe_wrapper.test()
	else:
		eval(cfg.EVAL)









