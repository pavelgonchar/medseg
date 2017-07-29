import caffe
import numpy as np
import os
import os.path as osp
from lits.TrainWrapper import train_net
from lits.TestWrapper import test_net
from lits.config import cfg, get_output_dir
from datasets.data_utils import training_imdb, test_imdb


class CaffeWrapper(object):
	"""A simple wrapper around Caffe.
	This wrapper gives us control over the training or test process.
	"""

	def __init__(self):
		if cfg.GPU_ID < 0:
			caffe.set_mode_cpu()
		else:
			caffe.set_mode_gpu()
			caffe.set_device(cfg.GPU_ID)

	def train(self):
		''' Parameters
		'''
		solver = cfg.TRAIN.SOLVER
		pretrained_model = cfg.TRAIN.PRETRAINED_MODEL
		max_iters = cfg.TRAIN.MAX_ITER
		imdb_name = cfg.TRAIN.IMDB_NAME
		'''
		Fix the random seeds (numpy and caffe) for reproducibility
		'''
		if cfg.RNG_SEED > 0:
			np.random.seed(cfg.RNG_SEED)
			# caffe.set_random_seed(cfg.RNG_SEED)
		''' 
		Setup training set
		'''
		trdb = training_imdb(imdb_name)
		assert len(trdb) > 0, 'The size of Dataset must be > 0: {}'.format(len(trdb))
		print('Training data size: {:d}'.format(len(trdb)))
		print('Sample looks like: {:s}'.format(trdb[len(trdb)-1]))
		'''
		Setup caffe model output directory
		'''
		output_dir = get_output_dir(imdb_name)
		print('Output will be saved to `{:s}`'.format(output_dir))
		print('=================================================================\n')
		'''
		Setup solver
		'''
		# train_net(solver, trdb, output_dir, pretrained_model=pretrained_model, max_iters=max_iters)
		train_net(cfg.TRAIN, trdb, output_dir)

	def test(self):
		''' Parameters
		'''
		imdb_name = cfg.TEST.IMDB_NAME
		prototxt = cfg.TEST.PROTOTXT
		caffemodel = cfg.TEST.CAFFEMODEL
		segmentation_mode = cfg.SEGMENTATION_MODE
		''' 
		Setup test set
		'''
		tedb = test_imdb(imdb_name)
		assert len(tedb) > 0, 'The size of Dataset must be > 0: {}'.format(len(tedb))
		print('Test data size: {:d}'.format(len(tedb)))
		print('Sample looks like: {:s}'.format(tedb[len(tedb)-1]))
		'''
		Setup output dir
		'''
		output_dir = get_output_dir(imdb_name, caffemodel_name=os.path.splitext(os.path.basename(caffemodel))[0])
		print('Output will be saved to `{:s}`'.format(output_dir))
		print('=================================================================\n')
		'''
		Load Caffe Net
		'''
		print('Load caffe Net:')
		test_net(cfg.TEST, tedb, output_dir)
		#net = caffe.Net(prototxt, caffemodel, caffe.TEST)
		#test_net(net, tedb, output_dir)










