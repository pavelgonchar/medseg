#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from multiprocessing import Process, Queue
import os
from os import path as osp
from utils.blob import load_data
from evaluation_notebook import get_scores

class EvaluationWrapper(object):
	"""docstring for EvaluationWrapper"""
	def __init__(self, params):
		super(EvaluationWrapper, self).__init__()
		self.gt_folder = params.GT_DIR
		self.pred_folder = params.PRED_DIR
		self.num_process =  params.NUM_PROCESS
		self.output_dir = params.OUT_DIR
		self.status = []
		print params
		self.imdb = self.load_imdb()

	def evaluation_thread(self, eval_queue, split_index):
		""" Evaluation Thread 
		"""
		data_list = range(split_index[0], split_index[1])
		for ind_data in data_list:
			"""  Get Item Infos
			"""
			gt_path = self.imdb[ind_data]['gt']
			pred_path = self.imdb[ind_data]['pred']
			# filename, ext = osp.splitext(osp.split(gt_path)[1])
			# filename = 'volume-{}'.format(filename.split('-')[1])
			# filename = '{}_pred{}'.format(filename, ext)
			# if ext in ('.jpg', '.png', '.npy'):
			# 	prob_path = osp.join(self.output_dir, filename.split('_slice_')[0], 'label',filename)
			# elif ext in ('.nii'):
			# 	prob_path = osp.join(self.output_dir, 'label', filename)
			# else:
			# 	print 'error'
			print gt_path, pred_path
			""" Load Label and Prob
			"""
			gt_metadata = load_data(gt_path, flags=0)
			prob_metadata = load_data(pred_path, flags=0)
			assert (gt_metadata is not None) and (prob_metadata is not None), 'load failed'
			gt_data = gt_metadata['image_data']
			voxelspacing = gt_metadata['image_header'].get_zooms()[:3]
			prob_data = prob_metadata['image_data']
			# add 2 in case
			#print np.sum(prob_data > 1), np.sum(gt_data > 1)
			if np.sum(prob_data>1) == 0:
				prob_data[0, 0, 0] = 2
			""" Calculate the Scores
			"""
			liver_scores = get_scores(prob_data>=1, gt_data>=1, voxelspacing)
			lesion_scores = get_scores(prob_data==2, gt_data==2, voxelspacing)
			print "Liver dice",liver_scores['dice'], "Lesion dice", lesion_scores['dice']
			eval_queue.put(tuple((gt_path, pred_path, liver_scores, lesion_scores)))

	def evaluation(self):
		""" Evaluation
		Start Evaluation Processing Thread
		"""
		how_many_images = len(self.imdb)
		print "The dataset has {} data".format(how_many_images)
		# Evaluation Queue
		# If maxsize is less than or equal to zero, the queue size is infinite.
		eval_queue = Queue()
		eval_preparation = [None] * self.num_process
		for proc in range(0, len(eval_preparation)):
			# split data to multiple thread
			split_start = (how_many_images//len(eval_preparation) + 1) * proc
			split_end = (how_many_images//len(eval_preparation) + 1) * (proc + 1)
			if split_end > how_many_images:
				split_end = how_many_images
			split_index = (split_start, split_end)

			eval_preparation[proc] = Process(target=self.evaluation_thread, args=(eval_queue, split_index))
			eval_preparation[proc].daemon = True
			eval_preparation[proc].start()
			print('Evaluation Thread {} started~'.format(proc))

		eval_results = osp.join(self.output_dir, 'eval_results.csv')
		results = []
		while len(self.status) < len(self.imdb):
			[im_path, gt_path, liver_scores, lesion_scores] = eval_queue.get()
			results.append([im_path, gt_path, liver_scores, lesion_scores])
			self.status.append([im_path, gt_path, 'Success'])
			print '{}/{}'.format(len(self.status), len(self.imdb))

		headerstr = ''
		if not osp.isfile(eval_results):
			headerstr +='Volume,'
			headerstr += 'Liver_{},'.format('dice')
			headerstr += 'Liver_{},'.format('jaccard')
			headerstr += 'Liver_{},'.format('voe')
			headerstr += 'Liver_{},'.format('rvd')
			headerstr += 'Liver_{},'.format('assd')
			headerstr += 'Liver_{},'.format('msd')
			headerstr += 'lesion_{},'.format('dice')
			headerstr += 'lesion_{},'.format('jaccard')
			headerstr += 'lesion_{},'.format('voe')
			headerstr += 'lesion_{},'.format('rvd')
			headerstr += 'lesion_{},'.format('assd')
			headerstr += 'lesion_{},'.format('msd')
			headerstr += '\n'

		outstr = ''
		outstr += headerstr

		scores = np.zeros((how_many_images, 12),dtype=np.float32)
		for r in xrange(len(results)):
			result = results[r]
			im_path = result[0]
			gt_path = result[1]
			liver_scores = result[2]
			lesion_scores = result[3]
			scores[r, 0:6] = np.array([liver_scores['dice'], liver_scores['jaccard'], liver_scores['voe'], liver_scores['rvd'], liver_scores['assd'], liver_scores['msd']], dtype=np.float32)
			scores[r, 6:] = np.array([lesion_scores['dice'], lesion_scores['jaccard'], lesion_scores['voe'], lesion_scores['rvd'], lesion_scores['assd'], liver_scores['msd']], dtype=np.float32)
			print gt_path
			print scores[r]

			#create line for csv file
			eachline = str(gt_path) + ','
			for i in xrange(scores.shape[1]):
				eachline += str(scores[r, i]) + ','
			eachline += '\n'

			outstr += eachline

		# add mean scores
		mean_scores = np.mean(scores, axis=0)
		print 'liver mean scores: dice, jaccard, voe, rvd, assd, msd'
		print mean_scores[0:6]
		print 'lesion mean scores: dice, jaccard, voe, rvd, assd, msd'
		print mean_scores[6:]

		#create line for csv file
		meanline = str('mean_scores') + ','
		for i in xrange(mean_scores.shape[0]):
			meanline += str(mean_scores[i]) + ','
		meanline += '\n'
		meanline += '\n'
		outstr += meanline
		# write to file
		f = open(eval_results, 'a+')
		f.write(outstr)
		f.close()
		print '====== ====== Evaluation Done ====== ======'

	def load_files_from_folder(self, folder, prefix, suffix):
		''' Load files from file_folder startswith prefix and endswith suffix
		Return a list of files name
		'''
		files = [f for f in os.listdir(folder) if (f.startswith(prefix) and f.endswith(suffix))]
		return files

	def load_prediction(self, folder):
		files = self.load_files_from_folder(folder, prefix='volume-', suffix='.nii')
		files.sort(key=lambda name: int(name.split('-')[1].split('_')[0]))
		return files

	def load_segmentations(self, folder):
		files = self.load_files_from_folder(folder, prefix='segmentation-', suffix='.nii')
		files.sort(key=lambda name: int(name.split('-')[1].split('.')[0]))
		return files

	def load_imdb(self):
		gts_path = []
		preds_path = []
		segmentations = self.load_segmentations(folder=self.gt_folder)
		# Divide dataset to the training set and validation set
		training_set_num = 100
		segmentations = segmentations[training_set_num:]
		predictions = self.load_prediction(folder=self.pred_folder)
		for i in xrange(len(segmentations)):
			gt_path = osp.join(self.gt_folder, segmentations[i])
			pred_path = osp.join(self.pred_folder, predictions[i])
			gt_index = osp.splitext(osp.split(gt_path)[1])[0].split('-')[1]
			pred_index = osp.splitext(osp.split(pred_path)[1])[0].split('-')[1].split('_')[0]
			assert gt_index == pred_index, 'gt_index:{} and pred_index:{} mismatch'.format(gt_index, pred_index)
			gts_path.append(gt_path)
			preds_path.append(pred_path)

		assert len(gts_path)==len(preds_path), 'the number of images must equal to the number of segmentations'

		imdb = []
		for i in xrange(len(gts_path)):
			tmp = dict()
			tmp['gt'] = gts_path[i]
			tmp['pred'] = preds_path[i]
			imdb.append(tmp)

		return imdb

def eval(params):
	ew = EvaluationWrapper(params)
	print 'evaluating...'
	ew.evaluation()
	print 'done solving'



