#!/usr/bin/env python
# encoding: utf-8


import os.path as osp

def load_image_gt_match_list(fpath):
	image_slice_name = list()
	gt_slice_name = list()
	label1count = list()
	label2count = list()

	with open(fpath, 'r') as f:
		for line in f.readlines():
			if line.startswith('#') or len(line)==0:
				continue

			line = line.strip('\n')
			line = line.strip().split(" ")
			image_slice_name.append(line[0])
			gt_slice_name.append(line[1])
			label1count.append(line[2])
			label2count.append(line[3])

	return image_slice_name, gt_slice_name, label1count, label2count

def make_trainval_set(image_gt_match_list):
	trainval_set_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/ImageSets/trainval.txt'
	test_set_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/ImageSets/test.txt'
	image_slice_name, gt_slice_name, label1count, label2count = load_image_gt_match_list(image_gt_match_list)
	
	test_set_index = list()
	#test_set from 105 to 130
	for i in range(101, 131):
		index = 'volume-{}'.format(i)
		test_set_index.append(index)
		
	f_trainval = open(trainval_set_path, 'w')
	f_test = open(test_set_path, 'w')

	for i in xrange(len(image_slice_name)):
		if image_slice_name[i].split('_slice_')[0] in test_set_index:
			f_test.write('{} {} {} {}\n'.format(image_slice_name[i], gt_slice_name[i], label1count[i], label2count[i]))
		else:
			f_trainval.write('{} {} {} {}\n'.format(image_slice_name[i], gt_slice_name[i], label1count[i], label2count[i]))

	f_trainval.close()
	f_test.close()

image_gt_match_list = '/home/zlp/dev/medseg/data/lits/Training_Batch/ImageSets/image_gt_match_list.txt'
assert osp.exists(image_gt_match_list), 'Path does not exist: {}'.format(image_gt_match_list)
'''
# image_slice_name # # gt_slice_name # # label1count # # label2count #
'''

make_trainval_set(image_gt_match_list)