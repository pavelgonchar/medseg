#!/usr/bin/env python
# encoding: utf-8


"""Factory method for easily getting imdbs by name."""
import numpy as np
from datasets.litsdb import litsdb
from datasets.litsdb3d import litsdb3d
from datasets.lctscdb import lctscdb

__sets = {}

# Set up lits_Training_Batch_1_<split>
for training_batch in ['Training_Batch', 'Training_Batch_isotropic', 'Training_Batch_256_256_64', 'Training_Batch_256_256', 'Test_Batch']:
	for split in ['train', 'val', 'trainval', 'test']:
		# 2D
		name = 'lits_{}_{}_{}'.format(training_batch, split, '2D')
		__sets[name] = (lambda split=split, training_batch=training_batch: litsdb(split, training_batch))
		# 3D
		name = 'lits_{}_{}_{}'.format(training_batch, split, '3D')
		__sets[name] = (lambda split=split, training_batch=training_batch: litsdb3d(split, training_batch))

# Set up lctsc_train
for training_batch in ['256','512']:
	for split in ['train', 'val', 'trainval', 'test']:
		name = 'lctsc_{}_{}'.format(training_batch,split)
		__sets[name] = (lambda split=split, training_batch=training_batch: lctscdb(split, training_batch))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
    	raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()