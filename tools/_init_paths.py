#!/usr/bin/env python
# encoding: utf-8


"""Set up paths for system."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH --> PYTHONPATH=/home/zlp/dev/caffe/python:$PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'caffe', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)
# Add LITS-CHALLENGE
lib_path = osp.join(this_dir, '..', 'evaluation')
add_path(lib_path)

# Add cv2 path
# sys.path.append('/usr/local/lib/python2.7/site-packages')
