#!/usr/bin/env python
# encoding: utf-8


"""Train CNN network."""
import caffe
import numpy as np
import os
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import matplotlib.pyplot as plt

from utils.timer import Timer
from lits.config import cfg
from data_layer.blobfetcher import BlobFetcher
import google.protobuf.text_format
#from data_layer.lits_layers import LiTSegDataLayer2D

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process.
    """

    def __init__(self, solver_prototxt, trdb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        ### Create BlobFetcher for fetch the minibatch ###
        self.blobfetcher = BlobFetcher(trdb)
        # blobs = self.blobfetcher.get_next_minibatch()
        # blobs = self.blobfetcher.get_next_minibatch()
        # exit()

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    def snapshot(self):
        """Take a snapshot of the network.
        """
        net = self.solver.net
        
        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        # Save plot
        #filename_plot = (self.solver_param.snapshot_prefix + infix + '_iter_{:d}'.format(self.solver.iter) + '.png')
        filename_plot = (self.solver_param.snapshot_prefix + infix + '.png')
        filename_plot = os.path.join(self.output_dir, filename_plot)
        plt.savefig(filename_plot)
        print 'Save Plot to: {:s}'.format(filename_plot)

        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()

        model_paths = []
        train_loss = np.zeros(max_iters)
        train_accuracy = np.zeros(max_iters)
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()

            ### get minibatch data and label for training the network ###
            blobs = self.blobfetcher.get_next_minibatch()
            # reshape network inputs
            #input_data_shape = blobs['data'].shape
            #input_label_shape = blobs['label'].shape
            #self.solver.net.blobs['data'].reshape(input_data_shape[0],input_data_shape[1],input_data_shape[2],input_data_shape[3])
            #self.solver.net.blobs['label'].reshape(input_label_shape[0],input_label_shape[1],input_label_shape[2],input_label_shape[3])
            
            self.solver.net.blobs['data'].data[...] = blobs['data']
            self.solver.net.blobs['label'].data[...] = blobs['label']
            self.solver.net.blobs['label_weight'].data[...] = blobs['label_weight']

            # start to solve
            self.solver.step(1)
            timer.toc()
            # plot loss and accuracy curve every cfg.TRAIN.DISPLAY_INTERVAL
            train_loss[self.solver.iter - 1] = self.solver.net.blobs['loss'].data
            # train_accuracy[self.sovler.iter] = self.solver.net.blobs['accuracy'].data
            if np.mod(self.solver.iter, cfg.TRAIN.DISPLAY_INTERVAL) == 0:
                # average accuracy every cfg.TRAIN.DISPLAY_INTERVAL
                # train_accuracy[self.solver.iter-cfg.TRAIN.DISPLAY_INTERVAL] = np.average(train_accuracy[self.solver.iter-cfg.TRAIN.DISPLAY_INTERVAL:self.solver.iter]) 
                plt.clf()
                # plt.subplot(1,2,1)
                # plt.plot(range(0, self.solver.iter, cfg.TRAIN.DISPLAY_INTERVAL), train_loss[plot_starts:self.solver.iter])
                # plot x= 1, 1 x cfg.TRAIN.DISPLAY_INTERVAL, 2 x cfg.TRAIN.DISPLAY_INTERVAL
                x = np.array(range(0, self.solver.iter, cfg.TRAIN.DISPLAY_INTERVAL)) - 1
                x[0] = 0
                x = np.append(x, self.solver.iter-1)
                plt.plot(x, train_loss[x])
                # plt.subplot(1,2,2)
                # plt.plot(range(0, self.solver.iter/cfg.TRAIN.DISPLAY_INTERVAL), train_accuracy[0:self.solver.iter:cfg.TRAIN.DISPLAY_INTERVAL])
                plt.pause(0.00001)
            plt.show()

            # calculate the average time
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())
                #plt.savefig(os.path.join(self.output_dir, 'plot.png'))

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
            #plt.savefig(os.path.join(self.output_dir, 'plot.png'))
        return model_paths

def train_net(solver_prototxt, trdb, output_dir,
              pretrained_model=None, max_iters=60000):
    """Train a CNN network.
        solver_prototxt: a configuration file used to tell caffe how you want the network trained.
        trdb: the dataset used to train the network.
        output_dir: the output directory used to store trained network models.
        pretrained_model: the pretrained model used to initialize your network weights
        max_iters: the maximum number of iterations
    """
    sw = SolverWrapper(solver_prototxt, trdb, output_dir, pretrained_model=pretrained_model)
    plt.ion()
    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths