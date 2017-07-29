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
# from lits.config import cfg
# from data_layer.blobfetcher import BlobFetcher
import google.protobuf.text_format
from utils.blob import im_list_to_blob, seg_list_to_blob, prep_weight_for_blob, ims_vis, prepare_data_unit
from multiprocessing import Process, Queue

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process.
    """

    def __init__(self, params, imdb, output_dir):
        """Initialize the SolverWrapper."""
        self.params = params
        self.imdb = imdb
        self.output_dir = output_dir
        self.solver = caffe.SGDSolver(self.params.SOLVER)
        if self.params.PRETRAINED_MODEL is not None:
            print ('Loading pretrained model weights from {:s}').format(self.params.PRETRAINED_MODEL)
            self.solver.net.copy_from(self.params.PRETRAINED_MODEL)
        self.solver_param = caffe_pb2.SolverParameter()
        with open(self.params.SOLVER, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    def snapshot(self):
        """Take a snapshot of the network.
        """
        net = self.solver.net
        
        infix = ('_' + self.params.SNAPSHOT_INFIX
                 if self.params.SNAPSHOT_INFIX != '' else '')
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

    def train_model(self, data_queue, max_iters):
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
            blobs = data_queue.get()
            assert len(blobs['data'].shape) in (4,5), 'dimension error'
            if len(blobs['data'].shape) == 4:
                if blobs['data'].shape[1] == 1:
                    self.solver.net.blobs['data'].data[...] = np.repeat(blobs['data'], 3, axis=1)
                else:
                    self.solver.net.blobs['data'].data[...] = blobs['data']

                if blobs['label'].shape[1] == 1:
                    self.solver.net.blobs['label'].data[...] = blobs['label']
                    ### label_weight
                    if self.params.CLASS.USE_WEIGHT:
                        self.solver.net.blobs['label_weight'].data[...] = blobs['label_weight']
                else:
                    target_channel =  int(blobs['label'].shape[1]) / int(2)
                    # print target_channel
                    target_label = blobs['label'][:,target_channel,:,:]
                    target_label = target_label[:,np.newaxis,:,:]
                    self.solver.net.blobs['label'].data[...] = target_label
                    ### label_weight
                    if self.params.CLASS.USE_WEIGHT:
                        target_label_weight = blobs['label_weight'][:,target_channel,:,:]
                        target_label_weight = target_label_weight[:,np.newaxis,:,:]
                        self.solver.net.blobs['label_weight'].data[...] = target_label_weight
            else:
                self.solver.net.blobs['data'].data[...] = blobs['data']
                self.solver.net.blobs['label'].data[...] = blobs['label']
            ### debug show the network input
            if self.params.DEBUG:
                for batch in xrange(self.solver.net.blobs['data'].data[...].shape[0]):
                    ims_vis_list = []
                    for channel in xrange(self.solver.net.blobs['data'].data[...].shape[1]):
                        ims_vis_list.extend([self.solver.net.blobs['data'].data[...][batch, channel,:,:]])
                    ims_vis_list.extend([self.solver.net.blobs['label'].data[...][batch, 0,:,:]])
                    if self.params.CLASS.USE_WEIGHT:
                        ims_vis_list.extend([self.solver.net.blobs['label_weight'].data[...][batch, 0,:,:]])
                    ims_vis(ims_vis_list)

            # start to solve
            self.solver.step(1)
            timer.toc()
            
            train_loss[self.solver.iter - 1] = self.solver.net.blobs['loss'].data
            if np.mod(self.solver.iter, self.params.DISPLAY_INTERVAL) == 0:
               plt.clf()
               x = np.array(range(0, self.solver.iter, self.params.DISPLAY_INTERVAL)) - 1
               x[0] = 0
               x = np.append(x, self.solver.iter-1)
               plt.plot(x, train_loss[x])
               plt.pause(0.00001)
            plt.show()

            # calculate the average time
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % self.params.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())
                #plt.savefig(os.path.join(self.output_dir, 'plot.png'))

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
            #plt.savefig(os.path.join(self.output_dir, 'plot.png'))
        return model_paths

    def get_blob(self, batch_data_list, batch_scale_list, batch_rotation_list, patch_per_image):
        """ Builds an input blob from imdb at
        the specified scales, flipped, rotated, cropped
        """
        processed_im_patchs = []
        processed_gt_patchs = []
        # for each data unit in batch_data_list, we do:
        for ind_data, ind_scale, ind_rotation in zip(batch_data_list, batch_scale_list, batch_rotation_list):
            """ Get Item Infos
            """
            im_path = self.imdb[ind_data]['image']
            gt_path = self.imdb[ind_data]['gt']
            scale = self.params.SCALES[ind_scale]
            rotation = self.params.ROTATIONS[ind_rotation]
            # im_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/Images/volume-100/volume-100_slice_551.npy'
            # gt_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/Segmentation/segmentation-100/segmentation-100_slice_551.npy'
            # im_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/volume-0.nii'
            # gt_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/segmentation-0.nii'
            # print im_path, gt_path
            # print scale, rotation
            """ Prepare each data unit
            """
            # if self.params.ADJACENT:
            #     im_prepared, gt_prepared = prepare_data_unit_adj(im_path=im_path, gt_path=gt_path, pixel_statistics=self.params.PIXEL_STATISTICS,
            #             target_size=scale, max_size=self.params.MAX_SIZE, trim_params=self.params.TRIM, class_params=self.params.CLASS, cropped_size=self.params.CROPPED_SIZE, patch_per_image=patch_per_image,
            #             hu_window=self.params.HU_WINDOW, data_range=self.params.DATA_RANGE, rotation=rotation, use_flip=self.params.USE_FLIP)
            # else:
            im_prepared, gt_prepared = prepare_data_unit(im_path=im_path, gt_path=gt_path, pixel_statistics=self.params.PIXEL_STATISTICS,
                    target_size=scale, max_size=self.params.MAX_SIZE, trim_params=self.params.TRIM, class_params=self.params.CLASS,
                    adjacnet=self.params.ADJACENT, bg=self.params.BG, cropped_size=self.params.CROPPED_SIZE, patch_per_image=patch_per_image,
                    hu_window=self.params.HU_WINDOW, data_range=self.params.DATA_RANGE, rotation=rotation, use_flip=self.params.USE_FLIP)

            processed_im_patchs.extend(im_prepared)
            processed_gt_patchs.extend(gt_prepared)
        """ Create a blob to hold the input images, gts and class_weights
        Axis order will become: (batch elem, channel, height, width, depth)
        """
        im_blob = im_list_to_blob(processed_im_patchs, im_type=self.params.SEGMENTATION_MODE)
        gt_blob = seg_list_to_blob(processed_gt_patchs, im_type=self.params.SEGMENTATION_MODE)
        weight_blob = prep_weight_for_blob(gt_blob, self.params.CLASS.WEIGHT)
        ### debug show blob
        if self.params.DEBUG:
            print im_blob.shape, gt_blob.shape, weight_blob.shape
            for batch in xrange(im_blob.shape[0]):
                ims_vis_list = []
                for channel in xrange(im_blob.shape[1]):
                    ims_vis_list.extend([im_blob[batch,channel,:,:], gt_blob[batch,channel,:,:], weight_blob[batch,channel,:,:]])
                ims_vis(ims_vis_list)
        """ Return Blobs
        """
        blobs = {'data': im_blob, 'label': gt_blob, 'label_weight': weight_blob}
        return blobs

    def prepare_data_thread(self, data_queue, proc):
        max_iters = self.params.MAX_ITER
        ims_per_batch = self.params.IMS_PER_BATCH
        batch_size = self.params.BATCH_SIZE
        assert(batch_size % ims_per_batch == 0), 'ims_per_batch ({}) must divide batch_size ({})'.format(ims_per_batch, batch_size) 
        patch_per_image = batch_size / ims_per_batch
        # the total number that need to process during the max_iters is num_iter_data
        num_iter_data = max_iters*ims_per_batch
        # the number of data that current thread need to process is len(which_data_list)
        rand_l = (len(self.imdb)//self.params.NUM_PROCESS + 1) * proc
        rand_h = (len(self.imdb)//self.params.NUM_PROCESS + 1) * (proc + 1)
        if rand_h > len(self.imdb):
            rand_h = len(self.imdb)
        which_data_list = np.random.randint(rand_l, high=rand_h, size=int(np.ceil(float(num_iter_data)/self.params.NUM_PROCESS)))
        which_scale_list = np.random.randint(len(self.params.SCALES), size=len(which_data_list))
        which_rotation_list = np.random.randint(len(self.params.ROTATIONS), size=len(which_data_list))
        # the number of batch that current thread need to process is num_batch
        num_batch = int(np.ceil(len(which_data_list)/float(ims_per_batch)))
        for ind_batch in xrange(num_batch):
            cur_batch_b = ims_per_batch*ind_batch
            cur_batch_e = ims_per_batch*(ind_batch+1)
            if cur_batch_e > len(which_data_list):
                cur_batch_b = len(which_data_list) - ims_per_batch
                cur_batch_e = len(which_data_list)
            cur_batch_data_list = which_data_list[cur_batch_b:cur_batch_e]
            cur_batch_scale_list = which_scale_list[cur_batch_b:cur_batch_e]
            cur_batch_rotation_list = which_rotation_list[cur_batch_b:cur_batch_e]
            """ Prepare current batch blobs
            """
            blobs = self.get_blob(cur_batch_data_list, cur_batch_scale_list, cur_batch_rotation_list, patch_per_image)
            """ Put useful infos to data_queue
            """
            data_queue.put(blobs)

    def train(self):
        """ train
        Prepare Data Queue
        Start Data Processing Thread
        Start Train Thread
        """
        how_many_images = len(self.imdb)
        print "The dataset has {} data".format(how_many_images)
        # Data Queue
        data_queue = Queue(32)
        # Data Preparation Thread Creation
        data_preparation = [None] * self.params.NUM_PROCESS
        for proc in range(0, self.params.NUM_PROCESS):
            data_preparation[proc] = Process(target=self.prepare_data_thread, args=(data_queue, proc))
            data_preparation[proc].daemon = True
            data_preparation[proc].start()
            print('Data Thread {} started~'.format(proc))

        # Train Thread
        self.train_model(data_queue, self.params.MAX_ITER)


def train_net(params, imdb, output_dir):
    """Train a CNN network.
        solver_prototxt: a configuration file used to tell caffe how you want the network trained.
        imdb: the dataset used to train the network.
        output_dir: the output directory used to store trained network models.
        pretrained_model: the pretrained model used to initialize your network weights
        max_iters: the maximum number of iterations
    """
    sw = SolverWrapper(params, imdb, output_dir)
    if params.DEBUG is False:
       plt.ion()
    print 'Solving...'
    model_paths = sw.train()
    print 'done solving'
    return model_paths
