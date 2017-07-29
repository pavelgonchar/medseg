#!/usr/bin/env python
# encoding: utf-8

import caffe
import numpy as np
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from utils.blob import im_list_to_blob, hounsfield_unit_window, normalizer,scale_image, split2chunks, vis_seg, load_data, save_data, trim_volume_square, prepare_adj_data
import os
from os import path as osp
import cv2
from medpy import metric
from evaluation_notebook import get_scores

class InferenceWrapper(object):
    """docstring for InferenceWrapper"""
    def __init__(self, params, imdb, output_dir):
        super(InferenceWrapper, self).__init__()
        self.params = params
        self.net = caffe.Net(self.params.PROTOTXT, self.params.CAFFEMODEL, caffe.TEST)
        self.imdb = imdb
        self.output_dir = output_dir
        self.status = []
        assert self.params.SEGMENTATION_MODE in ('2D', '3D'), '{} is not a valid segmentation mode'.format(self.params.SEGMENTATION_MODE)

    def prepare_data_unit(self, im_path, gt_path):
        """ Prepare Data
        Load, HU, Normalize, Subtract Mean, Zoom
        """
        """ Load Seg and Data Preprocessing
        """
        if gt_path is None:
            gt = None
        else:
            gt_metadata = load_data(gt_path, flags=0)
            assert gt_metadata is not None, 'gt_metadata is None'
            # parse metadata
            gt = gt_metadata['image_data']
            gt_type = gt_metadata['image_type']
            gt = gt.astype(np.float32, copy=False)
            """ Scale image
            """
            #gt = scale_image(gt, target_size=self.params.SCALES[0], max_size=self.params.MAX_SIZE, im_type=gt_type)
        """ Load Image and Data Preprocessing
        """
        im_metadata = load_data(im_path, flags=0)
        assert im_metadata is not None, 'im_metadata is None'
        # parse metadata
        im = im_metadata['image_data']
        im_type = im_metadata['image_type']
        """ ADJACENT
        Load Adjacent Slices if image type is 2D
        """
        if self.params.ADJACENT and im_type=="2D":
            adj_im_data = prepare_adj_data(im_path)
            im_merged = np.zeros((im.shape[0], im.shape[1], 5), dtype=np.float32)
            im_merged[:,:,0] = adj_im_data[0]
            im_merged[:,:,1] = adj_im_data[1]
            im_merged[:,:,2] = im[:,:,0]
            im_merged[:,:,3] = adj_im_data[2]
            im_merged[:,:,4] = adj_im_data[3]
            im = im_merged
        if (gt is not None) and self.params.ADJACENT and gt_type=="2D":
            adj_gt_data = prepare_adj_data(gt_path)
            gt_merged = np.zeros((gt.shape[0], gt.shape[1], 5), dtype=np.float32)
            gt_merged[:,:,0] = adj_gt_data[0]
            gt_merged[:,:,1] = adj_gt_data[1]
            gt_merged[:,:,2] = gt[:,:,0]
            gt_merged[:,:,3] = adj_gt_data[2]
            gt_merged[:,:,4] = adj_gt_data[3]
            gt = gt_merged
        im = im.astype(np.float32, copy=False)
        """ Apply Hounsfield Unit Window
        The Hounsfield unit values will be windowed in the range
        HU_WINDOW to exclude irrelevant organs and objects.
        """
        im = hounsfield_unit_window(im, hu_window=self.params.HU_WINDOW)
        """ Zero-center and Normalization
        """
        im -= self.params.PIXEL_STATISTICS[0]
        im /= self.params.PIXEL_STATISTICS[1]
        #im = normalizer(im, src_range=self.params.HU_WINDOW, dst_range=self.params.DATA_RANGE)
        """ Scale image
        """
        #if self.params.ADJACENT and im_type=="2D":
        #    im_scaled = np.zeros((self.params.SCALES[0], self.params.SCALES[0], im.shape[2]), dtype=np.float32)
        #    for ind in xrange(im.shape[2]):
        #        im_scaled[:,:,ind] = scale_image(im[:,:,ind], target_size=self.params.SCALES[0], max_size=self.params.MAX_SIZE, im_type=im_type) 
        #    im = im_scaled   
        #else:
        #    im = scale_image(im, target_size=self.params.SCALES[0], max_size=self.params.MAX_SIZE, im_type=im_type)
        """ return useful infos
        """
        return [im_path, gt_path, im, gt]

    def prepare_data_thread(self, data_queue, split_index):
        """ Prepare Data
        Load, HU, Normalize, Subtract Mean, Zoom
        """
        data_list = range(split_index[0], split_index[1])
        for ind_data in data_list:
            """  Get Item Infos
            """
            im_path = self.imdb[ind_data]['image']
            gt_path = self.imdb[ind_data]['gt']
            # im_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/Images/volume-0/volume-0_slice_58.npy'
            # gt_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/Segmentation/segmentation-0/segmentation-0_slice_58.npy'
            # im_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/Images/volume-100/volume-100_slice_550.npy'
            # gt_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/Segmentation/segmentation-100/segmentation-100_slice_550.npy'
            # im_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/volume-0.nii'
            # gt_path = '/home/zlp/dev/medseg/data/lits/Training_Batch/segmentation-0.nii'
            # print im_path, gt_path
            """ prepare_data_unit
            """
            data_unit = self.prepare_data_unit(im_path, gt_path)
            """ Put data_unit to data_queue
            """
            data_queue.put(tuple(data_unit))
        print "thread done"

    def do_forward_2d(self, image, chunk_shape=None, stride=None):
        input_data = im_list_to_blob([image], im_type='2D')
        # reshape network inputs and feed input to network data layer
        self.net.blobs['data'].reshape(*(input_data.shape))
        self.net.blobs['data'].data[...] = input_data
        # do forward
        blobs_out = self.net.forward()
        blobs_out = self.net.forward()
        prob = blobs_out['prob'][0]
        outmap = np.argmax(prob, axis=0)
        # Gets the network output score
        # score = blobs_out['score'][0]
        # Gets target map
        # outmap = np.argmax(score, axis=0)
        # outmap = np.amax(score, axis=0)
        # print blobs_out.keys()

        # upscore3 = blobs_out['upscore3'][0]
        # upscore2 = blobs_out['upscore2'][0]
        # upscore1 = blobs_out['upscore1'][0]
        # score = blobs_out['score'][0]
        # outmap = np.argmax(upscore3 * 0.125 + upscore2 * 0.125 + upscore1 * 0.25 + score * 0.5, axis=0)
        # fig, axes = plt.subplots(1, 6)
        # vis_seg(axes, [input_data[0,0,:,:], outmap, outmap_v])
        # vis_seg(axes, [input_data[0,0,:,:], np.argmax(upscore3, axis=0), np.argmax(upscore2, axis=0), np.argmax(upscore1, axis=0), np.argmax(score, axis=0), outmap])
        #exit()

        return outmap

    def do_forward_3d(self, image, chunk_shape=None, stride=None):
        """ Due to the limited GPU memory,
        use overlapped sliding windows strategy to crop sub-volumes
        then used the average of the probability maps of these sub-volumes to get the whole volume prediction
        """
        # count is for counting the times of each position has overlapped
        count_overlap = np.zeros(image.shape, dtype=np.float32)
        prediction_map = np.zeros(image.shape, dtype=np.float32)
        chunks_index = split2chunks(image.shape, chunk_shape, stride)
        for ind_chunk in chunks_index:
            im_chunk = image[ind_chunk[0][0]:ind_chunk[0][1], ind_chunk[1][0]:ind_chunk[1][1], ind_chunk[2][0]:ind_chunk[2][1]]
            """ Create a blob to hold the input blobs
            Axis order will become: (batch elem, channel, height, width, depth)
            """
            input_data = im_list_to_blob([im_chunk], im_type='3D')
            # reshape network inputs
            self.net.blobs['data'].reshape(*(input_data.shape))
            self.net.blobs['data'].data[...] = input_data
            # do forward
            blobs_out = self.net.forward()
            # get network output prob
            score = blobs_out['score']
            # get the target map
            outmap = np.argmax(score[0], axis=0)
            # stitch the chunk
            prediction_map[ind_chunk[0][0]:ind_chunk[0][1], ind_chunk[1][0]:ind_chunk[1][1], ind_chunk[2][0]:ind_chunk[2][1]] += outmap
            count_overlap[ind_chunk[0][0]:ind_chunk[0][1], ind_chunk[1][0]:ind_chunk[1][1], ind_chunk[2][0]:ind_chunk[2][1]] += 1
     
        # get the final results
        prediction_map = prediction_map / count_overlap

        return prediction_map
    def get_output_map(self, im, gt):
        """ Input image(3 dimension) and output pred_map(2 dimension) for input image
        """
        pred_map = np.zeros(im.shape[0:2], dtype=im.dtype)
        if not self.params.APPLY_MASK:
            if not self.params.ADJACENT:
                # convert one channel image to 3 channel
                im = np.repeat(im, 3, axis=2)
            pred_map[:,:] = self.do_forward_2d(im)
        else:
            ### Clean the bg of im
            if self.params.BG.CLEAN:
                assert gt is not None, 'gt must be not None'
                if self.params.BG.BRIGHT:
                    fg = np.zeros(im.shape, dtype=np.float)
                    fg[gt>=0.5] = im[gt>=0.5]
                    im[gt<0.5] = np.amax(fg)
                else:
                    bg = np.zeros(im.shape, dtype=np.float)
                    bg[gt<0.5] = im[gt<0.5]
                    im[gt<0.5] = np.amin(bg)
            ### Apply mask to get target Input Patch ###
            assert gt is not None, 'gt must be not None'
            row_col_page_b, new_size = trim_volume_square(gt, min_size=self.params.TRIM.MINSIZE, pad=self.params.TRIM.PAD)
            row_col_page_e = row_col_page_b + new_size
            input_patch = im[row_col_page_b[0]:row_col_page_e[0], row_col_page_b[1]:row_col_page_e[1],:]
            ### Scale input_patch to fix target_size
            if self.params.ADJACENT:
                input_patch_zoom_in = np.zeros((self.params.SCALES[0], self.params.SCALES[0], input_patch.shape[2]), dtype=np.float32)
                for ind in xrange(input_patch.shape[2]):
                    input_patch_zoom_in[:,:,ind] = scale_image(input_patch[:,:,ind], target_size=self.params.SCALES[0], max_size=self.params.MAX_SIZE, im_type='2D')
            else:
                input_patch_zoom_in = scale_image(input_patch, target_size=self.params.SCALES[0], max_size=self.params.MAX_SIZE, im_type='2D')
                # convert one channel image to 3 channel
                input_patch_zoom_in = np.repeat(input_patch_zoom_in, 3, axis=2)
            ### forward get the prob
            outmap = self.do_forward_2d(input_patch_zoom_in)
            ### Scale prob back to input_patch size
            outmap = outmap.astype(np.float32)
            outmap_zoom_out = scale_image(outmap, target_size=new_size[0], max_size=self.params.MAX_SIZE, im_type='2D')
            # pred_map
            pred_map[row_col_page_b[0]:row_col_page_e[0], row_col_page_b[1]:row_col_page_e[1]] = outmap_zoom_out
            if self.params.DEBUG:
                fig, axes = plt.subplots(1, 5)
                vis_seg(axes, [im[:,:,0], gt[:,:,0], input_patch_zoom_in[:,:,0], outmap_zoom_out, pred_map])
        return pred_map

    def test_2d(self, data_unit):
        """Test Thread for 2d segmentation
        Assumes images are already prepared
        """
        """ Get data from data_unit """
        [im_path, gt_path, im, gt] = data_unit
        print im_path, gt_path
        filename, ext = osp.splitext(osp.split(im_path)[1])
        if filename.split('-')[0] == 'test':
            '''TEST'''
            fileindex = filename.split('-')[2]
            filename = 'test-segmentation-{}{}'.format(fileindex, ext)
        else:
            filename = '{}_pred{}'.format(filename, ext)
        """ Do Forward
        """
        prediction_map = np.zeros(im.shape, dtype=im.dtype)
        if ext in ('.jpg', '.png', '.npy'):
            prediction_map[:,:,0] = self.get_output_map(im, gt)
            if self.params.DEBUG:
                fig, axes = plt.subplots(1, 3)
                vis_seg(axes, [im[:,:,im.shape[2]/2], gt[:,:,gt.shape[2]/2], prediction_map[:,:,0]])
                exit()
            # construct output_path
            outdir = osp.join(self.output_dir, filename.split('_slice_')[0])
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            output_path = osp.join(outdir, filename)
        elif ext in ('.nii'):
            if self.params.DEBUG:
                plt.ion()
            if self.params.FAST_INFERENCE:
                """ Apply mask to remove irrelavant slices for fast inference """
                # [0:row_col_page_b[2]] and row_col_page_e[2]:] no need inference
                assert gt is not None, 'gt must be not None'
                row_col_page_b, new_size = trim_volume_square(gt, min_size=[1,1,1], pad=[0,0,1])
                row_col_page_e = row_col_page_b + new_size
                ind_begin = row_col_page_b[2]
                ind_end = row_col_page_e[2]
            else:
                ind_begin = 0
                ind_end = im.shape[2]
            for ind_slice in xrange(ind_begin, ind_end):
                ### Get input data
                if self.params.ADJACENT:
                    ind_slices_merged = [max(0, ind_slice - 2), max(0, ind_slice - 1), ind_slice, min(im.shape[2]-1, ind_slice + 1), min(im.shape[2]-1, ind_slice + 2)]
                    each_im = im[:,:,ind_slices_merged]
                    each_gt = gt[:,:,ind_slices_merged]
                else:
                    each_im = im[:,:,ind_slice]
                    each_im = each_im[:,:,np.newaxis]
                    each_gt = gt[:,:,ind_slice]
                    each_gt = each_gt[:,:,np.newaxis]
                prediction_map[:,:,ind_slice] = self.get_output_map(each_im, each_gt)
            if self.params.DEBUG:
                fig, axes = plt.subplots(1, 3)
                for ind_slice in xrange(ind_begin, ind_end):
                    vis_seg(axes, [im[:,:,ind_slice], gt[:,:,ind_slice], prediction_map[:,:,ind_slice]])
                exit()
            # construct output_path
            output_path = osp.join(self.output_dir, filename)
        else:
            print 'error'
        ### transform the datatype to unint8
        prediction_map = prediction_map.astype(np.uint8)
        """ Save """
        print('Save to {}'.format(output_path))
        save_data(prediction_map, output_path)
        """ return status """
        return [im_path, gt_path, 'Success']

    def test_3d(self, data_unit):
        """ Test Thread 3d segmentation
        Assumes images are already prepared
        """
        """ Get Data from data_unit
        """
        [im_path, gt_path, im, gt] = data_unit
        print im_path, gt_path
        filename, ext = osp.splitext(osp.split(im_path)[1])
        filename = '{}_pred{}'.format(filename, ext)
        """ Do Forward
        """
        prediction_map = self.do_forward_3d(im, self.params.CHUNK_SHAPE, self.params.STRIDE)
        # transform the datatype to unint8
        prediction_map = prediction_map.astype(np.uint8)
        # fig, axes = plt.subplots(1, 5)
        # for slices in xrange(0,prediction_map.shape[2]):
        #     vis_seg(axes, [im[:,:,slices], gt[:,:,slices], prediction_map[:,:,slices]])
        """ Save
        """
        # construct output_path
        output_path = osp.join(self.output_dir, filename)
        print('Save to {}'.format(output_path))
        save_data(prediction_map, output_path)
        """ return status
        """
        return [im_path, gt_path, 'Success']

    def test_thread(self, data_queue):
        """ Test Thread for Segmentation Using multiprocess
        Assumes images are already prepared
        """
        while len(self.status) < len(self.imdb):
            """ Get data from queue
            """
            data_unit = data_queue.get()
            print('Infos: {} / {}'.format(len(self.status)+1, len(self.imdb)))
            if self.params.SEGMENTATION_MODE == "2D":
                status = self.test_2d(data_unit)
            else:
                status = self.test_3d(data_unit)
            self.status.append(status)

        print '====== ====== Test Done ====== ======'

    def test_worker(self):
        """ Test Worker for Segmentation without using multiprocessing
        Assumes images are already prepared
        """
        for ind_data in xrange(len(self.imdb)):
            """  Get Item Infos
            """
            im_path = self.imdb[ind_data]['image']
            gt_path = self.imdb[ind_data]['gt']
            """ prepare_data_unit
            """
            data_unit = self.prepare_data_unit(im_path, gt_path)
            print('Infos: {} / {}'.format(len(self.status)+1, len(self.imdb)))
            if self.params.SEGMENTATION_MODE == "2D":
                self.test_2d(data_unit)
            else:
                self.test_3d(data_unit)
            self.status.append([im_path, gt_path, 'Success'])

        print '====== ====== Test Done ====== ======'

    def evaluation_thread(self, eval_queue, split_index):
        """ Evaluation Thread 
        """
        data_list = range(split_index[0], split_index[1])
        for ind_data in data_list:
            """  Get Item Infos
            """
            im_path = self.imdb[ind_data]['image']
            gt_path = self.imdb[ind_data]['gt']
            filename, ext = osp.splitext(osp.split(gt_path)[1])
            filename = 'volume-{}'.format(filename.split('-')[1])
            filename = '{}_pred{}'.format(filename, ext)
            if ext in ('.jpg', '.png', '.npy'):
                prob_path = osp.join(self.output_dir, filename.split('_slice_')[0], filename)
            elif ext in ('.nii'):
                prob_path = osp.join(self.output_dir, filename)
            else:
                print 'error'
            print gt_path, prob_path
            """ Load Label and Prob
            """
            gt_metadata = load_data(gt_path, flags=0)
            prob_metadata = load_data(prob_path, flags=0)
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
            eval_queue.put(tuple((im_path, gt_path, liver_scores, lesion_scores)))

    def test(self):
        """ test
        Prepare Data Queue
        Start Data Processing Thread
        Start Test Thread
        """
        """Clear old status"""
        self.status = []

        how_many_images = len(self.imdb)
        print "The dataset has {} data".format(how_many_images)

        if self.params.NUM_PROCESS == 0:
            self.test_worker()
        else:
            # Data Queue
            data_queue = Queue(10)
            # Data Preparation Thread Creation
            data_preparation = [None] * self.params.NUM_PROCESS
            for proc in range(0, self.params.NUM_PROCESS):
                # split data to multiple thread
                split_start = (how_many_images//self.params.NUM_PROCESS + 1) * proc
                split_end = (how_many_images//self.params.NUM_PROCESS + 1) * (proc + 1)
                if split_end > how_many_images:
                    split_end = how_many_images
                split_index = (split_start, split_end)

                data_preparation[proc] = Process(target=self.prepare_data_thread, args=(data_queue, split_index))
                data_preparation[proc].daemon = True
                data_preparation[proc].start()
                print('Data Thread {} started~'.format(proc))

            # Test Thread
            self.test_thread(data_queue)

            for proc in range(0, self.params.NUM_PROCESS):
                data_preparation[proc].join()

    def evaluation(self):
        """ Evaluation
        Start Evaluation Processing Thread
        """
        """Clear old status"""
        self.status = []
        # self.imdb = self.imdb[0:4]
        how_many_images = len(self.imdb)
        print "The dataset has {} data".format(how_many_images)
        # Evaluation Queue
        # If maxsize is less than or equal to zero, the queue size is infinite.
        eval_queue = Queue()
        eval_preparation = [None] * self.params.NUM_PROCESS
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

        #     #create line for csv file
        #     outstr = str(gt_path) + ','
        #     for l in [liver_scores, lesion_scores]:
        #         for k,v in l.iteritems():
        #             outstr += str(v) + ','
        #     outstr += '\n'

        #     #create header for csv file if necessary
        # if not os.path.isfile(eval_results):
        #     headerstr = 'Volume,'
        #     for k,v in liver_scores.iteritems():
        #         headerstr += 'Liver_' + k + ','
        #     for k,v in liver_scores.iteritems():
        #         headerstr += 'Lesion_' + k + ','
        #     headerstr += '\n'
        #     outstr = headerstr + outstr

        #     #write to file
        #     f = open(eval_results, 'a+')
        #     f.write(outstr)
        #     f.close()

        #create header for csv file if necessary
        headerstr = ''
        if not os.path.isfile(eval_results):
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

def test_net(params, imdb, output_dir):
    """
    Test a CNN network.
    prototxt: the test network structure
    caffemodel: the trained model
    imdb: the data used to test the network
    output_dir: the output directory used to store results
    """
    iw = InferenceWrapper(params, imdb, output_dir)
    print 'Solving...'
    # plt.ion() # turn on interactive mode
    assert params.MODE in ('TEST', 'EVAL', 'TESTEVAL'), '{} not supported'.format(params.MODE)
    if params.MODE == 'TEST':
        iw.test()
    elif params.MODE == 'EVAL':
        iw.evaluation()
    else:
        iw.test()
        iw.evaluation()
    print 'done solving'


		
