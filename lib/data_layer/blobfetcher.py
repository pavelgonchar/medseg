#!/usr/bin/env python
# encoding: utf-8


""" Blob Fetcher for fetch minibatch blobs for training a network."""

import numpy as np
from multiprocessing import Process, Queue
from lits.config import cfg
from data_layer.minibatch import get_minibatch

class BlobFetcher(object):
    ''''''
    def __init__(self, imdb):
        # install training dataset in to BlobFetcher
        self._imdb = imdb
        # set up the blob fetcher thread
        self._set_blob_fetcher_thread()

    def _shuffle_imdb_inds(self):
        """Randomly permute the training db."""
        self._perm = np.random.permutation(np.arange(len(self._imdb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the imdb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._imdb):
            self._shuffle_imdb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds
      
    def _set_blob_fetcher_thread(self):
        """Set the training dataset to be used during training."""
        
        self._shuffle_imdb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcherThread(self._blob_queue, self._imdb)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcherThread'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._imdb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db)
            return blobs

class BlobFetcherThread(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, imdb):
        super(BlobFetcherThread, self).__init__()
        self._queue = queue
        self._imdb = imdb
        self._perm = None
        self._cur = 0
        self._shuffle_imdb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_imdb_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._imdb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._imdb):
            self._shuffle_imdb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcherThread started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._imdb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db)
            self._queue.put(blobs)