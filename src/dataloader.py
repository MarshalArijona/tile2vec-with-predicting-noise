#!/usr/bin/python
"""
Minor update from the default torch data loader.
This version yields the data indices along with the data.
Necessary to update the NAT targets.
"""

import sys
import queue
import threading
import multiprocessing

from typing import Callable, List, Tuple

import torch
from torch.utils import data
from torch.utils.data.sampler import (
    Sampler,
    BatchSampler,
    RandomSampler,
    SequentialSampler
)

import torch.utils.data.dataloader as torch_loader


def _worker_loop(dataset: data.Dataset,
                 index_queue: multiprocessing.SimpleQueue,
                 data_queue: multiprocessing.SimpleQueue,
                 collate_fn: Callable) -> None:
    torch_loader._use_shared_memory = True

    torch.set_num_threads(1)
    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            samples = collate_fn([dataset[i] for i in batch_indices])
        except Exception:
            data_queue.put((idx, torch_loader.ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, [batch_indices]+samples))


def _pin_memory_loop(in_queue: multiprocessing.SimpleQueue,
                     out_queue: queue.Queue,
                     done_event: threading.Event):
    while True:
        try:
            r = in_queue.get()
        except Exception:
            if done_event.is_set():
                return
            raise
        if r is None:
            break
        if isinstance(r[1], torch_loader.ExceptionWrapper):
            out_queue.put(r)
            continue

        idx, batch_content = r
        batch_indices = batch_content[0]
        batch = batch_content[1:]

        try:
            batch = torch_loader.pin_memory_batch(batch)
        except Exception:
            out_queue.put((idx, torch_loader.ExceptionWrapper(sys.exc_info())))
        else:
            out_queue.put((idx, [batch_indices]+batch))


class DataLoaderIter(object):
    """Iterates once over the DataLoader's dataset, as specified by the sampler"""

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.index_queue = multiprocessing.SimpleQueue()
            self.data_queue = multiprocessing.SimpleQueue()
            self.batches_outstanding = 0
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}

            self.workers = [
                multiprocessing.Process(
                    target=_worker_loop,
                    args=(self.dataset, self.index_queue, self.data_queue, self.collate_fn))
                for _ in range(self.num_workers)]

            for w in self.workers:
                w.daemon = True  # ensure that the worker exits on process exit
                w.start()

            if self.pin_memory:
                in_data = self.data_queue
                self.data_queue = queue.Queue()
                self.pin_thread = threading.Thread(
                    target=_pin_memory_loop,
                    args=(in_data, self.data_queue, self.done_event))
                self.pin_thread.daemon = True
                self.pin_thread.start()

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self) -> int:
        return len(self.batch_sampler)

    def __next__(self) -> Tuple[List[int], torch.FloatTensor, torch.FloatTensor]:
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collate_fn([self.dataset[i] for i in indices])
            if self.pin_memory:
                batch = torch_loader.pin_memory_batch(batch)
            return [indices] + batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queue.put((self.send_idx, indices))
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, torch_loader.ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()
            for _ in self.workers:
                self.index_queue.put(None)

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class DataLoader(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch.
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If False and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        iter_cls (Class, optional): Defines the dataloader iterator class.
        Default class: DataLoaderIter
    """

    def __init__(self,
                 dataset: data.Dataset,
                 batch_size: int=1,
                 shuffle: bool=False,
                 sampler: Sampler=None,
                 batch_sampler: Sampler=None,
                 num_workers: int=0,
                 collate_fn: Callable=torch_loader.default_collate,
                 pin_memory: bool=False,
                 drop_last: bool=False,
                 iter_cls: Callable=DataLoaderIter):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.iter_cls = iter_cls

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return self.iter_cls(self)

    def __len__(self):
        return len(self.batch_sampler)
