import pandas as pd
import datetime
import cv2
import numpy as np
import threading
import os
import io
import logging
import struct
import time
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
from petrel_client import client

import multiprocessing
from multiprocessing import Process, Queue, Pool
from multiprocessing import shared_memory
import copy


def convert_datetime_to_filepath(date_time):
    Image_Path = 'radar:s3://weather_radar_datasets/HKO-7/radarPNG'
    ret = os.path.join("%04d" % date_time.year,
                       "%02d" % date_time.month,
                       "%02d" % date_time.day,
                       'RAD%02d%02d%02d%02d%02d00.png'
                       % (date_time.year - 2000, date_time.month, date_time.day,
                          date_time.hour, date_time.minute))
    ret = os.path.join(Image_Path, ret)
    return ret

def get_exclude_mask():
    with np.load('/mnt/cache/gongjunchao/workdir/empty/radar-forecasting/nwp/datasets/hko/data_provider/mask_dat.npz') as dat:
        exclude_mask = dat['exclude_mask'][:]
        return exclude_mask

def _load_frames(loader, datetime_clips, total_length, height, width, mask):
    _exclude_mask = mask
    paths = [convert_datetime_to_filepath(datetime_clips[i]) for i in range(len(datetime_clips))]
    read_storage = []
    for i in range(len(paths)):
        read_storage.append(loader(paths[i]))
    frame_dat = np.array(read_storage)
    frame_dat = frame_dat * _exclude_mask
    data_batch = torch.from_numpy(frame_dat) / 255.0
    return data_batch

class PetrelLoader:
    def __init__(self):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')
        self._client = client.Client("~/petreloss.conf")

    def __call__(self, path):
        img_bytes = self._client.get(path)
        try:
            assert(img_bytes is not None)
        except:
            print(path)
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        frame = np.array(img)
        return frame

class data_hko(Dataset):
    def __init__(self, split, input_length, pred_length, base_freq, height=480, width=480, **kwargs):
        super().__init__()
        ## load data pkl ##
        if split == "train":
            pd_path = '/mnt/cache/gongjunchao/workdir/empty/radar-forecasting/nwp/datasets/hko/data_provider/pd/hko7_rainy_train.pkl'
        elif split == "valid":
            pd_path = '/mnt/cache/gongjunchao/workdir/empty/radar-forecasting/nwp/datasets/hko/data_provider/pd/hko7_rainy_valid.pkl'
        elif split == "test":
            pd_path = '/mnt/cache/gongjunchao/workdir/empty/radar-forecasting/nwp/datasets/hko/data_provider/pd/hko7_rainy_test.pkl'
        
        self.input_length = input_length
        self.pred_length = pred_length
        self.total_length = self.input_length + self.pred_length

        self.df = pd.read_pickle(pd_path)
        self.init_times = self.df[slice(0, len(self.df), self.total_length)] 
        self.base_freq = base_freq
        self.height = height
        self.width = width
        self.split = split
        self.loader = PetrelLoader()
        print('{} number: {}'.format(str(self.split), len(self.init_times)))

        self.client = client.Client("~/petreloss.conf")
        self._exclude_mask = 1-get_exclude_mask()[np.newaxis, :, :]

        ### multiprocess data reading ###
        self.max_numworker = 6
        self.task_queue = multiprocessing.Queue()
        self.compound_data_queue_dict = {}
        self.compound_data_queue_list = []
        self.dl_worker_shm_list = []
        self.dl_worker_shm_dict = {}

        self.dummy_frame = np.zeros((self.total_length, self.height, self.width), dtype=np.uint8)
        for _ in range(self.max_numworker):
            self.compound_data_queue_list.append(multiprocessing.Queue())
            shm = shared_memory.SharedMemory(create=True, size=self.dummy_frame.nbytes)
            shm.unlink()
            self.dl_worker_shm_list.append(shm)
            
        self.frame_queue = multiprocessing.Queue()
        self.frame_queue.cancel_join_thread()

        self.lock = multiprocessing.Lock()
        ## record the map from list idx to pid in the form of array ##
        self.arr = multiprocessing.Array('i', range(self.max_numworker))

        self._workers = []
        for _ in range(20):
            w = multiprocessing.Process(target=self.load_data_process)
            w.daemon = True
            w.start()
            self._workers.append(w)

        self.frame2compound_queue = multiprocessing.Queue()
        w = multiprocessing.Process(target=self.data_compound_process)
        w.daemon = True
        w.start()
        self._workers.append(w)

        

    def __len__(self):
        return len(self.init_times)
    

    def _load_frames(self, datetime_clips, total_length):
        paths = [convert_datetime_to_filepath(datetime_clips[i]) for i in range(len(datetime_clips))]
        read_storage = []
        for i in range(len(paths)):
            read_storage.append(self.loader(paths[i]))
        frame_dat = np.array(read_storage)
        frame_dat = frame_dat * self._exclude_mask
        data_batch = torch.from_numpy(frame_dat) / 255.0
        return data_batch

    def rainfall_to_pixel(self, rainfall_intensity, a=58.53, b=1.56):
        # dBZ = 10b log(R) +10log(a)
        dBR = np.log10(rainfall_intensity) * 10.0
        dBZ = dBR * b + 10.0 * np.log10(a)
        pixel_vals = (dBZ + 10.0) / 70.0
        return pixel_vals
    
    def data_compound_process(self):
        """
        compound process to collect data for differen data loader worker.
        """
        recorder_dict = {}
        while True:
            job_pid, loc = self.frame2compound_queue.get()
            ## check job_pid ##
            if job_pid not in self.compound_data_queue_dict:
                raise ValueError("job_pid not in compound_data_queue_dict")
            
            ## update recorder ##
            if (job_pid, loc) in recorder_dict:
                recorder_dict[(job_pid, loc)] += 1
            else:
                recorder_dict.update({(job_pid, loc): 1})
            
            ## send ready data ##
            if recorder_dict[(job_pid, loc)] == self.total_length:
                del recorder_dict[(job_pid, loc)]
                self.compound_data_queue_dict[job_pid].put(1) ## 1 indicating coressponding data is ready


    def load_data_process(self):
        """
        the target func of multiprocess
        """
        while True:
            job_pid, loc, st = self.task_queue.get()
            ## check job_pid
            if job_pid not in self.compound_data_queue_dict:
                raise ValueError("job_pid not in compound_data_queue_dict")
            ## read data ##
            path = convert_datetime_to_filepath(st)
            st_data = self.loader(path)
            b = np.ndarray(self.dummy_frame.shape, dtype=self.dummy_frame.dtype, buffer=self.dl_worker_shm_dict[job_pid].buf)
            b[loc] = st_data
            self.frame2compound_queue.put((job_pid, loc))


    
    def get_data(self, datetime_clips):
        job_pid = os.getpid()
        ## initialize ##
        if job_pid not in self.compound_data_queue_dict:
            try:
                self.lock.acquire()
                for i in range(self.max_numworker):
                    if i == self.arr[i]: ## meaning that queue and shm i is empty
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue_list[i]
                        self.dl_worker_shm_dict[job_pid] = self.dl_worker_shm_list[i]
                        break
                ## num_worker exceed the maximum resource ##
                if (i == self.max_numworker - 1) and job_pid != self.arr[i]:
                    print("error", job_pid, self.arr)
            except:
                raise ValueError("initialize error")
            finally:
                self.lock.release()
        #############################

        b = np.ndarray(self.dummy_frame.shape, dtype=self.dummy_frame.dtype, buffer=self.dl_worker_shm_dict[job_pid].buf)
        for loc, st in enumerate(datetime_clips):
            self.task_queue.put((job_pid, loc, st))
        idx = self.compound_data_queue_dict[job_pid].get()
        return_data = copy.deepcopy(b)
        return return_data


    def __getitem__(self, idx):
        start_time = datetime.datetime.strptime(self.init_times[idx], "%Y-%m-%d %H:%M:%S")
        datetime_clips = pd.date_range(start=start_time, periods=self.total_length, freq=self.base_freq)
        frame_data = _load_frames(self.loader, datetime_clips, total_length=self.total_length, height=self.height, width=self.width, mask=self._exclude_mask)
        packed_results = dict()
        packed_results['inputs'] = torch.unsqueeze(frame_data[:self.input_length], dim=1)
        packed_results['data_samples'] = torch.unsqueeze(frame_data[self.input_length:self.input_length+self.pred_length], dim=1)
        return packed_results

if __name__ == '__main__':
    height = 480
    width = 480
    base_freq = '6min'
    total_length = 20 #input+pred
    dataset = data_hko(split='train', input_length=10, pred_length=10, base_freq=base_freq)
    print(len(dataset))

    st_time = time.time()
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        ed_time = time.time()
        print("time cost: ", (ed_time - st_time)/(i + 1))

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u hko7.py ###
