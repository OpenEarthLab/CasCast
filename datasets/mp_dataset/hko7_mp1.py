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

        self.dummy_data = np.zeros((self.total_length, self.height, self.width), dtype=np.uint8)

        # ### multiprocess data reading ###
        # self.max_numworker = 6
        # self.dataloader_worker_queue_list = []
        # self.dataloader_worker_shm_list = []
        # self.dataloader_worker_dict = {}
        # self.dummy_frame = np.zeros((self.total_length, self.height, self.width), dtype=np.float32)
        # for _ in range(self.max_numworker):
        #     self.dataloader_worker_queue_list.append(multiprocessing.Queue())
        #     shm = shared_memory.SharedMemory(create=True, size=self.dummy_frame.nbytes)
        #     self.dataloader_worker_shm_list.append(shm)
            
        # self.frame_queue = multiprocessing.Queue()
        # self.frame_queue.cancel_join_thread()

        # self.lock = multiprocessing.Lock()

        # self._workers = []
        # for _ in range(20):
        #     w = multiprocessing.Process(target=self.load_data_process)
        #     w.daemon = True
        #     w.start()
        #     self._workers.append(w)
        







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
    
    def load_data_process(self):
        loc, datetime = self.frames_idx_queue.get()
        # print(f'loc:{loc}')
        # if loc is None:
        #     print("break")
        path = convert_datetime_to_filepath(datetime)
        st_data = self.loader(path)
        np_shm = np.ndarray(self.dummy_data.shape, dtype=self.dummy_data.dtype, buffer=self.frames_shm.buf)
        np_shm[loc, :] = st_data[:]
        print(f'end:{loc}')

    def __getitem__(self, idx):
        start_time = datetime.datetime.strptime(self.init_times[idx], "%Y-%m-%d %H:%M:%S")
        datetime_clips = pd.date_range(start=start_time, periods=self.total_length, freq=self.base_freq)

        ## create multiprocess ##
        self.workers = []
        self.frames_shm = shared_memory.SharedMemory(create=True, size=self.dummy_data.nbytes)
        self.frames_idx_queue = multiprocessing.Queue()
        ## load task to queue ##
        for loc, st in enumerate(datetime_clips):
            self.frames_idx_queue.put((loc, st))

        for _ in range(20):
            w = multiprocessing.Process(target=self.load_data_process)
            self.workers.append(w)
            w.start()
        
        ## wait for all process to finish
        for w in self.workers:
            w.join()
        
        ## read data from shm ##
        frame_data = np.ndarray((self.total_length, self.height, self.width), dtype=np.uint8, buffer=self.frames_shm.buf)
        self.frames_shm.unlink()
        frame_data = torch.from_numpy(frame_data)/255.0
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
