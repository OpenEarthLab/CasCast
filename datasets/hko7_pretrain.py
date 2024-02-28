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
try:
    from petrel_client import client
except:
    pass

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
        st = time.time()
        img_bytes = self._client.get(path)
        et = time.time()
        try:
            assert(img_bytes is not None)
        except:
            print(path)
            import pdb; pdb.set_trace()
        st = time.time()
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        frame = np.array(img)
        et = time.time()
        return frame

class hko7_pretrain(Dataset):
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



    def __len__(self):
        return len(self.init_times)
    

    def _load_frames(self, datetime_clips, total_length):
        paths = [convert_datetime_to_filepath(datetime_clips[i]) for i in range(len(datetime_clips))]
        read_storage = []
        for i in range(len(paths)):
            read_storage.append(self.loader(paths[i]))
        frame_dat = np.array(read_storage)
        frame_dat = frame_dat * self._exclude_mask
        data_batch = torch.from_numpy(frame_dat) / 255.0 ### c, h, w ###
        return data_batch

    def rainfall_to_pixel(self, rainfall_intensity, a=58.53, b=1.56):
        # dBZ = 10b log(R) +10log(a)
        dBR = np.log10(rainfall_intensity) * 10.0
        dBZ = dBR * b + 10.0 * np.log10(a)
        pixel_vals = (dBZ + 10.0) / 70.0
        return pixel_vals
 
    def __getitem__(self, idx):
        start_time = datetime.datetime.strptime(self.init_times[idx], "%Y-%m-%d %H:%M:%S")
        datetime_clips = pd.date_range(start=start_time, periods=self.total_length, freq=self.base_freq)
        frame_data = _load_frames(self.loader, datetime_clips, total_length=self.total_length, height=self.height, width=self.width, mask=self._exclude_mask)
        packed_results = dict()
        packed_results['inputs'] = frame_data
        packed_results['data_samples'] = frame_data
        return packed_results

if __name__ == '__main__':
    height = 480
    width = 480
    base_freq = '6min'
    total_length = 20 #input+pred
    dataset = hko7_pretrain(split='test', input_length=1, pred_length=0, base_freq=base_freq)
    print(len(dataset))

    st_time = time.time()
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        ed_time = time.time()
        print("time cost: ", (ed_time - st_time)/(i + 1))

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u hko7_pretrain.py ###
