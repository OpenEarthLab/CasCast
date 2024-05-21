import imp
from torch.utils.data import Dataset

try:
    from petrel_client.client import Client
except:
    pass
from tqdm import tqdm
import numpy as np
import io
import torch
import os

class checkpoint_ceph(object):
    def __init__(self, conf_path="~/petreloss.conf", checkpoint_dir="weatherbench:s3://weatherbench/checkpoint") -> None:
        self.client = Client(conf_path=conf_path)
        self.checkpoint_dir = checkpoint_dir

    def load_checkpoint(self, url):
        url = os.path.join(self.checkpoint_dir, url)
        # url = self.checkpoint_dir + "/" + url
        if not self.client.contains(url):
            return None
        with io.BytesIO(self.client.get(url, update_cache=True)) as f:
            checkpoint_data = torch.load(f, map_location=torch.device('cpu')) 
        return checkpoint_data
    
    def load_checkpoint_with_ckptDir(self, url, ckpt_dir):
        url = os.path.join(ckpt_dir, url)
        if not self.client.contains(url):
            return None
        with io.BytesIO(self.client.get(url, update_cache=True)) as f:
            checkpoint_data = torch.load(f, map_location=torch.device('cpu')) 
        return checkpoint_data
    
    def save_checkpoint(self, url, data):
        url = os.path.join(self.checkpoint_dir, url)
        # url = self.checkpoint_dir + "/" + url
        with io.BytesIO() as f:
            torch.save(data, f)
            f.seek(0)
            self.client.put(url, f)

    def save_prediction_results(self, url, data):
        url = os.path.join(self.checkpoint_dir, url)
        # url = self.checkpoint_dir + "/" + url
        with io.BytesIO() as f:
            np.save(f, data)
            f.seek(0)
            self.client.put(url, f)
