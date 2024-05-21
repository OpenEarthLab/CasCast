import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
import random
try:
    from petrel_client.client import Client
except:
    pass
import io



class sevir_pretrain(Dataset):
    def __init__(self, split, input_length=13, pred_length=12, data_dir='path/to/sevir', base_freq='5min', height=384, width=384, **kwargs):
        super().__init__()
        assert input_length == 13, pred_length==12
        self.input_length = 13
        self.pred_length = 12
        self.total_length = self.input_length + self.pred_length

        self.file_list = self._init_file_list(split)
        self.data_dir = os.path.join(data_dir, f'{split}_2h')


    def _init_file_list(self, split):
        if split == 'train':
            txt_path = 'datasets/sevir_list/train.txt'
        elif split == 'valid':
            txt_path = 'datasets/sevir_list/val.txt'
        elif split == 'test':
            txt_path = 'datasets/sevir_list/test.txt'
        files = []
        with open(f'{txt_path}', 'r') as file:
            for line in file.readlines():
                files.append(line.strip())
        return files
    
    def __len__(self):
        return len(self.file_list)

    def _load_frames(self, file):
        file_path = os.path.join(self.data_dir, file)
        frame_data = np.load(file_path)
        tensor = torch.from_numpy(frame_data) / 255
        ## 1, h, w, t -> t, c, h, w
        tensor = tensor.permute(3, 0, 1, 2)
        return tensor


    def __getitem__(self, index):
        file = self.file_list[index]
        frame_data = self._load_frames(file)
        packed_results = dict()

        select_frame_idx = random.randint(0, self.total_length - 1)
        packed_results['inputs'] = frame_data[select_frame_idx]
        packed_results['data_samples'] = frame_data[select_frame_idx]
        return packed_results