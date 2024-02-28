import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
try:
    from petrel_client.client import Client
except:
    pass
import io



class sevir_pretrain(Dataset):
    def __init__(self, split, data_dir='radar:s3://sevir_pretrain_data', **kwargs):
        super().__init__()
        self.file_list = self._init_file_list(split)
        self.data_dir = data_dir
        ## sproject client ##
        self.client = Client("~/petreloss.conf")

    def _init_file_list(self, split):
        if split == 'train':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/train.txt'
        elif split == 'valid':
            split = 'val'
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/val.txt'
        elif split == 'test':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/test.txt'
        files = []
        with open(f'{txt_path}', 'r') as file:
            for line in file.readlines():
                line = line.strip()
                for i in range(0, 49):
                    orig_name = line.split('/')[-1][:-4]
                    file_name = split + '/' + orig_name + '_' + str(i) + '.npy'
                    files.append(file_name)
        return files
    
    def __len__(self):
        return len(self.file_list)

    def _load_frames(self, file):
        file_path = os.path.join(self.data_dir, file)
        with io.BytesIO(self.client.get(file_path)) as f:
            frame_data = np.load(f)
        tensor = torch.from_numpy(frame_data) / 255
        ## c, h, w
        return tensor

    def __getitem__(self, index):
        file = self.file_list[index]
        frame_data = self._load_frames(file)
        packed_results = dict()
        packed_results['inputs'] = frame_data
        packed_results['data_samples'] = frame_data
        return packed_results

if __name__ == "__main__":
    dataset = sevir_pretrain('valid')
    print(len(dataset))

    import time
    st_time = time.time()
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        ed_time = time.time()
        print((ed_time - st_time)/(i+1))
        print(data['inputs'].shape)
        print(data['data_samples'].shape)

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u sevir_pretrain.py ###