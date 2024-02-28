import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
try:
    from petrel_client.client import Client
except:
    pass
import io



class meteonet_latent(Dataset):
    def __init__(self, split, input_length=12, pred_length=12, data_dir='radar:s3://meteonet_latent/50x50x4', **kwargs):
        super().__init__()
        assert input_length == 12, pred_length==12
        self.input_length = 12
        self.pred_length = 12
        self.total_length = self.input_length + self.pred_length

        self.file_list = self._init_file_list(split)

        self.data_dir = data_dir
        self.coarse_model = kwargs.get('coarse_model', 'SimVP')
        self.client = Client("~/petreloss.conf")

    def _init_file_list(self, split):
        if split == 'train':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/meteonet/train_2h.txt'
        elif split == 'valid':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/meteonet/valid_2h.txt'
        elif split == 'test':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/meteonet/test_2h.txt'
        files = []
        with open(f'{txt_path}', 'r') as file:
            for line in file.readlines():
                files.append(line.strip())
        return files
    
    def __len__(self):
        return len(self.file_list)
    
    def _load_frames(self, file):
        file_path = os.path.join('radar:s3://meteonet_data/24Frames', file)
        with io.BytesIO(self.client.get(file_path)) as f:
            frame_data = np.load(f)
        tensor = torch.from_numpy(frame_data) / 70 ##TODO: get max
        ## 1, h, w, t -> t, c, h, w
        tensor = tensor.unsqueeze(dim=1)
        return tensor

    def _load_latent_frames(self, file, datasource='gt'):
        file_path = os.path.join(self.data_dir, datasource, file)
        with io.BytesIO(self.client.get(file_path)) as f:
            frame_data = np.load(f)
        tensor = torch.from_numpy(frame_data) ## t, c, h, w
        return tensor
    
    def __getitem__(self, index):
        file = self.file_list[index]
        gt_frame_data = self._load_frames(file)[self.input_length:]
        gt_latent_data = self._load_latent_frames(file, datasource='gt')
        coarse_latent_data = self._load_latent_frames(file, datasource=self.coarse_model)
        packed_results = dict()
        packed_results['inputs'] = coarse_latent_data #{'coarse_pred':coarse_latent_data, 'prev_inp':gt_latent_data[:self.input_length]}
        packed_results['data_samples'] = {'latent':gt_latent_data[self.input_length:], 'original':gt_frame_data}
        return packed_results
    


if __name__ == "__main__":
    dataset = meteonet_latent(split='train', input_length=12, pred_length=12,
                            data_dir='radar:s3://meteonet_latent/50x50x4', coarse_model='SimVP')
    print(len(dataset))

    import time
    st_time = time.time()
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        ed_time = time.time()

        print((ed_time - st_time)/(i+1))
        print(data['inputs'].shape)

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u meteonet_latent.py ###
        
