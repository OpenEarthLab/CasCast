import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
try:
    from petrel_client.client import Client
except:
    pass
import io




def get_sevir_field_cond_dataset( split, input_length=13, pred_length=12, data_dir='/mnt/data/oss_beijing/video_prediction_dataset/sevir/sevir', base_freq='5min', height=384, width=384, **kwargs):
    if data_dir == '/mnt/data/oss_beijing/video_prediction_dataset/sevir/sevir':
        raise NotImplementedError
        return sevir_3090(split, input_length=input_length, pred_length=pred_length, data_dir=data_dir, base_freq=base_freq, height=height, width=width, **kwargs)
    elif data_dir == 'radar:s3://weather_radar_datasets/sevir':
        return sevir_field_cond_sproject(split, input_length=input_length, pred_length=pred_length, data_dir=data_dir, base_freq=base_freq, height=height, width=width, **kwargs)
    else:
        raise NotImplementedError



class sevir_field_cond_sproject(Dataset):
    def __init__(self, split, input_length=13, pred_length=12, data_dir='radar:s3://weather_radar_datasets/sevir', base_freq='5min', height=384, width=384, **kwargs):
        super().__init__()
        assert input_length == 13, pred_length==12
        self.input_length = 13
        self.pred_length = 12

        self.file_list = self._init_file_list(split)
        
        self.era5_field_mean = np.load('/mnt/cache/gongjunchao/workdir/radar_forecasting/preprocess/field/field_mean.npy')
        self.era5_field_std =  np.load('/mnt/cache/gongjunchao/workdir/radar_forecasting/preprocess/field/field_std.npy')

        ## sproject client ##
        self.client = Client("~/petreloss.conf")



    def _init_file_list(self, split):
        if split == 'train':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/train_2h.txt'
        elif split == 'valid':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/val_2h.txt'
        elif split == 'test':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/test_2h.txt'
        files = []
        with open(f'{txt_path}', 'r') as file:
            for line in file.readlines():
                files.append(line.strip())
        return files
    
    def __len__(self):
        return len(self.file_list)
    

    def _load_frames(self, file):
        file_path = 'radar:' + file
        with io.BytesIO(self.client.get(file_path)) as f:
            frame_data = np.load(f)
        tensor = torch.from_numpy(frame_data) / 255
        ## 1, h, w, t -> t, c, h, w
        tensor = tensor.permute(3, 0, 1, 2)
        return tensor
    
    def _load_field(self, file):
        split = file.split('/')[-2][:-3]
        bucket_name = 'sevir_field_data'
        prefix = 'filed'
        vil_file = file.split('/')[-1]
        internal_id = vil_file[-5]
        field_file = prefix + '-' + vil_file[:-6] + '.npy'
        field_file_path = os.path.join(f'radar:s3://{bucket_name}', split, field_file)
        with io.BytesIO(self.client.get(field_file_path)) as f:
            field_data = np.load(f)
        ## clip the field data according to internal id ##
        field_data = field_data[int(internal_id):int(internal_id)+3]
        field_data = (field_data - self.era5_field_mean.reshape(1, -1, 1, 1)) / (self.era5_field_std.reshape(1, -1, 1, 1) + 1e-8)
        tensor = torch.from_numpy(field_data)
        return tensor

    def __getitem__(self, index):
        file = self.file_list[index]
        frame_data = self._load_frames(file)
        field_data = self._load_field(file)
        packed_results = dict()
        packed_results['inputs'] = {'radar':frame_data[:self.input_length], 'field':field_data[:self.input_length]}
        packed_results['data_samples'] = frame_data[self.input_length:self.input_length+self.pred_length]
        return packed_results


if __name__ == "__main__":
    dataset = get_sevir_field_cond_dataset(split='valid', input_length=13, pred_length=12, data_dir='radar:s3://weather_radar_datasets/sevir', base_freq='5min', height=384, width=384)
    print(len(dataset))

    import time
    st_time = time.time()
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        ed_time = time.time()
        print((ed_time - st_time)/(i+1))
        print(data['inputs']['radar'].shape)
        print(data['inputs']['field'].shape)
        print(data['data_samples'].shape)

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u sevir.py ###