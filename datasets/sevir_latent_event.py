import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
try:
    from petrel_client.client import Client
except:
    pass
import io



EVENTS = {
    '': 0,
    'Flood': 1,
    'Tornado': 2,
    'Hail': 3,
    'Heavy Rain': 4,
    'Funnel Cloud': 5,
    'Lightning': 6,
    'Flash Flood': 7,
    'Thunderstorm Wind': 8,
}



def get_sevir_latent_event_dataset( split, input_length=13, pred_length=12, data_dir='/mnt/data/oss_beijing/video_prediction_dataset/sevir/sevir', base_freq='5min', height=384, width=384, **kwargs):
    if data_dir == '/mnt/data/oss_beijing/video_prediction_dataset/sevir/sevir':
        return sevir_3090(split, input_length=input_length, pred_length=pred_length, data_dir=data_dir, base_freq=base_freq, height=height, width=width, **kwargs)
    elif data_dir == 'radar:s3://sevir_latent':
        return sevir_latent_event_sproject(split, input_length=input_length, pred_length=pred_length, data_dir=data_dir, base_freq=base_freq, height=height, width=width, **kwargs)
    else:
        raise NotImplementedError



class sevir_latent_event_sproject(Dataset):
    def __init__(self, split, input_length=13, pred_length=12, data_dir='radar:s3://sevir_latent', base_freq='5min', height=384, width=384, coarse_model='earthformer', 
                 latent_size='48x48x4', **kwargs):
        super().__init__()
        assert input_length == 13, pred_length==12
        self.input_length = 13
        self.pred_length = 12

        self.latent_size = latent_size
        self.coarse_model = coarse_model

        self.event_dict = self._init_event_dict(split)

        self.file_list = self._init_file_list(split)
        self.data_dir = data_dir

        self.event_type_list = self._get_event_type(self.file_list, self.event_dict)
        assert len(self.event_type_list) == len(self.file_list)
        ## sproject client ##
        self.client = Client("~/petreloss.conf")

    def _get_name_and_index(self, file):
        ceph_file_name = file.split('/')[-1]
        file_name = ceph_file_name.split('.')[0]
        radar_type, year, name = file_name.split('-')
        sevir_file_name = os.path.join(radar_type, year, name) + '.h5'
        file_index = ceph_file_name.split('.')[-2].split('-')[-2]
        return sevir_file_name, file_index
    
    def _get_event_type(self, file_list, event_dict):
        event_type_list = []
        for file in file_list:
            sevir_file_name, file_index = self._get_name_and_index(file)
            event_type = event_dict[sevir_file_name][file_index]
            event_type_list.append(event_type)
        return event_type_list
    
    def _init_event_dict(self, split):
        import pickle
        if split == 'train':
            pkl_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/train_event_dict.pkl'
        elif split == 'valid':
            pkl_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/val_event_dict.pkl'
        elif split == 'test':
            pkl_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/test_event_dict.pkl'
        with open(pkl_path, 'rb') as f:
            event_dict = pickle.load(f)
        return event_dict

    def _init_file_list(self, split):
        if split == 'train':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/train_2h_event.txt'
        elif split == 'valid':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/val_2h_event.txt'
        elif split == 'test':
            txt_path = '/mnt/cache/gongjunchao/workdir/radar_forecasting/datasets/sevir_list/test_2h_event.txt'
        files = []
        with open(f'{txt_path}', 'r') as file:
            for line in file.readlines():
                files.append(line.strip())
        return files
    
    def __len__(self):
        return len(self.file_list)
    
    # def _load_frames(self, file):
    #     interal_id = int(file[-5])
    #     file_path = file[:-6] + '.npy'
    #     import pdb; pdb.set_trace()
    #     with io.BytesIO(self.client.get(file_path)) as f:
    #         frame_data = np.load(f)
    #     import pdb; pdb.set_trace()

    def _load_latent_frames(self, file, datasource='gt'):
        file_path = 'radar:' + file
        sevir_file_name = file.split('/')[-1]
        split = file.split('/')[-2]
        file_path = os.path.join(self.data_dir, self.latent_size, datasource, split, sevir_file_name)
        with io.BytesIO(self.client.get(file_path)) as f:
            frame_data = np.load(f)
        ## t, c, h, w ##
        tensor = torch.from_numpy(frame_data)
        return tensor

    def _load_frames(self, file):
        file_path = 'radar:' + file
        with io.BytesIO(self.client.get(file_path)) as f:
            frame_data = np.load(f)
        tensor = torch.from_numpy(frame_data) / 255
        ## 1, h, w, t -> t, c, h, w
        tensor = tensor.permute(3, 0, 1, 2)
        return tensor

    def __getitem__(self, index):
        file = self.file_list[index]
        event_type = EVENTS[self.event_type_list[index]]
        gt_data = self._load_frames(file)[self.input_length:]
        gt_latent_data = self._load_latent_frames(file)
        coarse_latent_data = self._load_latent_frames(file, datasource=self.coarse_model)
        packed_results = dict()
        packed_results['inputs'] = coarse_latent_data
        packed_results['data_samples'] = {'latent':gt_latent_data, 'original':gt_data, 'event_type':event_type}
        return packed_results

if __name__ == "__main__":
    dataset = get_sevir_latent_event_dataset(split='test', input_length=13, pred_length=12, data_dir='radar:s3://sevir_latent', base_freq='5min', height=384, width=384)
    print(len(dataset))

    import time
    st_time = time.time()
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        ed_time = time.time()
        print((ed_time - st_time)/(i+1))
        # print(data['inputs'].shape)
        # print(data['data_samples']['original'].shape)
        # print(data['data_samples']['latent'].shape)

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u sevir_latent_event.py ###