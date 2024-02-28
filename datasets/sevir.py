import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import torch
try:
    from petrel_client.client import Client
except:
    pass
import io



# SEVIR Dataset constants
SEVIR_DATA_TYPES = ['vis', 'ir069', 'ir107', 'vil', 'lght']
SEVIR_RAW_DTYPES = {'vis': np.int16,
                    'ir069': np.int16,
                    'ir107': np.int16,
                    'vil': np.uint8,
                    'lght': np.int16}
LIGHTING_FRAME_TIMES = np.arange(- 120.0, 125.0, 5) * 60
SEVIR_DATA_SHAPE = {'lght': (48, 48), }
PREPROCESS_SCALE_SEVIR = {'vis': 1,  # Not utilized in original paper
                          'ir069': 1 / 1174.68,
                          'ir107': 1 / 2562.43,
                          'vil': 1 / 47.54,
                          'lght': 1 / 0.60517}
PREPROCESS_OFFSET_SEVIR = {'vis': 0,  # Not utilized in original paper
                           'ir069': 3683.58,
                           'ir107': 1552.80,
                           'vil': - 33.44,
                           'lght': - 0.02990}
PREPROCESS_SCALE_01 = {'vis': 1,
                       'ir069': 1,
                       'ir107': 1,
                       'vil': 1 / 255,  # currently the only one implemented
                       'lght': 1}
PREPROCESS_OFFSET_01 = {'vis': 0,
                        'ir069': 0,
                        'ir107': 0,
                        'vil': 0,  # currently the only one implemented
                        'lght': 0}

def get_sevir_dataset( split, input_length=13, pred_length=12, data_dir='/mnt/data/oss_beijing/video_prediction_dataset/sevir/sevir', base_freq='5min', height=384, width=384, **kwargs):
    if data_dir == '/mnt/data/oss_beijing/video_prediction_dataset/sevir/sevir':
        return sevir_3090(split, input_length=input_length, pred_length=pred_length, data_dir=data_dir, base_freq=base_freq, height=height, width=width, **kwargs)
    elif data_dir == 'radar:s3://weather_radar_datasets/sevir':
        return sevir_sproject(split, input_length=input_length, pred_length=pred_length, data_dir=data_dir, base_freq=base_freq, height=height, width=width, **kwargs)
    else:
        raise NotImplementedError



class sevir_sproject(Dataset):
    def __init__(self, split, input_length=13, pred_length=12, data_dir='radar:s3://weather_radar_datasets/sevir', base_freq='5min', height=384, width=384, **kwargs):
        super().__init__()
        assert input_length == 13, pred_length==12
        self.input_length = 13
        self.pred_length = 12

        self.file_list = self._init_file_list(split)
        
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
    
    # def _load_frames(self, file):
    #     interal_id = int(file[-5])
    #     file_path = file[:-6] + '.npy'
    #     import pdb; pdb.set_trace()
    #     with io.BytesIO(self.client.get(file_path)) as f:
    #         frame_data = np.load(f)
    #     import pdb; pdb.set_trace()

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
        frame_data = self._load_frames(file)
        packed_results = dict()
        packed_results['inputs'] = frame_data[:self.input_length]
        packed_results['data_samples'] = frame_data[self.input_length:self.input_length+self.pred_length]
        return packed_results



class sevir_3090(Dataset):
    def __init__(self, split, input_length=13, pred_length=12, data_dir='/mnt/data/oss_beijing/video_prediction_dataset/sevir/sevir', base_freq='5min', height=384, width=384, **kwargs):
        super().__init__()

        self.data_dir = data_dir
        
        self.input_length = input_length
        self.pred_length = pred_length
        self.total_length = self.input_length + self.pred_length
        self.file_list = self._init_file_list(split)


    def _init_file_list(self, split):
        ## split each npy file into different segments ##
        segments = [i for i in range(0, 49) if self.total_length + i*self.pred_length <= 49]
        if split == 'train':
            _dir = os.path.join(self.data_dir, 'train')
            npy_file_list = os.listdir(_dir) ## (1, 384, 384, 49) 
        elif split == 'valid':
            _dir = os.path.join(self.data_dir, 'val')
            npy_file_list = os.listdir(_dir)
        elif split == 'test':
            _dir = os.path.join(self.data_dir, 'test')
            npy_file_list = os.listdir(_dir)
        file_list = [os.path.join(_dir, f'{npy_file}:{seg}') for npy_file in npy_file_list for seg in segments]
        return file_list

    def __len__(self):
        return len(self.file_list)
    
    def _load_frames(self, file):
        file, seg = file.split(':')
        seg = int(seg)
        # if seg == 2:
        #     import pdb; pdb.set_trace()
        start_frame_id = seg*self.pred_length
        frame_data = np.load(file)[..., start_frame_id:start_frame_id + self.total_length]
        tensor = torch.from_numpy(frame_data) / 255 
        ## 1, h, w, t -> t, c, h, w
        tensor = tensor.permute(3, 0, 1, 2)
        return tensor

    # def _load_frames(self, datetime_clips, total_length):
    #     paths = [convert_datetime_to_filepath(datetime_clips[i]) for i in range(len(datetime_clips))]
    #     read_storage = []
    #     for i in range(len(paths)):
    #         read_storage.append(self.loader(paths[i]))
    #     frame_dat = np.array(read_storage)
    #     frame_dat = frame_dat * self._exclude_mask
    #     data_batch = torch.from_numpy(frame_dat) / 255.0
    #     return data_batch

    # def rainfall_to_pixel(self, rainfall_intensity, a=58.53, b=1.56):
    #     # dBZ = 10b log(R) +10log(a)
    #     dBR = np.log10(rainfall_intensity) * 10.0
    #     dBZ = dBR * b + 10.0 * np.log10(a)
    #     pixel_vals = (dBZ + 10.0) / 70.0
    #     return pixel_vals
 
    def __getitem__(self, idx):
        file = self.file_list[idx]
        frame_data = self._load_frames(file)
        packed_results = dict()
        packed_results['inputs'] = frame_data[:self.input_length]
        packed_results['data_samples'] = frame_data[self.input_length:self.input_length+self.pred_length]
        return packed_results

if __name__ == "__main__":
    dataset = get_sevir_dataset(split='train', input_length=13, pred_length=12, data_dir='radar:s3://weather_radar_datasets/sevir', base_freq='5min', height=384, width=384)
    print(len(dataset))

    import time
    st_time = time.time()
    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        ed_time = time.time()
        print((ed_time - st_time)/(i+1))
        print(data['inputs'].shape)
        print(data['data_samples'].shape)

### srun -p ai4earth --kill-on-bad-exit=1 --quotatype=reserved --gres=gpu:0 python -u sevir.py ###