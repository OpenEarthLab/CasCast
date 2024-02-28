import functools
import torch
import numpy as np
import pandas as pd
from calendar import isleap
from datetime import timedelta
from torchvision import transforms
from torch.utils.data import Dataset


class RainDataset(Dataset):
    """
    Dataset that includes filtered and augmented rainfall frame derived from npy file.
    Args:
        cfg: config format transformed by config.py .
        memory_map: np.memmap, source of this dataset, expected shape = (T, H, W).
        rain_record: 1-d np.ndarray, record number of nonzero rainfall at each frame in memmap, expected length = T.
        rain_coverage: float, the accepted rainfall coverage of the first input frame
        in_step: int, the number of input frames.
        out_step: int, the number of output frames.
        parser: function
        normalizer: function
        transform: transforms.Compose, the function of data augmentation
    """
    def __init__(
        self,
        memory_map: np.memmap=None,
        rain_record: np.ndarray=None,
        rain_coverage: float=None,
        in_step: int=None,
        out_step: int=None,
        parser=None,
        normalizer=None,
        transform: transforms.Compose=None,
        ):
        assert memory_map is not None, "memory_map should not be None"
        assert 0 <= rain_coverage <=1, "rain_coverage must be included in [0, 1]"

        self.memory_map = memory_map
        self.rain_record = rain_record
        self.rain_coverage = rain_coverage
        self.in_step = in_step
        self.out_step = out_step
        self.parser = self.get_parser(parser)        
        self.normalizer = self.get_normalizer(normalizer)
        self.transform = self.get_transform(transform)

        self.nonzeros = self.memory_map.shape[1] * self.memory_map.shape[2] * self.rain_coverage
        self.target_indices = np.where(self.rain_record >= self.nonzeros)[0]
        np.random.shuffle(self.target_indices)

    def get_parser(self, parser):
        if parser is not None:
            return parser
        else:
            return False
    
    def get_normalizer(self, normalizer):
        if normalizer is not None:
            return normalizer
        else:
            return False
        
    def get_transform(self, transform):
        if transform is not None:
            return transform
        else:
            return transforms.Compose([
                transforms.ToTensor()
                ])

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, index):
        map_index = self.target_indices[index]
        
        if map_index + self.in_step + self.out_step >= self.memory_map.shape[0]:
            img = self.memory_map[-self.in_step-self.out_step:]
        else:
            img = self.memory_map[map_index: map_index+self.in_step+self.out_step]

        if not isinstance(img, np.ndarray):
            img = np.array(img)
        img = img.transpose(1, 2, 0)
        img = self.transform(img)
        img = self.parser(img) if self.parser else img
        img = self.normalizer(img) if self.normalizer else img
        x = img[:self.in_step, ...]
        y = img[self.in_step:, ...]

        if index == len(self.target_indices) - 1:
            np.random.shuffle(self.target_indices)

        return (x, y)

   

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    #array = np.random.randint(low=0, high=3600, size=(10000, 512, 512)).astype(np.int16)
    #np.save("test.npy", array)
    memmap = np.memmap("test.npy", shape=(10000, 512, 512), dtype=np.int16)
    record = []
    for m in memmap:
        record.append((m>2500).sum())
    record = np.array(record)
    dataset = RainDataset(
        memory_map=memmap,
        rain_record=record,
        rain_coverage=0.2,
        in_step=4,
        out_step=6
        )
    for i in range(5):
        print(dataset[3])
        break
