from collections import namedtuple
import os
import numpy as np
import pandas as pd
import dask.array as da
from tqdm import tqdm
from collections import namedtuple
from torchvision import transforms
import numba

from networks.DGMR.utils.dataset import RainDataset
from networks.DGMR.utils.data_parser import *

import warnings
warnings.filterwarnings("ignore")
import time

@numba.jit()
def count_nonzeros(memmap):
    rain_record = []
    for map in memmap:
        rain_record.append((map>0).sum())
    return np.array(rain_record)

def get_memmap(npy_path, dtype=None, shape=None, get_nonzeros=True):
    if isinstance(npy_path, list):
        memmap_list = []
        rain_record = []
        os.system("echo")

        for npy, sp in zip(npy_path, shape):
            memmap = np.memmap(npy, dtype=dtype, shape=tuple(sp), order='C')
            memmap_list.append(memmap)
            if get_nonzeros:
                os.system("echo Counting nonzeros may take a while...")
                start_time = time.time()
                rain_record.append(count_nonzeros(memmap))
                total_time = int(time.time() - start_time)
                os.system(f"echo Finish counting! {total_time} seconds are taken.")

        rain_record = np.concatenate(rain_record, axis=0) if get_nonzeros else None
        final_memmap = da.concatenate(memmap_list, axis=0)
        os.system(f"echo Total length of memory-map : {final_memmap.shape[0]}")

        return final_memmap, rain_record

    elif isinstance(npy_path, str):
        memmap = np.memmap(npy_path, dtype=dtype, shape=tuple(shape))

        if get_nonzeros:
            os.system("echo Counting nonzeros may take a while...")
            start_time = time.time()
            rain_record = count_nonzeros(memmap)
            total_time = int(time.time() - start_time)
            os.system(f"echo Finish counting! {total_time} seconds are taken.")
        else:
            rain_record = None
        os.system(f"echo Total length of memory-map : {memmap.shape[0]}")

        return memmap, rain_record

def save_rain_record(rain_record, npy_path):
    rain_record_df = pd.DataFrame(rain_record, columns=["nonzeros"])
    if isinstance(npy_path, str):
        label = npy_path.split("/")[-1].split(".")[0]
    elif isinstance(npy_path, list):
        label = ""
        for npy in npy_path:
            label += npy.split("/")[-1].split(".")[0]
            label += "_"
        label = label[:-1]
    rain_record_df.to_csv(f"rain_record_{label}.csv", index=False)
    return

def get_arguments(
    params: namedtuple=None
    ):
    args = {}
    for k in params._asdict().keys():
        args[k.lower()] = params._asdict()[k]
    return args

def make_dataset(
    cfg,
    mode: str=None
    ):
    npy_path = cfg.SETTINGS.DATA_PATH
    memory_map, rain_record = get_memmap(
        npy_path,
        dtype=getattr(np, cfg.SETTINGS.DATA_TYPE),
        shape=cfg.SETTINGS.DATA_SHAPE,
        get_nonzeros=True if cfg.SETTINGS.RAIN_RECORD_PATH is None else False
        )
    
    if cfg.SETTINGS.RAIN_RECORD_PATH is not None:
        rain_record = pd.read_csv(cfg.SETTINGS.RAIN_RECORD_PATH)
    else:
        save_rain_record(rain_record=rain_record, npy_path=npy_path)
    
    parser = None
    if cfg.PARAMS.PARSER.FUNCTION is not None:
        parser_name = cfg.PARAMS.PARSER.FUNCTION
        parser_args = get_arguments(cfg.PARAMS.PARSER.PARAMS)
        parser = lambda x: eval(parser_name)(x, **parser_args)
        os.system(f"echo Parser: {parser_name}")

    normalizer = None
    if cfg.PARAMS.NORMALIZER.FUNCTION is not None:
        normalizer_name = cfg.PARAMS.NORMALIZER.FUNCTION
        normalizer_args = get_arguments(cfg.PARAMS.NORMALIZER.PARAMS)
        normalizer = lambda x: eval(normalizer_name)(x, **normalizer_args)
        os.system(f"echo Normalizer: {normalizer_name}")
    
    if mode == "train":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(cfg.PARAMS.INPUT_SIZE)
        ])
    elif mode == "test" or mode == "val":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(cfg.PARAMS.INPUT_SIZE)
        ])
    
    dataset = RainDataset(
        memory_map=memory_map,
        rain_record=rain_record,
        rain_coverage=cfg.PARAMS.COVERAGE,
        in_step=cfg.PARAMS.INPUT_FRAME,
        out_step=cfg.PARAMS.OUTPUT_FRAME,
        parser=parser,
        normalizer=normalizer,
        transform=transform
    )
    return dataset