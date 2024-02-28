# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from ipaddress import ip_address
import torch.distributed as dist
import torch
import os
import time
import datetime
from collections import defaultdict, deque
import numpy as np
import random
from typing import Any, List, Tuple, Union
import re
from megatron_utils import mpu
import subprocess

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=100, fmt=None, sync=True):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f}, {value:.4f})"  #fmt = "{median:.10f} ({global_avg:.10f})" 
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.var = 0.0
        self.fmt = fmt
        self.sync = sync


    ####### 3 sigma skip ############
    # def update(self, value, n=1):
    #     """
    #     skip value beyond 3sigma
    #     """
    #     prev_mean = self.total / (self.count + 1e-12)
    #     if self.var==0 or (value - prev_mean) < 3 * self.var**0.5: 
    #         self.deque.append(value)
    #         self.count += n
    #         self.total += value * n
    #         cur_mean = self.total / (self.count + 1e-12)
    #         self.var = (self.count - 1)/self.count * self.var + (value - prev_mean) * (value - cur_mean) / (self.count + 1e-12)
    #     else:
    #         ## if beyond 3sigma, only update var
    #         self.deque.append(value)
    #         cur_mean = (self.total + n*value) / (n + self.count + 1e-12)
    #         self.var = self.count/(self.count + 1) * self.var + (value - prev_mean) * (value - cur_mean) / (self.count - 1 + 1e-12)


    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if (not is_dist_avail_and_initialized()) or (not self.sync):
            return
        t = torch.tensor([self.count, self.total, self.var], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t / get_world_size()
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
        self.var = t[2]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        self.synchronize_between_processes()
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        median      = self.median
        avg         = self.avg
        max         = self.max
        value       = self.value
        if is_dist_avail_and_initialized() and self.sync:
            t = torch.tensor([median, avg, max, value], dtype=torch.float64, device='cuda')
            dist.barrier()
            dist.all_reduce(t)
            t = t / get_world_size()
            t = t.tolist()
            median, avg, max, value = t
        # else:
        #     median      = self.median
        #     avg         = self.avg
        #     max         = self.max
        #     value       = self.value

        return self.fmt.format(
            median=median,
            avg=avg,
            global_avg=self.global_avg,
            max=max,
            value=value)


class MetricLogger(object):
    def __init__(self, delimiter="  ", sync=True):
        # self.meters = defaultdict(SmoothedValue)(sync=sync)
        self.meters = defaultdict(lambda:SmoothedValue(sync=sync))
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    


    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))



def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


# def init_distributed_mode(args):
#     if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.local_rank = int(os.environ['LOCAL_RANK'])
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.local_rank = args.rank % torch.cuda.device_count()
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return

#     args.distributed = True

#     torch.cuda.set_device(args.local_rank)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}'.format(
#         args.rank, args.init_method), flush=True)
#     torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.init_method,
#                                          world_size=args.world_size, rank=args.rank)
#     torch.distributed.barrier()
#     setup_for_distributed(args.rank == 0)

def get_ip(ip_list):
    if "," in ip_list:
        ip_list = ip_list.split(',')[0]
    if "[" in ip_list:
        ipbefore_4, ip4 = ip_list.split('[')
        ip4 = re.findall(r"\d+", ip4)[0]
        ip1, ip2, ip3 = ipbefore_4.split('-')[-4:-1]
    else:
        ip1,ip2,ip3,ip4 = ip_list.split('-')[-4:]
    ip_addr = "tcp://" + ".".join([ip1, ip2, ip3, ip4]) + ":"
    return ip_addr
    


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        ip_addr = get_ip(os.environ['SLURM_STEP_NODELIST'])
        port = int(os.environ['SLURM_SRUN_COMM_PORT'])
        # args.init_method = ip_addr + str(port)
        args.init_method = ip_addr + args.init_method.split(":")[-1]
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True


    # addr = subprocess.getoutput(
    #         "scontrol show hostname {} | head -n1".format(os.environ["SLURM_NODELIST"])
    #     )
    # os.environ["MASTER_PORT"] = args.init_method.split(":")[-1]
    # os.environ["MASTER_ADDR"] = addr
    # os.environ["WORLD_SIZE"] = str(args.world_size)
    # os.environ["RANK"] = str(args.rank)

    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, local_rank {}): {}'.format(
        args.rank, args.local_rank, args.init_method), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    mpu.initialize_model_parallel(args.tensor_model_parallel_size)
    setup_for_distributed(args.rank == 0)
    print(f'> initialized tensor model parallel with size '
            f'{mpu.get_tensor_model_parallel_world_size()}')


def DistributedParallel_Model(model, gpu_num):
    if is_dist_avail_and_initialized():
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        if device == torch.device('cpu'):
            raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
        model.to(device)
        for key in model.model:
            # model.model[key].to(device)
            # ddp_sub_model = torch.nn.parallel.DistributedDataParallel(model.model[key], device_ids=[gpu_num], 
            #                                                         output_device=gpu_num, process_group=mpu.get_data_parallel_group(),
            #                                                         find_unused_parameters=True)
            ddp_sub_model = torch.nn.parallel.DistributedDataParallel(model.model[key], device_ids=[gpu_num], 
                                                                    output_device=gpu_num, process_group=mpu.get_data_parallel_group(),
                                                                    find_unused_parameters=True)
            model.model[key] = ddp_sub_model
        
        # model.to(device)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_num])
        # model_without_ddp = model.module
    else:
        device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        # for key in model.model:
        #     model.model[key].to(device)

    return model


class Dict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
    # __setattr__ = dict.__setitem__
    # __getattr__ = dict.__getitem__

def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        # if tensor.is_floating_point():
        #     tensor = nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=mpu.get_data_parallel_src_rank(),
                                group=mpu.get_data_parallel_group())
        # print(fullname, tensor.sum(), other.sum())
        assert (tensor == other).all(), fullname


def collate_fn(batch):
    batch = list(zip(*batch))
    array_seq = np.stack(batch[0])
    origin_array_seq = np.stack(batch[1])
    date_time_seq = np.stack(batch[2])
    # samples = NestedTensor(img, mask)
    # targets = tensor_dict_from_dict_list(batch[2])
    return tuple([array_seq, origin_array_seq, date_time_seq])


class data_prefetcher():
    def __init__(self, loader, device):
        self.length = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.next_array_seq, self.next_origin_array_seq, self.next_date_time_seq = next(self.loader)
        except StopIteration:
            self.next_array_seq = self.next_origin_array_seq = self.next_date_time_seq = None
            return
        with torch.cuda.stream(self.stream):
            self.next_array_seq = self.next_array_seq.to(self.device, non_blocking=True)
            self.next_origin_array_seq = self.next_origin_array_seq.to(self.device, non_blocking=True)
            self.next_date_time_seq = self.next_date_time_seq.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        array_seq, origin_array_seq, date_time_seq = self.next_array_seq, self.next_origin_array_seq, self.next_date_time_seq
        self.preload()
        return array_seq, origin_array_seq, date_time_seq

    def __next__(self):
        array_seq, origin_array_seq, date_time_seq = self.next()
        if array_seq == None:
            raise StopIteration
        return array_seq, origin_array_seq, date_time_seq

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

