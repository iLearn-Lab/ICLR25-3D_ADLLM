import os
import json
import datetime
import functools

import torch
import torch.distributed as dist

import numpy as np
import random
import csv
import pandas as pd

def now():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M")[:-1]

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
"""
Distribution
"""

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
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

def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:  # non-distributed training
        rank = 0
        world_size = 1
    return rank, world_size

def update_cfg_for_dist(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", args.gpu)
        args.distributed = True
        print(
            "| distributed init (rank {}, world {})".format(
                args.rank, args.world_size
            ),
            flush=True,
        )
        setup_for_distributed(args.rank == 0)
    else:
        print("Not using distributed mode")
        args.distributed = False
        
    
    
def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples



def worker_init_fn(worker_id,seed):
    worker_seed = seed + get_rank() + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    # print("rank",get_rank())
    # print("rankseed",seed)
    # print("workid",worker_id)


def get_worker_init_fn(seed):
    return lambda worker_id: worker_init_fn(worker_id, seed)

def save_metrics_to_csv(file_path, metrics):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["task_name", "metric_name", "metric_value"])
        for task_name, task_metrics in metrics.items():
            for metric_name, metric_value in task_metrics.items():
                writer.writerow([task_name, metric_name, metric_value])



def append_metrics_to_csv(csv_file_path, metrics, task_eval, epoch):
    columns = ["dataname", "acc5_global_avg", "iou_global_avg", "pointAcc_global_avg", "pointPrecision_global_avg", "pointRecall_global_avg"]
    data = {col: metrics.get(col, None) for col in columns[1:]}  # Extract values for specified columns from the metrics dictionary
    data["dataname"] = task_eval.name + str(epoch)
    
    df = pd.DataFrame([data], columns=columns)
    
    if os.path.exists(csv_file_path):
        df.to_csv(csv_file_path, mode='a', header=False, index=False)  # Append data to existing file without writing headers
    else:
        df.to_csv(csv_file_path, index=False)

