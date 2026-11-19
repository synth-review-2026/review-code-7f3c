"""
Helpers for distributed training (single GPU version for Windows).
"""

import os
import socket
import torch as th
import torch.distributed as dist


def setup_dist(devices=None):
    """
    Setup distributed training - single GPU mode.
    """
    # 设置单卡环境变量
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"

    # 不初始化分布式，直接跳过
    print("🖥️ Running in single-GPU mode (distributed disabled)")
    return


def dev():
    """
    Get the device to use.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize parameters (no-op for single GPU).
    """
    return


def get_world_size():
    return 1


def get_rank():
    return 0


def all_gather(data):
    """
    All gather (no-op for single GPU).
    """
    return [data]