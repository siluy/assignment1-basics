import torch
import torch.nn as nn
import torch.optim as optim
from typing import BinaryIO, Union
import os

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO]
):
    """
    将模型，优化器和迭代步数的状态保存到一个 ckpt

    Args: 
        model: 待保存的 torch.nn.Module
        optimizer: 待保存的优化器
        iteration: 当前训练的迭代步数
        out: 输出的文件路径或对象
    """
    # 1. 将需要保存的状态打包进一个字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }

    torch.save(checkpoint, out)
    if isinstance(out, (str, os.PathLike)):
        print(f"ckpt 已保存到: {out}")

def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO],
    model: nn.Module,
    optimizer: optim.Optimizer 
) -> int:
    """
    从一个 ckpt 加载状态，恢复模型和优化器

    Args: 
        src: 检查点文件的路径或文件对象
        model: 需要恢复状态的模型
        optimizer: 需要恢复状态的优化器

    Returns:
        从 ckpt 恢复的迭代步数
    """
    checkpoint = torch.load(src, map_location='cpu') # 用cpu防止在没有gpu设备加载失败
    # 将加载的状态字典应用到模型和优化器上
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 提取并返回迭代步数
    iteration = checkpoint['iteration']

    if isinstance(src, (str, os.PathLike)):
        print(f"ckpt已从{src}加载，将从第{iteration}步继续训练")
    return iteration