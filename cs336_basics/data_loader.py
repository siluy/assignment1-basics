import torch
import numpy as np

def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从一个大的、一维的 token 数组中随机生成一个 batch 的输入 x 和输出 y

    Args:
        data: 包含所有 token ID 的 numpy 数组
        batch_size: 每个批次的序列数量
        context_length: 每个序列的长度
        device: 要将张量放置的设备，cuda:0或者cpu

    Returns:
        一个元组(x, y)，其中 x 和 y 都是 torch.Tensor
    """
    # 1. 确定随机起始点的有效范围，最后一个可能的起始点必须保证 y 的最后一个元素 data[i+1+context_length-1] 不会越界
    max_start_index = len(data) - context_length - 1
    # 2. 随机生成 batch_size 个起始索引
    start_indices = torch.randint(0, max_start_index + 1, (batch_size,))
    # 3. 切片创建 x 和 y 的序列列表
    x_list = [torch.from_numpy((data[i : i + context_length]).astype(np.int64)) for i in start_indices]
    y_list = [torch.from_numpy((data[i+1 : i + 1 + context_length]).astype(np.int64)) for i in start_indices]
    # 4. 将序列列表统一成一个批处理张量
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    # 5. 移动到指定设备
    x, y = x.to(device), y.to(device)

    return x, y