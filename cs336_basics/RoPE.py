import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE
    """
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0, device=None, dtype=None):
        super().__init__()

        # 1. 计算旋转频率，freqs 的形状是(d_k / 2)
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))

        # 2. 计算所有可能位置的角度，t 的形状是(max_seq_len)
        t = torch.arange(max_seq_len, device=device)
        # angles 的形状是(max_seq_len, d_k / 2)
        angles = torch.outer(t, freqs)

        # 3. 计算并存储 sin，cos 入 buffer
        self.register_buffer("sin_cache", torch.sin(angles).to(dtype=dtype))
        self.register_buffer("cos_cache", torch.cos(angles).to(dtype=dtype))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        对 tensor x 使用 RoPE

        Args:
            x: 输入 tensor，形状为(..., seq_len, d_k)
            token_positions: 每个 token 的位置索引，形状为(..., seq_len)

        Returns:
            旋转后的 tensor
        """
        # 1. 根据输入位置从缓存获取对应的 sin cos
        # sin cos 的形状将和 token_positions 匹配，在最后增加一个维度
        # (batch, seq_len) -> (batch, seq_len, d_k / 2)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # 2. 将 x 的最后一维配对
        # (..., seq_len, d_k) -> (..., seq_len, d_k / 2, 2)
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)

        # 3. 旋转和分离
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # 4. 重新组合旋转后的对
        rotated_pairs = torch.stack((rotated_x1, rotated_x2), dim=-1)

        # 5. 恢复初始形状
        rotated_x = rotated_pairs.flatten(start_dim=-2)

        return rotated_x.to(x.dtype)