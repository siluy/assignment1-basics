import torch
import torch.nn as nn
import math

from Linear import Linear
from RoPE import RotaryPositionalEmbedding
from attention_functions import scaled_dot_product_attention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = True, max_seq_len: int = 2048, theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_heads = d_model // num_heads
        self.use_rope = use_rope

        # 1. 创建 Q K V 和输出的线性投影层
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if self.use_rope:
        # 在 Attention 模块内应用 RoPE，其维度是 d_head，因为它会作用于每个 head
            self.rope = RotaryPositionalEmbedding(d_k=self.d_heads, max_seq_len=max_seq_len, theta=theta)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args: 
            x: 输入的 tensor，形状(batch_size, seq_len, d_model)
            positions: 位置索引 tensor，形状(batch_size, seq_len)

        Returns:
            输出 tensor
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 分头: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_heads).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_heads).transpose(1, 2)

        if self.use_rope:
            # 3. apply RoPE
            if positions.dim() == 1:
                positions = positions.unsqueeze(0) # -> (1, seq_len)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        # 4. Causal Mask
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1) == False # torch.triu 创建一个上三角阵，遮盖后面的输入

        # 5. 并行注意力计算
        attention_output = scaled_dot_product_attention(q, k, v, mask=mask)

        # 6. 合并头，输出投影
        # (batch, num_heads, seq_len, d_head) -> (batch, seq_len, d_model)
        output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(output)

        return output