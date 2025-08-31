import torch
import torch.nn as nn

from MultiHeadSelfAttention import MultiHeadSelfAttention
from RMSNorm import RMSNorm
from SwiGLU import PositionWiseFeedForward

from Embedding import Embedding
from Linear import Linear

class TransformerBlock(nn.Module):
    """
    pre-norm 的 transformer 实现
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, max_seq_len: int = 2048, theta: float = 10000.0, device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype
        )
        self.ffn = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        )
        # self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        # self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 tensor，(batch_size, seq_len, d_model)
            positions: 位置索引 tensor，(batch_size, seq_len)

        Returns:
            输出 tensor
        """
        # 第一层，多头自注意力
        # pre-norm -> Function -> add
        # attn_output = self.attn(self.norm1(x), positions=positions)
        attn_output = self.attn(x, positions=positions) # without RMSnorm
        # residual add
        x = x + attn_output

        # 第二层，ffn
        # ffn_output = self.ffn(self.norm2(x))
        ffn_output = self.ffn(x) # without RMSnorm
        # residual add
        x = x + ffn_output

        return x
    
class TransformerLM(nn.Module):
    """
    完整的 pre-norm 架构 transformer 搭建
    """
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, # i.e. max_seq_len
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.context_length = context_length
        # 1. 实例化所以层
        # 词嵌入层
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        # 堆叠 num_layers 个 TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        # 归一化层
        # self.norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        # 输出投影层(LM Head)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            token_ids: 输入的 token ID tensor, (batch_size, seq_len)

        Returns:
            logits tensor, (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        # 1. 创建 RoPE 的位置索引
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        # 2. token ID -> 词向量
        x = self.token_embeddings(token_ids)
        # 3. 通过每一个 Transformer Block
        for block in self.blocks:
            x = block(x, positions)
        # 4. 通过归一化层
        # x = self.norm_final(x)
        # 5. 通过输出投影层得到 logits
        logits = self.lm_head(x)

        return logits