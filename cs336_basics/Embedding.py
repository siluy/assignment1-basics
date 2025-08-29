import torch
import torch.nn as nn

class Embedding(nn.Module):
    """
    将 token ID 映射到密集向量的 embedding 层
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 创建嵌入矩阵，形状为(词汇表大小，嵌入维度)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        N(μ=0, σ^2=1) 截断于 [-3, 3]
        """
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播，查表
        Args: 
            token_ids: 形状为(...,)的长 int tensor，包含要查找的 token ID
        
        Returns:
            形状为(..., embedding_dim)的 embedding tensor
        """
        return self.weight[token_ids]
    
    def extra_repr(self) -> str:
        return f'{self.num_embeddings}, {self.embedding_dim}'