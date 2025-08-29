import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    """
    Linear layer but has no bias to match morden LLM design
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 创建一个权重矩阵 W，并包装为 nn.Parameter
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        按照正态截断初始化权重
        """
        variance = 2.0 / (self.in_features + self.out_features)
        std = math.sqrt(variance)

        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：y = Wx
        输入的 x 形状：(..., in_feature)
        输出的 y 形状：(..., out_feature)
        """
        return torch.einsum('oi, ...i -> ...o', self.W, x)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'