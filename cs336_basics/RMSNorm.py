import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    均方根归一化
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # g_i is a learnable “gain” parameter
        self.g = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入 tensor，形状(..., d_model)
        
        Returns:
            均方根归一化后的 tensor
        """
        # 存好初始数据类型，转化为 fp32 以保证精度
        original_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        # 计算均方根
        rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        normalized_x = (x_fp32 / rms) * self.g

        return normalized_x.to(original_dtype)