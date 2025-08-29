import torch
import torch.nn as nn

from .Linear import Linear

class PositionWiseFeedForward(nn.Module):
    """
    SeiGLU 的 FFN
    """
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            # d_ff 是 8/3 * d_model, 且是64倍数
            d_ff = int((8 / 3 * d_model) / 64) * 64

        # 初始化三个线性层
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN(x) = W2(SiLU(W1(x)) * W3(x))
        """
        w1 = self.w1(x)
        silu_output = w1 * torch.sigmoid(w1)

        w3 = self.w3(x)

        gated_output = silu_output * w3

        output = self.w2(gated_output)
        return output
    