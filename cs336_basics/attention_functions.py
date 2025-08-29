import torch
import math

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    对输入的 tensor 应用 softmax，并使用减去最大值的方法避免数值上溢(overflow)

    Args:
        x: input tensor
        dim: 应用 softmax 的维度，默认最后一个

    Returns:
        与输入形状相同的 tensor，但在数值上经过了 softmax
    """
    # minus max
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_minus = x - max_val

    exp = torch.exp(x_minus)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum_exp

def scaled_dot_product_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
) -> torch.Tensor:
    """
    计算缩放点积注意力
    Args:
        q: Query, 形状(..., seq_len_q, d_k)
        k: Key, 形状(..., seq_len_k, d_k)
        v: Value, 形状(..., seq_len_k, d_v)
        mask: 掩码(bool), 形状(..., seq_len_q, seq_len_k)
              True 表示应该 attend，False 相反

    Returns:
        形状为(..., seq_len_q, d_v)的 tensor
    """
    # 1. 获取 d_k 并计算点积
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) # 计算 Q 与 K^T 的矩阵乘法 (..., seq_len_q, seq_len_k)
    # 2. 缩放
    scaled_scores = scores / math.sqrt(d_k)
    # 3. 掩码
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(mask == False, -torch.inf)
    # 4. softmax
    attention_weights = softmax(scaled_scores, dim=-1) # (..., seq_len_q, seq_len_k)
    # 5. times V
    output = attention_weights @ v # (..., seq_len_q, d_v)
    return output
