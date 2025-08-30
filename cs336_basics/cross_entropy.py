import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失，减去最大值避免上溢

    Args:
        logits: 模型的原始输出(logits)，形状(..., vocab_size)
        targets: 真实 token ID, 形状(...)，能与 logits 的前导维度匹配

    Returns:
        一个标量(scalar) tensor，代表所有样本的平均损失
    """
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stable_logits = logits - max_logits
    # log(sum(exp(...))), 是 softmax 分母的对数
    log_sum = torch.log(torch.sum(torch.exp(stable_logits), dim=-1))
    # 加回来 max
    log_denominator = log_sum + max_logits.squeeze(-1)
    # 获取真实 token 的 logit
    correct_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = log_denominator - correct_logits

    return loss.mean()