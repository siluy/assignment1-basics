import torch
import math
from typing import Iterable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
                state = self.state[p]

                # 1. 初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 一阶矩 动量
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # 二阶矩 梯度平方
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                t = state['step']

                # 2. 更新一阶矩和二阶矩
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3. 计算 bias 修正和步长
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # 4. 计算分母并更新参数
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # 5. 应用解耦的权重衰减 weight decay
                # p.data = p.data - lr * lambda * p.data
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss
    

def consine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    根据当前 step t，计算余弦退火的 lr

    Args: 
        t: 当前迭代步数
        alpha_max: 最大学习率
        alpha_min: 最小学习率
        T_w: 预热阶段总步数
        T_c: 余弦退火阶段总步数(不包括预热)

    Returns:
        为当前 step t 计算出的 lr
    """
    if T_c < T_w:
        raise ValueError("T_c must be >= T_w")
    # 1. warm up(t < T_w)
    if t < T_w:
        # alpha_t = (t / T_w) * alpha_max
        return (t / T_w) * alpha_max
    # 2. 余弦退火(T_w <= t <= T_c)
    elif T_w <= t <= T_c:
        # 归一化当前步数在退火阶段的位置
        progress = (t - T_w) / (T_c - T_w)
        cosine_component = 0.5 * (1 + math.cos(math.pi * progress))
        # alpha_t = alpha_min + cosine_component * (alpha_max - alpha_min)
        return alpha_min + cosine_component * (alpha_max - alpha_min)
    # 3. 后退火(t > T_c)
    else:
        return alpha_min
    
def clip_grad_norm(parameters: Iterable[torch.Tensor], max_norm: float, eps: float = 1e-6):
    """
    裁剪一个参数的 iterator 里所有参数的梯度，在所有梯度上计算范数

    Args:
        parameters: 一个包含待裁剪的梯度的张量的可迭代对象
        max_norm: 梯度的最大范数
        eps: 数值稳定
    """
    # 1. 获取所有非空的梯度
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return
    # 2. 计算所有梯度的总 L2 范数
    # 先计算每个梯度张量的范数，然后将这些范数的值作为向量再计算一次总范数
    total_norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in grads]), p=2)
    # 3. 计算裁剪系数
    clip_coef = max_norm / (total_norm + eps)
    # 4. 只有总范数超过了最大值（即系数<1），才应用裁剪
    if clip_coef < 1.0:
        for g in grads:
            g.mul_(clip_coef)