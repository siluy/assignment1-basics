import torch

from .tokenizer import Tokenizer
from .Transformer import TransformerLM
from .attention_functions import softmax

@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0
) -> str:
    """
    从给定的 prompt 开始进行自回归生成文本
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)

    for _ in range(max_new_tokens):
        if idx.size(1) > model.context_length:
            idx = idx[:, -model.context_length:]

        logits = model(idx)

        next_token_logits = logits[:, -1, :] # (1, vocab_size)

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        probs = softmax(next_token_logits, dim=-1)

        if top_p < 1.0:
            # 对概率降序排列
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            # 计算累积概率
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # 找到超出 top_p阈值的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            # 将第一个超过的 token 保留，故向右移一位
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            # 创建 False 掩码，将要移除的 token 设为 True
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
                dim=-1, index = sorted_indices[sorted_indices_to_remove], src=torch.ones_like(probs, dtype=torch.bool)
            )
            # 将要移除的 token 概率设0
            probs[indices_to_remove] = 0
            # 重新归一化确保概率和为1
            probs = probs / torch.sum(probs, dim=-1, keepdim=True)

        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

        eos_token_id = tokenizer.encode.get(b"<|endoftext|>")
        if eos_token_id and next_id.item() == eos_token_id:
            break

    generated_text = tokenizer.decode(idx[0].tolist())
    model.train()
    return generated_text