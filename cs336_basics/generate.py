import torch

from tokenizer import Tokenizer
from Transformer import TransformerLM
from attention_functions import softmax
from checkpointing import load_checkpoint

@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.95
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
            
            # 找到累积概率首次超过 top_p 的位置，并创建移除掩码
            # 这是标准的 nucleus sampling 逻辑，保留第一个超过阈值的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # --- 这是修正的核心部分 ---
            # 1. 直接在排好序的概率上应用掩码，将不满足条件的概率设为0
            sorted_probs[sorted_indices_to_remove] = 0

            # 2. 创建一个新的、全为0的张量，用于接收恢复顺序后的概率
            probs_new = torch.zeros_like(probs)

            # 3. 使用 scatter_ 将排序后的、经过筛选的概率值放回它们在词汇表中的原始位置
            #    - `sorted_indices` 告诉我们每个排序后的值应该去哪里
            #    - `sorted_probs` 是我们要放置的值
            probs_new.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
            
            # 4. 用新的、经过筛选的概率替换旧的概率
            probs = probs_new
            
            # 重新归一化确保概率和为1
            # 加上一个极小值防止除以零
            probs = probs / (torch.sum(probs, dim=-1, keepdim=True) + 1e-9)

        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

        # eos_token_id = tokenizer.encode.get("<|endoftext|>".encode('utf-8'))
        # if eos_token_id and next_id.item() == eos_token_id:
        #     break

    generated_text = tokenizer.decode(idx[0].tolist())
    model.train()
    return generated_text

if __name__ == "__main__":
    # --- 1. 设置和定义路径 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 定义模型检查点和分词器相关文件的路径
    base_path = "/home/siluyang/CS336/assignment1-basics/ckpt"
    ckpt_path = f"{base_path}/checkpoint.pt"
    vocab_path = "/home/siluyang/CS336/assignment1-basics/cs336_basics/vocab_ts10k.json"  # <--- 确认你的 vocab.json 路径
    merges_path = "/home/siluyang/CS336/assignment1-basics/cs336_basics/merges_ts10k.txt"  # <--- 确认你的 merges.txt 路径

    # --- 2. 加载模型 (这部分无需修改) ---
    checkpoint = torch.load(ckpt_path, map_location=device)
    try:
        model_args = checkpoint['model_args']
    except KeyError:
        model_args = {
        'vocab_size': 10000,      # 你的词汇表大小
        'd_model': 512,          # 模型的维度
        'num_layers': 4,          # Transformer层数
        'num_heads': 16,           # 多头注意力头数
        'context_length': 256,  # 上下文长度
        'd_ff': 1344
        # ... 确保这里包含了所有 TransformerLM 初始化时需要的参数
    }

    model = TransformerLM(**model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"模型已从 {ckpt_path} 加载。")

    # --- 3. 加载分词器 (根据你的 Tokenizer 类进行修改) ---
    try:
        # 定义你的模型可能使用的特殊 tokens
        special_tokens = ["<|endoftext|>"]
        
        # 使用 from_files 类方法来实例化
        tokenizer = Tokenizer.from_files(
            vocab_filepath=vocab_path, 
            merges_filepath=merges_path,
            special_tokens=special_tokens
        )
        print(f"分词器已从 {vocab_path} 和 {merges_path} 加载。")
    except FileNotFoundError:
        print(f"错误：找不到分词器文件。请检查路径：\n- {vocab_path}\n- {merges_path}")
        exit()

    # --- 4. 开始交互式生成 (这部分无需修改) ---
    print("\n模型准备就绪，可以开始对话。输入 'exit' 或 'quit' 退出。")
    while True:
        prompt = input("用户: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        
        response = generate(
            model, 
            tokenizer, 
            prompt, 
            max_new_tokens=256
        )
        response_only = response[len(prompt):]
        
        print("模型:", response_only)
        print("-" * 20)