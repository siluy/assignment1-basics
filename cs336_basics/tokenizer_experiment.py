import time
import random
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional

# 假设你的 Tokenizer 类保存在 tokenizer.py 文件中
from tokenizer import Tokenizer

# ==============================================================================
# 脚本配置部分 (*** 请务必修改这里的路径 ***)
# ==============================================================================

# 1. 直接指定你训练好的4个分词器文件路径
TS_VOCAB_PATH = "/home/siluyang/CS336/assignment1-basics/cs336_basics/vocab_ts10k.json"
TS_MERGES_PATH = "/home/siluyang/CS336/assignment1-basics/cs336_basics/merges_ts10k.txt"
OWT_VOCAB_PATH = "/home/siluyang/CS336/assignment1-basics/cs336_basics/vocab_owt.json"
OWT_MERGES_PATH = "/home/siluyang/CS336/assignment1-basics/cs336_basics/merges_owt.txt"

# 2. 指定你的数据集文件路径
TS_VALID_PATH = "/home/siluyang/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"    # TinyStories 验证集
OWT_VALID_PATH = "/home/siluyang/CS336/assignment1-basics/data/owt_train.txt" # OpenWebText 验证集

# 3. 指定编码后数据集的输出路径
TS_ENCODED_OUTPUT_PATH = "/home/siluyang/CS336/assignment1-basics/data/tinystories_tra_encoded.npy"
OWT_ENCODED_OUTPUT_PATH = "/home/siluyang/CS336/assignment1-basics/data/openwebtext_tra_encoded.npy"

# ==============================================================================
# 实验辅助函数
# ==============================================================================

def sample_documents(file_path: str, num_docs: int = 10, delimiter: bytes = b"<|endoftext|>") -> str:
    """从一个大数据文件中随机抽样指定数量的文档。"""
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，跳过抽样。")
        return ""
    
    print(f"从 {file_path} 中抽样 {num_docs} 个文档...")
    with open(file_path, "rb") as f:
        content = f.read()
    
    documents = content.split(delimiter)
    sampled_docs = random.sample(documents, min(num_docs, len(documents)))
    
    # 将抽样出的文档拼接起来并解码为字符串
    return delimiter.join(sampled_docs).decode("utf-8", errors="ignore")

def tokenize_and_save_dataset(tokenizer: Tokenizer, input_path: str, output_path: str):
    """使用 encode_iterable 对整个数据集进行编码并保存为 .npy 文件。"""
    if not os.path.exists(input_path):
        print(f"警告: 文件 {input_path} 不存在，跳过编码。")
        return
        
    print(f"开始编码文件 {input_path}...")
    start_time = time.time()
    
    all_ids = []
    with open(input_path, "r", encoding="utf-8") as f:
        # 使用内存高效的迭代编码
        for token_id in tokenizer.encode_iterable(f):
            all_ids.append(token_id)
            
    # 使用 uint16 类型来节省空间
    token_array = np.array(all_ids, dtype=np.uint16)
    np.save(output_path, token_array)
    
    end_time = time.time()
    print(f"编码完成！共 {len(token_array)} 个 token。")
    print(f"结果已保存到: {output_path}")
    print(f"耗时: {end_time - start_time:.2f} 秒")

# ==============================================================================
# 主执行逻辑
# ==============================================================================

if __name__ == "__main__":
    
    # 加载两个分词器
    print("正在加载分词器...")
    try:
        tokenizer_ts = Tokenizer.from_files(
            vocab_filepath=TS_VOCAB_PATH,
            merges_filepath=TS_MERGES_PATH,
            special_tokens=["<|endoftext|>"]
        )
        print("TinyStories 分词器加载成功。")
    except FileNotFoundError:
        tokenizer_ts = None
        print(f"警告: 无法找到 TinyStories 分词器文件，请检查路径: {TS_VOCAB_PATH} 和 {TS_MERGES_PATH}")

    try:
        tokenizer_owt = Tokenizer.from_files(
            vocab_filepath=OWT_VOCAB_PATH,
            merges_filepath=OWT_MERGES_PATH,
            special_tokens=["<|endoftext|>"]
        )
        print("OpenWebText 分词器加载成功。")
    except FileNotFoundError:
        tokenizer_owt = None
        print(f"警告: 无法找到 OpenWebText 分词器文件，请检查路径: {OWT_VOCAB_PATH} 和 {OWT_MERGES_PATH}")

    print("-" * 50)
    
    # # --- (a) & (b) 小问: 计算压缩率 ---
    # print("\n>>> 开始执行 (a) 和 (b) 小问: 计算压缩率\n")
    
    # text_sample_ts = sample_documents(TS_VALID_PATH)
    # text_sample_owt = sample_documents(OWT_VALID_PATH)

    # if tokenizer_ts and text_sample_ts:
    #     bytes_ts = len(text_sample_ts.encode('utf-8'))
    #     tokens_ts = len(tokenizer_ts.encode(text_sample_ts))
    #     print(f"(a) TinyStories 分词器在 TinyStories 样本上的压缩率: {bytes_ts / tokens_ts:.2f} 字节/token")

    # if tokenizer_owt and text_sample_owt:
    #     bytes_owt = len(text_sample_owt.encode('utf-8'))
    #     tokens_owt = len(tokenizer_owt.encode(text_sample_owt))
    #     print(f"(a) OpenWebText 分词器在 OpenWebText 样本上的压缩率: {bytes_owt / tokens_owt:.2f} 字节/token")

    # if tokenizer_ts and text_sample_owt:
    #     tokens_cross = len(tokenizer_ts.encode(text_sample_owt))
    #     bytes_owt_for_cross = len(text_sample_owt.encode('utf-8'))
    #     print(f"(b) TinyStories 分词器在 OpenWebText 样本上的压缩率: {bytes_owt_for_cross / tokens_cross:.2f} 字节/token")

    # print("-" * 50)
    
    # # --- (c) 小问: 估算吞吐量 ---
    # print("\n>>> 开始执行 (c) 小问: 估算吞吐量\n")
    # if tokenizer_owt and os.path.exists(OWT_VALID_PATH):
    #     print(f"正在使用 {OWT_VALID_PATH} 测试 OpenWebText 分词器吞吐量...")
    #     with open(OWT_VALID_PATH, "r", encoding="utf-8") as f:
    #         content = f.read()
        
    #     total_bytes = len(content.encode('utf-8'))
        
    #     start_time = time.time()
    #     _ = tokenizer_owt.encode(content) # 使用 encode 测试原始速度
    #     end_time = time.time()
        
    #     elapsed_time = end_time - start_time
    #     throughput_mb_s = (total_bytes / elapsed_time) / (1024 * 1024)
        
    #     print(f"吞吐量: {throughput_mb_s:.2f} MB/s")
        
    #     pile_gb = 825
    #     time_for_pile_hours = (pile_gb * 1024) / throughput_mb_s / 3600
    #     print(f"按此速度，处理 825GB 的 Pile 数据集大约需要: {time_for_pile_hours:.2f} 小时")

    # print("-" * 50)

    # --- (d) 小问: 编码完整数据集 ---
    print("\n>>> 开始执行 (d) 小问: 编码完整数据集\n")
    
    if tokenizer_ts:
        tokenize_and_save_dataset(tokenizer_ts, TS_VALID_PATH, TS_ENCODED_OUTPUT_PATH)
    
    if tokenizer_owt:
        tokenize_and_save_dataset(tokenizer_owt, OWT_VALID_PATH, OWT_ENCODED_OUTPUT_PATH)
        
    print("\n(d) 小问的概念性问题回答:")
    print("为什么 uint16 是一个合适的 dtype 选择？")
    print("回答: uint16 (无符号16位整数) 可以表示的范围是 0 到 65,535 (2^16 - 1)。")
    print("对于 TinyStories (10,000 词汇表) 和 OpenWebText (32,000 词汇表)，所有的 token ID 都在这个范围内。")
    print("使用 uint16 相比默认的 int64 或 uint32，每个 ID 只占用 2 个字节，可以大大节省磁盘空间和加载到内存/显存中的体积。")

    print("\n所有实验已执行完毕！")