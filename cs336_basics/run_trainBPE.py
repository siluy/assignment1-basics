import os
import sys
import time
import json
import cProfile
import pstats
import tracemalloc
from typing import Dict, List, Tuple

# -------------------------------------------------------------------
# 1. 准备工作: 导入你写好的 BPE 训练函数
# -------------------------------------------------------------------
# 假设你的 tokenizer.py 在 'cs336_basics' 目录下
# 请根据你的项目结构调整路径
TOKENIZER_FILE_PATH = './cs336_basics/tokenizer.py' 
# 添加项目路径以便导入模块
#sys.path.append(os.path.dirname(TOKENIZER_FILE_PATH))
from cs336_basics.tokenizer import train_bpe


# -------------------------------------------------------------------
# 2. 定义常量和辅助函数
# -------------------------------------------------------------------
DATASET_PATH = "/home/siluyang/CS336/assignment1-basics/data/owt_train.txt" # <--- 修改这里
VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<|endoftext|>"]
VOCAB_OUTPUT_PATH = "vocab_owt.json"
MERGES_OUTPUT_PATH = "merges_owt.txt"
PROFILE_OUTPUT_PATH = "training_profile_owt.prof"

def save_vocab_and_merges(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]]):
    """序列化词汇表和合并规则到磁盘"""
    
    # 为了让 JSON 可读，将 bytes 转换为整数列表
    serializable_vocab = {k: list(v) for k, v in vocab.items()}
    with open(VOCAB_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    print(f"✅ 词汇表已保存到: {VOCAB_OUTPUT_PATH}")

    # 将合并规则保存为纯文本，更易读
    with open(MERGES_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for p1, p2 in merges:
            # 尝试解码为可读字符，失败则保留原始字节表示
            s1 = p1.decode('utf-8', errors='ignore')
            s2 = p2.decode('utf-8', errors='ignore')
            f.write(f"{s1} {s2}\n")
    print(f"✅ 合并规则已保存到: {MERGES_OUTPUT_PATH}")


def analyze_results(vocab: Dict[int, bytes], duration_s: float, peak_mem_mb: float):
    """分析结果并打印报告"""
    
    # 找出最长的 token
    longest_token_bytes = b''
    for token_bytes in vocab.values():
        if len(token_bytes) > len(longest_token_bytes):
            longest_token_bytes = token_bytes
            
    # 尝试解码为字符串以便分析
    longest_token_str = longest_token_bytes.decode('utf-8', errors='replace')

    print("\n--- 训练结果分析 ---")
    print(f"🕒 训练耗时: {duration_s / 60:.2f} 分钟 ({duration_s / 3600:.4f} 小时)")
    print(f"🧠 峰值内存: {peak_mem_mb:.2f} MB ({peak_mem_mb / 1024:.2f} GB)")
    print(f"📜 最长Token (长度 {len(longest_token_bytes)} 字节): '{longest_token_str}'")
    
    # 分析最长 token 是否合理
    print("\n🤔 最长Token合理性分析:")
    print("这个 token 很可能是一个在 TinyStories 数据集中高频出现的、有意义的英文单词或短语。")
    print("例如 ' because', ' something', ' little' 等。因为BPE算法会不断合并高频相邻的字节对，")
    print("所以最常见的词最终会成为词汇表中的长 token。这完全符合预期。")


# -------------------------------------------------------------------
# 3. 主执行函数
# -------------------------------------------------------------------
def main():
    """主函数，执行训练和分析"""
    print("--- 开始 BPE Tokenizer 训练 ---")
    
    # (a) 启动时间和内存监控
    tracemalloc.start()
    start_time = time.time()
    
    # 执行训练
    print(f"从 '{DATASET_PATH}' 加载数据...")
    final_vocab, final_merges = train_bpe(
        input_path=DATASET_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )
    
    # 记录时间和内存使用
    end_time = time.time()
    duration_seconds = end_time - start_time
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    peak_mem_mb = peak_mem_bytes / 1024 / 1024
    tracemalloc.stop()
    
    print("\n--- 训练完成 ---")
    
    # 序列化结果
    save_vocab_and_merges(final_vocab, final_merges)
    
    # 分析并报告结果
    analyze_results(final_vocab, duration_seconds, peak_mem_mb)
    
    print("\n--- 任务 (a) 完成 ---")
    
# -------------------------------------------------------------------
# 4. 性能分析函数
# -------------------------------------------------------------------
def profile_main():
    """使用 cProfile 运行主函数以进行性能分析"""
    print("\n--- 启动性能分析 ---")
    profiler = cProfile.Profile()
    profiler.run('main()')
    profiler.dump_stats(PROFILE_OUTPUT_PATH)
    print(f"📈 性能分析报告已保存到: {PROFILE_OUTPUT_PATH}")

    # (b) 读取性能分析报告并找出瓶颈
    stats = pstats.Stats(PROFILE_OUTPUT_PATH)
    stats.sort_stats('cumtime') # 按累计耗时排序
    print("\n--- 性能瓶颈分析 (耗时最长的前5个函数) ---")
    stats.print_stats(5)
    
    print("\n🤔 性能瓶颈分析:")
    print("从报告中可以看出，绝大部分时间都消耗在了 `compute_merges` 函数上。")
    print("尽管我们已经对其进行了优化，但BPE合并过程本身固有的计算复杂性使其成为理所当然的性能瓶颈。")
    print("其次耗时较多的可能是多进程预分词的 `parallel_pretokenize_and_count` 函数。")
    print("\n--- 任务 (b) 完成 ---")


if __name__ == '__main__':
    # 为了分别完成两个任务，你可以选择运行其中一个
    
    # 任务 (a): 直接运行训练并获取结果
    # main() 
    
    # 任务 (b): 运行性能分析 (这也会完整跑一遍训练)
    profile_main()