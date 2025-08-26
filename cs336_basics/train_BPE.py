import os
import regex as re
from collections import Counter
from typing import List, Tuple, Dict
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries


def initialization(special_tokens: list[str] = None) -> dict[int, bytes]:
    """
    初始化词表，加入256个数字，并为后续 special tokens 的加入做好准备
    """
    vocab = {}
    for i in range(0, 256):
        vocab[i] = bytes([i])
    
    if special_tokens:
        next_id = 256
        for token_str in special_tokens:
            if token_str.encode("utf-8") not in vocab.values():
                vocab[next_id] = token_str.encode("utf-8")
                next_id += 1

    return vocab

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize_and_count(full_text: str, special_tokens: List[str]) -> Counter:
    """
    首先对特殊 token 做好切分
    然后按照频率切分非特殊 token
    逻辑正确的单线程版本
    """
    # 1. 处理特殊 token，这里使用了 re.escape 去转义 special token 中的特殊符号，比如|
    # 处理经过排序的序列，去优先匹配长的 special token，防止一个长 special token 被误当作两个短的组合去处理
    escaped_special_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]

    # 把所有转义后的 token 用‘|’连接，形成大的匹配模式
    # f"({ ... })" 最外层的括号是“捕获组”。re.split 在使用带捕获组的模式时，会将匹配到的分隔符（也就是 special_tokens）也保留在结果列表中
    split_pattern = f"({ '|'.join(escaped_special_tokens) })"

    # 切分
    text_chunks = re.split(split_pattern, full_text)

    word_counts = Counter()

    for chunk in text_chunks:
        if not chunk:
            continue
        if chunk in special_tokens:
            # 特殊
            word_bytes = chunk.encode("utf-8")
            word_counts[word_bytes] += 1
        else:
            # 普通，用规定的 PAT 匹配规则细分
            for match in re.finditer(PAT, chunk):
                word_bytes = match.group(0).encode("utf-8")
                word_counts[word_bytes] += 1

    return word_counts

# 多线程化
def worker_process(args: Tuple[str, int, int, List[str]]) -> Counter:
    """
    给每一个子进程使用的函数，负责读取文件中的一个部分，然后对其分词，计数
    """
    file_path, start, end, special_tokens = args

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    return pretokenize_and_count(chunk_text, special_tokens)

def parallel_pretokenize_and_count(file_path: str, special_tokens: List[str]) -> Counter:
    """
    并行化地对大文件进行分词计数
    """
    special_token_bytes = special_tokens[0].encode("utf-8")
    num_processes = cpu_count()
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token_bytes)

    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((file_path, start, end, special_tokens))

    total_counts = Counter()
    with Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(worker_process, tasks)

        for single_chunk_counts in results:
            total_counts.update(single_chunk_counts)

    return total_counts


def compute_merges(word_counts: Dict[bytes, int], vocab_size: int, special_tokens: List[str] = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    一个使用倒排索引进行高效且正确 BPE 合并的实现。
    此版本修复了处理单词内重复字节对时的 KeyError 问题。
    """
    if special_tokens is None:
        special_tokens = []

    # 1. 初始化词汇表和合并列表
    vocab = initialization(special_tokens)
    num_merges = vocab_size - len(vocab)
    merges = []

    # 2. 将数据转换为内部处理格式
    word_freqs = {
        tuple(bytes([b]) for b in word): count for word, count in word_counts.items()
    }
    special_tokens_as_tuples = {
        tuple(bytes([b]) for b in s.encode('utf-8')) for s in special_tokens
    }

    # 3. 计算初始字节对频率 (stats) 和构建倒排索引 (indices)
    stats = defaultdict(int)
    indices = defaultdict(set)
    for word, count in word_freqs.items():
        if word in special_tokens_as_tuples or len(word) < 2:
            continue
        for pair in zip(word[:-1], word[1:]):
            stats[pair] += count
            indices[pair].add(word)

    # 4. 主合并循环
    for i in range(num_merges):
        if not stats:
            break

        # 找到最高频的字节对 (使用稳定排序以确保确定性)
        best_pair = max(stats, key=lambda p: (stats[p], p))

        # 记录合并信息
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        
        # 5. 高效更新
        words_to_process = list(indices.pop(best_pair, set()))
        for word in words_to_process:
            if word not in word_freqs:
                continue
            
            count = word_freqs.pop(word)

            # a. 从 stats 和 indices 中移除旧词的影响
            #    首先，更新 stats（需要考虑所有重复的 pair）
            for pair in zip(word[:-1], word[1:]):
                if pair in stats:
                    stats[pair] -= count
                    if stats[pair] == 0:
                        del stats[pair]
            
            #    然后，更新 indices（只需考虑 unique pair）
            for pair in set(zip(word[:-1], word[1:])):
                if pair in indices:
                    indices[pair].discard(word)
                    if not indices[pair]:
                        del indices[pair]

            # b. 创建新词
            new_word_list = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and (word[j], word[j+1]) == best_pair:
                    new_word_list.append(new_token)
                    j += 2
                else:
                    new_word_list.append(word[j])
                    j += 1
            new_word = tuple(new_word_list)

            # c. 将新词加入 word_freqs
            new_count = word_freqs.get(new_word, 0) + count
            word_freqs[new_word] = new_count

            # d. 为新词在 stats 和 indices 中添加影响
            if len(new_word) < 2:
                continue
            for pair in zip(new_word[:-1], new_word[1:]):
                stats[pair] += count
                indices[pair].add(new_word)

    return vocab, merges



def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[dict, list]:
    """
    整合所有步骤，完成作业文档的要求：
    Your BPE training function should handle (at least) the following input parameters:
    input_path: str Path to a text file with BPE tokenizer training data.
    vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
    otherwise affect BPE training.
    Your BPE training function should return the resulting vocabulary and merges:
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
    lary) to bytes (token bytes).
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation.
    """
    # 1. 并行地完成分词和计数任务
    word_counts_bytes = parallel_pretokenize_and_count(input_path, special_tokens)

    # 2. 调用合并函数计算最终的词汇表和合并规则
    final_vocab, final_merges = compute_merges(word_counts_bytes, vocab_size, special_tokens)
    return final_vocab, final_merges