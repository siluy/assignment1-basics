import regex as re
from typing import List, Dict, Tuple, Optional, Iterable, Iterator
import json
import codecs
from cs336_basics.train_BPE import PAT

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = special_tokens

        # 1. 创建反向词表 bytes->int
        self.encoder: Dict[bytes, int] = {token: i for i, token in vocab.items()}
        # 2. 创建 BPE merge 查找表
        self.bpe_ranks: Dict[Tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        # 3. 创建切分 special token 的正则表达式
        if self.special_tokens:
            escaped_special = [re.escape(s) for s in self.special_tokens]
            self.special_pattern = f"({ '|'.join(escaped_special) })"
        else:
            self.special_pattern = ""
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        """
        从文件构造一个 Tokenizer 实例（最终兼容版）。
        增加了对 merges.txt 中特殊情况的兼容处理。
        """
        
        # 加载 vocab.json (这部分逻辑是正确的，无需修改)
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_loaded = json.load(f)
            vocab = {int(idx): bytes(token_list) for idx, token_list in vocab_loaded.items()}

        # 加载 merges.txt
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        # 尝试正常解码
                        p1 = codecs.decode(parts[0], 'unicode_escape').encode('latin-1')
                        p2 = codecs.decode(parts[1], 'unicode_escape').encode('latin-1')
                        merges.append((p1, p2))
                    except UnicodeDecodeError as e:
                        # 捕获我们预期的特定错误
                        if str(e).endswith('\\ at end of string'):
                            print(f"警告: 在 {merges_filepath} 第 {line_num} 行检测到特殊情况，已进行兼容处理。")
                            
                            # 手动处理这个边缘情况
                            processed_parts = []
                            for part in parts:
                                if part.endswith('\\'):
                                    # 解码前半部分，再手动加上反斜杠的字节
                                    main_part = part[:-1]
                                    decoded_bytes = codecs.decode(main_part, 'unicode_escape').encode('latin-1') + b'\\'
                                    processed_parts.append(decoded_bytes)
                                else:
                                    # 如果错误不是由这个 part 引起的，它可能是好的，或者有其他问题
                                    # 我们假设它没问题，但如果还有错，就需要重新抛出
                                    try:
                                        processed_parts.append(codecs.decode(part, 'unicode_escape').encode('latin-1'))
                                    except UnicodeDecodeError:
                                        raise e # 重新抛出原始错误，因为问题比我们预想的复杂
                            
                            if len(processed_parts) == 2:
                                merges.append(tuple(processed_parts))

                        else:
                            # 如果是其他未知的解码错误，还是需要报错
                            raise e

        return cls(vocab, merges, special_tokens)
    
    @staticmethod
    def _get_pairs(word_parts: List[bytes]) -> set:
        """从词表提取相邻对"""
        return set(zip(word_parts[:-1], word_parts[1:]))
    
    def _bpe_merge(self, word_bytes: bytes) -> List[bytes]:
        """对单个字节串做 BPE 合并"""
        # 1. 拆分单字节列表
        parts = [bytes([b]) for b in word_bytes]

        while True:
            # 2. 找出所有相邻字节对
            pairs = self._get_pairs(parts)
            if not pairs:
                break
            # 3. 找出优先级最高的合并规则
            # 为不能合并的部分赋无穷大排名
            best_pair = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if self.bpe_ranks.get(best_pair, float('inf')) == float('inf'):
                break
            # 4. mmerge
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else: 
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        return parts

    # def encode(self, text: str) -> List[int]:
    #     """将文本 string 编码为 token ID list"""
    #     token_ids = []
    #     # __init__ 中的正则表达式将文本切分
    #     chunks = re.split(self.special_pattern, text) if self.special_pattern else [text]

    #     for chunk in chunks:
    #         if not chunk:
    #             continue
    #         if chunk in self.special_tokens:
    #             token_ids.append(self.encoder[chunk.encode('utf-8')])
    #         else: 
    #             # word_bytes = chunk.encode('utf-8')
    #             # merged_parts = self._bpe_merge(word_bytes)
    #             pre_tokenized_parts = re.findall(PAT, chunk)

    #             for part in pre_tokenized_parts:
    #                 part_bytes = part.encode('utf-8')
    #                 merged_parts = self._bpe_merge(part_bytes)
    #                 for final_part in merged_parts:
    #                     token_ids.append(self.encoder[final_part])
    #                 # token_ids.append(self.encoder[part])

    #     return token_ids
    def encode(self, text: str) -> List[int]:
        """将文本字符串编码为 token ID 列表（最终、最强健的版本）。"""
        
        # 如果没有特殊 token，直接使用旧逻辑，快速高效
        if not self.special_tokens:
            # ... （这里的逻辑和你之前的 encode 内部 else 块一样）
            token_ids = []
            pre_tokenized_parts = re.findall(PAT, text)
            for part in pre_tokenized_parts:
                part_bytes = part.encode('utf-8')
                merged_parts = self._bpe_merge(part_bytes)
                for final_part in merged_parts:
                    if final_part in self.encoder:
                        token_ids.append(self.encoder[final_part])
            return token_ids

        # --- 新的、更强健的特殊 token 处理逻辑 ---
        token_ids = []
        
        # 按照长度降序排序，确保优先匹配最长的特殊 token
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        
        start_idx = 0
        while start_idx < len(text):
            # 寻找下一个特殊 token 的出现位置
            next_special_idx = -1
            next_special_token = None

            for special_token in sorted_special_tokens:
                idx = text.find(special_token, start_idx)
                if idx != -1:
                    # 找到了一个，但要确保它是最早出现的那个
                    if next_special_idx == -1 or idx < next_special_idx:
                        next_special_idx = idx
                        next_special_token = special_token
            
            if next_special_idx == -1:
                # 文本末尾没有更多特殊 token 了
                remaining_text = text[start_idx:]
                # 编码剩余的普通文本
                pre_tokenized = re.findall(PAT, remaining_text)
                for part in pre_tokenized:
                    token_ids.extend(self.encoder.get(p) for p in self._bpe_merge(part.encode('utf-8')))
                break # 结束循环
            else:
                # 编码特殊 token 前面的普通文本
                plain_text = text[start_idx:next_special_idx]
                if plain_text:
                    pre_tokenized = re.findall(PAT, plain_text)
                    for part in pre_tokenized:
                        token_ids.extend(self.encoder.get(p) for p in self._bpe_merge(part.encode('utf-8')))
                
                # 编码找到的特殊 token
                token_ids.append(self.encoder[next_special_token.encode('utf-8')])
                
                # 更新下一次搜索的起始位置
                start_idx = next_special_idx + len(next_special_token)
                
        # 过滤掉 BPE 合并中可能产生的 None 值
        return [tid for tid in token_ids if tid is not None]
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """对字符串 iterator 做懒惰编码，逐个产生 token ID"""
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            yield from token_ids

    def decode(self, ids: List[int]) -> str:
        """将 token ID list 解码回 string"""
        # 从词表查找每个 ID 对应的字节，如果有 ID 不存在，就返回空
        all_bytes = b"".join(self.vocab.get(i, b"") for i in ids)
        # 将拼接后的完整字节串解码为字符串
        text = all_bytes.decode("utf-8", errors="replace")

        return text