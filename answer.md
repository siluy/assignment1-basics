## answer to the questions in the assignment

### 2 Byte-Pair Encoding (BPE) Tokenizer

#### Problem (unicode1): Understanding Unicode (1 point)
- We skip this cause you can get the answer from the Python run result easily

---

#### Problem (unicode2): Unicode Encodings (3 points)
(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings
- utf8是三种utf编码中最经济的，对同一个hello! こんにちは!作编码，长度分别是23，28，56，而且它的前 128 个字符和 ASCII 完全一样，便于系统过渡和兼容
- UTF-8 is the most economical of the three UTF encodings. For the same strings "hello!" and "こんにちは!", the encoded lengths are 23, 28, and 56 bytes respectively. Additionally, the first 128 characters of UTF-8 are identical to ASCII, making it easier for systems to transition and maintain compatibility.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
- 不是所有 Unicode 字符都能用一个字节来表示. 尝试用一个需要多个字节表示的字符（比如汉字“牛” ）编码成字节串，然后把它作为输入传给这个错误的函数，就能得到首个编码错误的报错。这个函数的核心错误在于它假设了“一个字节对应一个字符”，并尝试独立解码每一个字节 。但实际上，在 UTF-8 中，很多字符是由多个字节共同表示的。这些字节作为一个整体才有意义，分开来单独解码就会导致`UnicodeDecodeError`，因为它破坏了字符的完整表示
- Not all Unicode characters can be represented with a single byte. Try encoding a character that requires multiple bytes (such as the Chinese character "牛") into a byte string, and then pass it as input to this erroneous function to get the first encoding error. The core error of this function is that it assumes "one byte corresponds to one character" and attempts to decode each byte independently. However, in UTF-8, many characters (such as the Chinese character "牛") are represented by multiple bytes. These bytes only make sense as a whole, and decoding them separately will lead to a `UnicodeDecodeError`, as it breaks the complete representation of the character.

(c) Give a two byte sequence that does not decode to any Unicode character(s).
- print(b'\x80\x80'.decode("utf-8"))
- 对第一个字节的要求导致了特定开头的utf-8无法被解码为unicode
- The requirement for the first byte prevents certain leading UTF-8 sequences from being decoded into Unicode.

---

#### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points) && Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
- We skip this cause I failed to store the result from my `run_trainBPE.py`. You can easily get the answer by running it again...

---

#### Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinySStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
- TinyStories 分词器在 TinyStories 样本上的压缩率: 2.54 字节/token
- OpenWebText 分词器在 OpenWebText 样本上的压缩率: 2.44 字节/token
- The TinyStories tokenizer has a compression ratio of 2.54 bytes/token on the TinyStories samples.
- The OpenWebText tokenizer has a compression ratio of 2.44 bytes/token on the OpenWebText samples.

(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
-  TinyStories 分词器在 OpenWebText 样本上的压缩率: 2.36 字节/token
- The TinyStories tokenizer has a compression ratio of 2.36 bytes/token on the OpenWebText samples.

(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
- 吞吐量: 0.48 MB/s.按此速度，处理 825GB 的 Pile 数据集大约需要: 492.72 小时
- Throughput: 0.48 MB/s. At this speed, it would take approximately 492.72 hours to process the 825GB Pile dataset.

(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?
- TinyStoriesV2-GPT4-valid.txt 共 8810222 个 token。耗时: 40.70 秒
- owt_valid.txt 共 118683561 个 token。耗时: 603.75 秒
- uint16 (无符号16位整数) 可以表示的范围是 0 到 65,535 (2^16 - 1)。对于 TinyStories (10,000 词汇表) 和 OpenWebText (32,000 词汇表)，所有的 token ID 都在这个范围内。使用 uint16 相比默认的 int64 或 uint32，每个 ID 只占用 2 个字节，可以大大节省磁盘空间和加载到内存/显存中的体积。
- uint16 (unsigned 16-bit integer) can represent values from 0 to 65,535 (2^16 - 1). For TinyStories (10,000 vocabulary) and OpenWebText (32,000 vocabulary), all token IDs fall within this range. Using uint16 instead of the default int64 or uint32 allows each ID to occupy only 2 bytes, significantly saving disk space and memory/GPU memory when loaded.

---
### 3 Transformer Language Model Architecture

---

#### Problem (transformer_accounting): Transformer LM resource accounting (5 points) **I think this part is really practical and interesting**
我们遵循作业中的规则（§3.6），即矩阵乘法 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$ 的计算量为 $2mnp$ FLOPs。

We follow the rule that the computational cost of matrix multiplication $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$ is $2mnp$ FLOPs.

(a) Consider GPT-2 XL, which has the following configuration:
- vocab_size : 50,257
- context_length : 1,024
- num_layers : 48
- d_model : 1,600
- num_heads : 25
- d_ff : 6,400

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

---

我们根据作业中所构建的模型架构（包含 SwiGLU FFN）和给定的 GPT-2 XL 配置来计算。

**参数计算**:
1.  **词嵌入层 (Token Embeddings)**:
    * `vocab_size * d_model` = 50,257 * 1,600 = 80,411,200 参数

2.  **每个 Transformer Block (共 48 个)**:
    * **多头自注意力 (Multi-Head Self-Attention)**:
        * Q, K, V 投影: $3 \times (d_{model} \times d_{model})$ = 3 * 1,600 * 1,600 = 7,680,000
        * 输出投影: $d_{model} \times d_{model}$ = 1,600 * 1,600 = 2,560,000
        * *小计*: 10,240,000
    * **前馈网络 (SwiGLU FFN)**:
        * W1, W3 门控层: $2 \times (d_{model} \times d_{ff})$ = 2 * 1,600 * 6,400 = 20,480,000
        * W2 下采样层: $d_{ff} \times d_{model}$ = 6,400 * 1,600 = 10,240,000
        * *小计*: 30,720,000
    * **RMSNorm 层**:
        * 两个 Norm 层的增益参数: $2 \times d_{model}$ = 2 * 1,600 = 3,200
    * **每个 Block 的总参数**: 10,240,000 + 30,720,000 + 3,200 = 40,963,200

3.  **所有 Transformer Block 的总参数**:
    * `num_layers * params_per_block` = 48 * 40,963,200 = 1,966,233,600

4.  **最终的 RMSNorm 层**:
    * `d_model` = 1,600

5.  **输出投影层 (LM Head)**:
    * `d_model * vocab_size` = 1,600 * 50,257 = 80,411,200

**模型总参数量**:
$80,411,200 (\text{Embed}) + 1,966,233,600 (\text{Blocks}) + 1,600 (\text{Final Norm}) + 80,411,200 (\text{LM Head}) = \textbf{2,127,057,600}$

该模型大约有 **21.3 亿** 个可训练参数。假设每个参数使用单精度浮点数（4字节），加载该模型需要的内存为 $2,127,057,600 \times 4$ 字节 $\approx \textbf{7.92 GB}$。

---

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens

我们计算当输入序列长度为 `context_length` (S=1,024) 时，所有矩阵乘法所需的 FLOPs。

**矩阵乘法列表及 FLOPs**:
1.  **每个 Transformer Block 内部 (循环 48 次)**:
    * **注意力 Q, K, V, O 投影**: 4 个 `(S, d_model)` 与 `(d_model, d_model)` 的乘法。
        * $4 \times (2 \times S \times d_{model} \times d_{model}) = 8 \times 1024 \times 1600^2 \approx 2.10 \times 10^{10}$
    * **注意力分数计算 ($QK^T$)**: `num_heads` 个 `(S, d_head)` 与 `(d_head, S)` 的乘法。
        * $2 \times S^2 \times d_{model} = 2 \times 1024^2 \times 1600 \approx 3.36 \times 10^9$
    * **注意力值加权 ($Weights \cdot V$)**: `num_heads` 个 `(S, S)` 与 `(S, d_head)` 的乘法。
        * $2 \times S^2 \times d_{model} = 2 \times 1024^2 \times 1600 \approx 3.36 \times 10^9$
    * **FFN (W1, W3, W2)**: 两个 `(S, d_model)` 与 `(d_model, d_ff)` 的乘法，一个 `(S, d_ff)` 与 `(d_ff, d_model)` 的乘法。
        * $2 \times (2 \times S \times d_{model} \times d_{ff}) + (2 \times S \times d_{ff} \times d_{model}) = 6 \times 1024 \times 1600 \times 6400 \approx 6.29 \times 10^{10}$
2.  **模型顶层**:
    * **LM Head**: `(S, d_model)` 与 `(d_model, vocab_size)` 的乘法。
        * $2 \times S \times d_{model} \times vocab\_size = 2 \times 1024 \times 1600 \times 50257 \approx 1.65 \times 10^{11}$

**总 FLOPs**:
$48 \times (2.10 \times 10^{10} + 3.36 \times 10^9 + 3.36 \times 10^9 + 6.29 \times 10^{10}) + 1.65 \times 10^{11} \approx \textbf{4.52 TeraFLOPs}$

完成一次前向传播总共需要约 **4.52 万亿 (Tera)** FLOPs。

---

(c) Based on your analysis above, which parts of the model require the most FLOPs?

根据上面的分析，可以看到**前馈网络 (FFN) 的计算量是最大的**。在每个 Block 中，FFN 的 FLOPs ($6.29 \times 10^{10}$) 远超注意力机制中所有投影和分数计算的总和 ($2.77 \times 10^{10}$)。

---

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

我们使用标准的 GPT-2 FFN 结构（$d_{ff}=4d_{model}$，2个权重矩阵）进行比较，其 FFN FLOPs 为 $2 \times (2 \times S \times d_{model} \times d_{ff}) = 4 \times S \times d_{model} \times (4 d_{model}) = 16Sd_{model}^2$。注意力部分的 FLOPs 保持不变。

| 模型         | L  | d_model | h  | 注意力 FLOPs (占比) | FFN FLOPs (占比) | LM Head FLOPs (占比) |
|--------------|----|---------|----|-----------------------|--------------------|------------------------|
| **GPT-2 Small** | 12 | 768     | 12 | 0.22 TFLOPs (35%)     | 0.39 TFLOPs (62%)  | 0.02 TFLOPs (3%)       |
| **GPT-2 Medium** | 24 | 1024    | 16 | 0.70 TFLOPs (35%)     | 1.26 TFLOPs (63%)  | 0.02 TFLOPs (2%)       |
| **GPT-2 Large** | 36 | 1280    | 20 | 1.59 TFLOPs (36%)     | 2.83 TFLOPs (62%)  | 0.03 TFLOPs (2%)       |

随着模型尺寸的增加，**注意力和 FFN 的 FLOPs 占比基本保持稳定**，FFN 约占总计算量的 62-63%，注意力约占 35-36%。这是因为注意力和标准 FFN 的计算量都大致与 $d_{model}^2$ 成正比，所以它们会按比例同步增长。

---

(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?

我们将 GPT-2 XL 的 `context_length` (S) 从 1,024 增加到 16,384 (16倍)。

模型的 FLOPs 主要由两部分构成：与 S 线性相关的部分（所有投影和 FFN）和与 $S^2$ 平方相关的部分（注意力分数计算）。

* **线性部分 FLOPs**: $O(S \cdot d_{model}^2)$
* **平方部分 FLOPs**: $O(S^2 \cdot d_{model})$

当 S 很小时，线性部分占主导。但当 S 变得非常大时，平方部分的增长速度会快得多。
将上下文长度增加 16 倍，总 FLOPs 会大幅增加。更重要的是，**计算成本的瓶颈会从 FFN 和线性投影转移到注意力分数计算上**，因为这部分的计算量增长了 $16^2 = 256$ 倍，而其他部分只增长了 16 倍。

---

### 4 Training a Transformer LM

---

#### Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)
Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.
我们继续使用 float32（4bit）和上一问中 GPT-2 XL 的模型配置（`P ≈ 2.13` 十亿参数）来进行计算。

(a) How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume d_ff = 4 ×d_model.

For simplicity, when calculating memory usage of activations, consider only the following compo-
nents:
- Transformer block
    - RMSNorm(s)
    - Multi-head self-attention sublayer: QKV projections, Q⊤K matrix multiply, softmax,
weighted sum of values, output projection.
    - Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
- final RMSNorm
- output embedding
- cross-entropy on logits

---

训练过程中的峰值内存主要由四部分构成：**模型参数**、**梯度**、**优化器状态**和**激活值**。

* **模型参数 (Parameters)**: 存储模型权重所需的内存。
    * $P \times 4$ 字节
* **梯度 (Gradients)**: 在反向传播后，每个参数都需要存储其对应的梯度，大小与参数完全相同。
    * $P \times 4$ 字节
* **优化器状态 (Optimizer State)**: AdamW 为每个参数维护两个“动量”状态（$m$ 和 $v$），它们都是 float32 类型。
    * $2 \times P \times 4$ 字节
* **激活值 (Activations)**: 这是最复杂的一部分，是在前向传播过程中，为了反向传播而需要临时存储的中间结果。根据作业的要求，我们只考虑几个关键部分的最大值，这是一个简化的估算：
    * **注意力分数 ($QK^T$)**: 这是注意力机制中最大的中间产物之一，其大小与序列长度的平方成正比。
        * $L \times (B \times H \times S^2) \times 4$ 字节  (其中 L=层数, B=批大小, H=头数, S=序列长度)
    * **FFN 中间激活**:
        * $L \times (B \times S \times d_{ff}) \times 4$ 字节
    * **最终 Logits**:
        * $(B \times S \times V) \times 4$ 字节 (其中 V=词汇表大小)

**代数表达式汇总**:

* **参数内存**: $M_{params} = 4P$
* **梯度内存**: $M_{grads} = 4P$
* **优化器状态内存**: $M_{optim} = 8P$
* **激活值内存 (估算)**: $M_{acts} \approx 4 \times (L \cdot B \cdot S \cdot d_{ff} + L \cdot B \cdot H \cdot S^2 + B \cdot S \cdot V)$
* **总峰值内存**: $M_{total} = M_{params} + M_{grads} + M_{optim} + M_{acts} = 16P + M_{acts}$

---

(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?

现在我们代入 GPT-2 XL 的具体数值，来估算在 80GB 内存中能容纳的最大批处理大小。

* **模型固定内存**: $16P = 16 \times 2.13 \times 10^9 \approx 34.08 \times 10^9$ 字节 $\approx \textbf{31.7 GB}$。这是与批处理大小无关的固定开销。
* **激活值内存 (与批大小 B 相关)**: 让我们计算 `a`，即每个批次项（`B=1`）所需的激活值内存。
    * `L=48`, `S=1024`, `d_ff=6400`, `H=25`, `V=50257`
    * $M_{acts} \approx 4 \times B \times (48 \cdot 1024 \cdot 6400 + 48 \cdot 25 \cdot 1024^2 + 1024 \cdot 50257)$
    * $M_{acts} \approx 4 \times B \times (3.14 \times 10^8 + 12.58 \times 10^8 + 0.51 \times 10^8) \approx 4 \times B \times (16.23 \times 10^8)$
    * $M_{acts} \approx B \times 6.49 \times 10^9$ 字节 $\approx B \times \textbf{6.05 GB}$。

**总内存表达式**:
$TotalMemory(GB) = 6.05 \times B + 31.7$

**求解最大批大小 B**:
$80 \text{ GB} = 6.05 \times B + 31.7 \text{ GB}$
$B = (80 - 31.7) / 6.05 \approx 8.0$

因此，在 80GB 内存的限制下，你最多可以使用的批处理大小为 **8**。

---

(c) How many FLOPs does running one step of AdamW take?

AdamW 的 `step` 函数主要由对每个参数的逐元素操作构成。对于总共 `P` 个参数，主要的计算如下：
* 更新一阶矩 `m`：约 $2P$ FLOPs (1次乘法，1次加法)
* 更新二阶矩 `v`：约 $3P$ FLOPs (1次平方，1次乘法，1次加法)
* 参数更新（含 `sqrt`）：约 $3P$ FLOPs
* 权重衰减：约 $2P$ FLOPs

**总计**: $2P + 3P + 3P + 2P = \textbf{10P}$ FLOPs。
这相对于前向和反向传播的计算量（万亿级别）来说非常小。

---

(d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022]. An NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022], assume that the backward pass has twice the FLOPs of the forward pass

1.  **每一步的 FLOPs**:
    * 前向传播 (fwd): $\approx 4.52$ TFLOPs / 样本 (来自上一题)
    * 反向传播 (bwd): $\approx 2 \times \text{fwd} = 9.04$ TFLOPs / 样本
    * 总计/样本: $4.52 + 9.04 = 13.56$ TFLOPs
    * **每一步总 FLOPs**: $13.56 \text{ TFLOPs/样本} \times 1024 \text{ 样本/批} \approx 1.39 \times 10^{16}$ FLOPs

2.  **GPU 的有效计算速度**:
    * A100 峰值 (float32): $19.5$ TFLOPs/s
    * 50% MFU: $0.5 \times 19.5 = 9.75$ TFLOPs/s $= 9.75 \times 10^{12}$ FLOPs/s

3.  **总训练时间**:
    * 总 FLOPs: $(1.39 \times 10^{16} \text{ FLOPs/步}) \times 400,000 \text{ 步} = 5.56 \times 10^{21}$ FLOPs
    * 总秒数: $\frac{5.56 \times 10^{21} \text{ FLOPs}}{9.75 \times 10^{12} \text{ FLOPs/s}} \approx 5.7 \times 10^8$ 秒
    * 转换为天: $\frac{5.7 \times 10^8 \text{ 秒}}{3600 \text{ 秒/小时} \times 24 \text{ 小时/天}} \approx \textbf{6,597 天}$

在**一台 A100** 上用 **1024 的批大小**训练 GPT-2 XL 是不现实的。训练大模型需要大规模的 GPU 集群并行计算，才能在合理的时间内完成。

---

#### Problem (experiment_log): Experiment logging (3 points)
- u can find them in my `log/` folder

---

### Ablations

我观察到的这几个ablation study的结果基本都是收敛困难，训练中loss不稳定，最后同样50000step下在训练集和验证集上获得的loss都更高. 所以想更详细地知道它们具体的行为和原因，求助 Gemini

---

#### Problem (layer_norm_ablation): Remove RMSNorm and train (1 point) (1 H100 hr)
移除 RMSNorm 基本上等于拆掉了 Transformer 的“稳定器”和“减震器”。这会导致两个严重的问题：

- 内部协变量偏移 (Internal Covariate Shift): 在深度网络中，每一层的参数在训练中都会更新，这导致它输出的数据分布也在不断变化。对于下一层来说，它刚要适应前一层的输出分布，这个分布就又变了，就像在追一个不断移动的目标。归一化层（如 RMSNorm）通过将每一层的输出强制拉回到一个相对稳定的分布上（均值为0，方差为1，或者在 RMSNorm 中是均方根值为1），极大地缓解了这个问题，让每一层都能在一个更稳定的基础上进行学习。没有了它，训练自然会变得困难和不稳定。


- 梯度爆炸/消失: Transformer Block 中的残差连接 x + Attention(Norm(x)) 允许梯度在反向传播时顺畅地流动。但如果没有 Norm 层，Attention(x) 的输出可能会有非常大或非常小的数值。当这些未经控制的数值在数十个 Block 中累积传递时，梯度就很容易变得过大（爆炸）或过小（消失），导致模型无法有效更新参数。作业中也提到，归一化层对于提升训练稳定性至关重要 。

---

#### Problem (pre_norm_ablation): Implement post-norm and train (1 point) (1 H100 hr)
这个问题比上一个更微妙，关键在于梯度流的通畅程度。

- Pre-Norm (我们的基准模型): 结构是 x + Attention(Norm(x))。在反向传播时，损失的梯度可以直接通过 + 这个残差连接的“高速公路”无障碍地向前传递，几乎不会衰减。另一条分支的梯度需要穿过 Attention 和 Norm 层，但这条“高速公路”保证了总有清晰的梯度信号能传到前面的层。这正是 Pre-Norm 架构能极大改善梯度流和训练稳定性的原因 。

- Post-Norm (消融实验): 结构是 Norm(x + Attention(x))。在这种结构下，梯度在反向传播时，必须先穿过 Norm 层，然后才能到达残差连接的分叉口。Norm 层的计算会改变梯度的数值大小，可能会削弱梯度信号。在非常深的网络中，这种每层一次的削弱累积起来，就会使得深层网络的训练变得更加困难，通常需要更精细的学习率预热（warmup）策略来防止训练早期发散。

---

#### Problem (no_pos_emb): Implement NoPE (1 point) (1 H100 hr)
语言的本质在于顺序。“我打他”和“他打我”的意思天差地别。标准的自注意力机制在计算时，其实是将输入看作一个无序的“词袋”，它本身无法感知 token 的位置信息。

- RoPE 的作用: RoPE 通过“旋转”Query 和 Key 向量的方式，将绝对位置信息巧妙地编码成了相对位置信息 。经过旋转后，两个 token 之间的注意力分数不仅取决于它们的内容，还取决于它们的相对距离，这让模型能够理解顺序和语法结构。


- NoPE 的困境: 拿掉 RoPE 后，模型就失去了最主要的位置信息来源。正如作业中提到的，因果掩码（Causal Mask）确实提供了一种微弱的、隐式的位置信号 （例如，位置5的token能看到5个前面的词，而位置10的token能看到10个），但模型需要从这种信号中间接学习顺序关系，这比直接使用 RoPE 要困难得多。因此，NoPE 模型的性能会大打折扣，虽然不至于完全无法训练，但它很难掌握对顺序要求高的语言任务，导致最终损失更高。

---

#### Problem (swiglu_ablation): SwiGLU vs. SiLU (1 point) (1 H100 hr)
这个实验完美地证明了**门控机制 (Gating Mechanism)** 的威力。

* **简单 SiLU FFN**: 结构是 $W_2(SiLU(W_1x))$。信息流是固定的：输入 `x` 经过 `W1` 变换，通过 `SiLU` 激活，再经过 `W2` 输出。

* **SwiGLU FFN**: 结构是 $W_2(SiLU(W_1x) \odot W_3x)$。这里多了一个并行的 $W_3x$ 分支。这个分支的作用就像一个动态的“阀门”或“门控”。它根据输入 `x` 的内容，决定 $SiLU(W_1x)$ 分支中哪些信息是重要的、应该被通过，哪些信息不重要、应该被抑制（通过逐元素相乘时乘以一个接近0的数）。这种数据依赖的门控机制，让 FFN 变得更加灵活和强大，能够为不同的 token 定制不同的信息处理路径。正如 Shazeer 的论文所说，虽然其中的原理不完全清晰，但实验证明它确实非常有效，他将其归功于“神之恩典” (divine benevolence) 

