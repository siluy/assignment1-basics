import torch
import numpy as np
import os
import time

from .tokenizer import Tokenizer
from .Transformer import TransformerLM
from .AdamW import AdamW, clip_grad_norm, consine_schedule
from cross_entropy import cross_entropy_loss
from .data_loader import get_batch
from .checkpointing import save_checkpoint, load_checkpoint

TRAIN_DATA_PATH = "/home/siluyang/CS336/assignment1-basics/data/tinystories_tra_encoded.npy"
VALID_DATA_PATH = "/home/siluyang/CS336/assignment1-basics/data/tinystories_val_encoded.npy"
CHECKPOINT_DIR = "/home/siluyang/CS336/assignment1-basics/ckpt"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint.pt")

VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = int((8 / 3 * D_MODEL) / 64) * 64
ROPE_THETA = 10000

BATCH_SIZE = 64
TOTAL_STEPS = 50000
LR_MAX = 2e-4
LR_MIN = 1e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1

LOG_INTERVAL = 100
EVAL_INTERVAL = 500
EVAL_STEPS = 100
SAVE_INTERVAL = 1000

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备：{device}")

    print("正在加载数据")
    train_data = np.load(TRAIN_DATA_PATH, mmap_mode='r')
    valid_data = np.load(VALID_DATA_PATH, mmap_mode='r')

    model = TransformerLM(
        vocab_size=VOCAB_SIZE, context_length=CONTEXT_LENGTH, d_model=D_MODEL,
        num_layers=NUM_LAYERS, num_heads=NUM_HEADS, d_ff=D_FF, theta=ROPE_THETA,
        device=device, dtype=torch.float32
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)

    start_iter = 0
    if os.path.exists(CHECKPOINT_PATH):
        start_iter = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
        print(f"成功从ckpt恢复，将从第{start_iter}步继续")
    else:
        print("未找到ckpt，将从头开始训练")
    
    model.train()
    start_time = time.time()

    for step in range(start_iter, TOTAL_STEPS):
        current_lr = consine_schedule(step, LR_MAX, LR_MIN, WARMUP_STEPS, TOTAL_STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        x, y = get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, device)

        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if (step + 1) % LOG_INTERVAL == 0:
            elapsed_time = time.time() - start_time
            print(f"step [{step+1}/{TOTAL_STEPS}] | loss: {loss.item():.4f} | "
                  f"lr: {current_lr:.6f} | time: {elapsed_time:.2f}s")
            start_time = time.time()

        if (step + 1) % EVAL_INTERVAL == 0:
            model.eval()
            print("-" * 20 + " eval " + "-" * 20)
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(EVAL_STEPS):
                    vx, vy = get_batch(valid_data, BATCH_SIZE, CONTEXT_LENGTH, device)
                    vlogits = model(vx)
                    val_loss += cross_entropy_loss(vlogits, vy).item()
            val_loss /= EVAL_STEPS
            print(f"validation set loss: {val_loss:.4f}")
            print("-" * 45)
            model.train()

        if (step + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, step + 1, CHECKPOINT_PATH)

    print("训练完成")