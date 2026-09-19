# Copyright Pathway Technology, Inc.

import os
from contextlib import nullcontext

import bdh
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# On a Mac you can also try
# device=torch.device('mps')

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)
scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
print(f"Using device: {device} with dtype {dtype}")


# Configuration
BDH_CONFIG = bdh.BDHConfig()
BLOCK_SIZE = 512
BATCH_SIZE = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 100

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")


# Fetch the tiny Shakespeare dataset
def fetch_data():
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)


# Cached train/val byte tensors (pre-tensorized once from memmap).
_train_data = None
_val_data = None


def _load_splits():
    """Load corpus once: train = first 90%, val = last 10% (same as before)."""
    global _train_data, _val_data
    if _train_data is not None:
        return
    mm = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    n = len(mm)
    split = int(0.9 * n)
    # Pre-tensorize int64 once; tiny Shakespeare is ~1MB → ~8MB resident.
    _train_data = torch.from_numpy(np.asarray(mm[:split], dtype=np.int64))
    _val_data = torch.from_numpy(np.asarray(mm[split:], dtype=np.int64))


def get_batch(split):
    # Vectorized window gather — no Python list of from_numpy per sample.
    _load_splits()
    data = _train_data if split == "train" else _val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    offsets = torch.arange(BLOCK_SIZE + 1)
    windows = data[ix.unsqueeze(1) + offsets.unsqueeze(0)]  # (B, BLOCK_SIZE+1)
    x = windows[:, :-1]
    y = windows[:, 1:]
    if torch.cuda.is_available():
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def eval(model):
    model.eval()


if __name__ == "__main__":
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")

    loss_acc = 0
    loss_steps = 0
    for step in range(MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)
        x, y = get_batch("train")
        loss_acc += loss
        loss_steps += 1
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if step % LOG_FREQ == 0:
            print(f"Step: {step}/{MAX_ITERS} loss {loss_acc.item() / loss_steps:.3}")
            loss_acc = 0
            loss_steps = 0
    print("Training done, now generating a sample ")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)
