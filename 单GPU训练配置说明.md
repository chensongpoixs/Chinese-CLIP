# 单GPU训练配置说明

## 概述

即使只有1个GPU，Chinese-CLIP的训练代码仍然使用分布式训练框架。这样可以保持代码的一致性，并且便于后续扩展到多GPU训练。

## 单GPU训练的配置参数

### 1. 使用 torch.distributed.launch 启动（推荐）

```bash
python -m torch.distributed.launch \
    --nproc_per_node=1 \      # 单GPU：每个节点1个进程
    --nnodes=1 \               # 单节点：只有1个节点
    --node_rank=0 \            # 节点排名：0（只有1个节点）
    --master_addr=localhost \  # 主节点地址：localhost（单机训练）
    --master_port=8514 \       # 通信端口：8514（任意可用端口）
    -- \
    cn_clip/training/main.py [训练参数...]
```

### 2. init_process_group 参数说明

对于单GPU训练，`init_process_group` 的参数配置如下：

```python
dist.init_process_group(
    backend="nccl",  # 或 "gloo"（Windows兼容性更好）
    # 其他参数从环境变量自动读取：
    #   - rank=0（只有1个进程）
    #   - world_size=1（只有1个进程）
    #   - init_method="env://"（从环境变量读取）
)
```

### 3. 环境变量设置

单GPU训练时，`torch.distributed.launch` 会自动设置以下环境变量：

| 环境变量 | 单GPU训练的值 | 说明 |
|---------|--------------|------|
| `LOCAL_RANK` | `0` | 本地GPU编号（使用GPU 0） |
| `RANK` | `0` | 全局进程排名（只有1个进程） |
| `WORLD_SIZE` | `1` | 总进程数（只有1个进程） |
| `MASTER_ADDR` | `localhost` | 主节点地址 |
| `MASTER_PORT` | `8514` | 通信端口 |

## 不同后端的配置

### NCCL 后端（推荐，NVIDIA GPU）

```python
dist.init_process_group(backend="nccl")
```

**优点：**
- 性能最优
- 专为NVIDIA GPU优化
- 支持多GPU、多节点训练

**缺点：**
- Windows上可能不可用
- 仅支持CUDA设备

**适用场景：**
- Linux系统
- NVIDIA GPU
- 单GPU或多GPU训练

### Gloo 后端（Windows兼容）

```python
dist.init_process_group(backend="gloo")
```

**优点：**
- 跨平台支持（Windows/Linux/Mac）
- 支持CPU和GPU
- Windows上兼容性更好

**缺点：**
- 性能较NCCL低
- 主要用于单机训练

**适用场景：**
- Windows系统
- 单GPU训练
- NCCL不可用时

## 完整的单GPU训练命令示例

### Windows 批处理文件

```bat
@echo off
set USE_LIBUV=0
call conda activate stable-diffusion-webui

python -m torch.distributed.launch ^
    --nproc_per_node=1 ^
    --nnodes=1 ^
    --node_rank=0 ^
    --master_addr=localhost ^
    --master_port=8514 ^
    -- ^
    cn_clip/training/main.py ^
    --train-data=datapath/datasets/MUGE/lmdb/train ^
    --val-data=datapath/datasets/MUGE/lmdb/valid ^
    --num-workers=4 ^
    --valid-num-workers=4 ^
    --resume=datapath/pretrained_weights/clip_cn_vit-b-16.pt ^
    --reset-data-offset ^
    --reset-optimizer ^
    --logs=datapath/experiments/ ^
    --name=muge_finetune_vit-b-16_roberta-base_bs48_1gpu ^
    --save-step-frequency=999999 ^
    --save-epoch-frequency=1 ^
    --log-interval=10 ^
    --report-training-batch-acc ^
    --context-length=52 ^
    --warmup=100 ^
    --batch-size=48 ^
    --valid-batch-size=48 ^
    --valid-step-interval=1000 ^
    --valid-epoch-interval=1 ^
    --lr=3e-06 ^
    --wd=0.001 ^
    --max-epochs=1 ^
    --vision-model=ViT-B-16 ^
    --use-augment ^
    --grad-checkpointing ^
    --text-model=RoBERTa-wwm-ext-base-chinese
```

### Linux/Mac Shell 脚本

```bash
#!/bin/bash
export USE_LIBUV=0

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=8514 \
    -- \
    cn_clip/training/main.py \
    --train-data=datapath/datasets/MUGE/lmdb/train \
    --val-data=datapath/datasets/MUGE/lmdb/valid \
    [其他参数...]
```

## 修改代码以支持单GPU（可选）

如果需要在代码中显式处理单GPU情况，可以这样修改：

```python
# 检查是否是单GPU训练
is_single_gpu = int(os.environ.get("WORLD_SIZE", "1")) == 1

if is_single_gpu:
    # 单GPU训练的特殊处理
    print("Single GPU training detected")
    # 可以调整某些参数，如batch size等
else:
    # 多GPU训练
    print(f"Multi-GPU training: {os.environ.get('WORLD_SIZE')} GPUs")
```

## 常见问题

### Q1: 单GPU训练必须使用分布式框架吗？

**A:** 不是必须的，但推荐使用，因为：
1. 代码已经设计为使用分布式框架
2. 保持代码一致性
3. 便于后续扩展到多GPU

### Q2: Windows上NCCL不可用怎么办？

**A:** 改用 `backend="gloo"`：
```python
dist.init_process_group(backend="gloo")
```

### Q3: 单GPU训练的性能会受影响吗？

**A:** 不会。单GPU训练时，分布式框架的开销很小，几乎可以忽略。

### Q4: 如何验证单GPU训练配置正确？

**A:** 检查输出：
- `args.rank: 0`
- `args.world_size: 1`
- `args.local_device_rank: 0`
- 没有连接错误

### Q5: 可以不用 torch.distributed.launch 吗？

**A:** 可以，但需要手动设置环境变量：
```python
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8514'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'
```

## 总结

单GPU训练的配置要点：
1. ✅ 使用 `--nproc_per_node=1`
2. ✅ 使用 `--master_addr=localhost`
3. ✅ 使用 `backend="nccl"`（Linux）或 `backend="gloo"`（Windows）
4. ✅ 其他参数由 `torch.distributed.launch` 自动设置
5. ✅ 代码会自动处理单GPU情况

