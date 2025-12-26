# Windows 训练脚本使用说明

## 📋 文件说明

已将原始的 Linux shell 脚本 `muge_finetune_vit-b-16_rbt-base.sh` 转换为 Windows 版本：

1. **`muge_finetune_vit-b-16_rbt-base.bat`** - Windows 批处理文件版本
2. **`muge_finetune_vit-b-16_rbt-base.ps1`** - PowerShell 版本（推荐）

## 🚀 使用方法

### 方法1：使用批处理文件（.bat）

**直接运行：**
```cmd
run_scripts\muge_finetune_vit-b-16_rbt-base.bat
```

**指定数据路径：**
```cmd
run_scripts\muge_finetune_vit-b-16_rbt-base.bat datapath
```

### 方法2：使用 PowerShell 脚本（.ps1，推荐）

**直接运行：**
```powershell
.\run_scripts\muge_finetune_vit-b-16_rbt-base.ps1
```

**指定数据路径：**
```powershell
.\run_scripts\muge_finetune_vit-b-16_rbt-base.ps1 -DATAPATH datapath
```

## ⚙️ 配置说明

### 主要配置参数

脚本中的主要配置参数与原始 shell 脚本保持一致：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GPUS_PER_NODE` | 1 | 每个节点的GPU数量 |
| `WORKER_CNT` | 1 | GPU工作节点数量 |
| `MASTER_ADDR` | localhost | 主节点地址 |
| `MASTER_PORT` | 8514 | 通信端口 |
| `RANK` | 0 | 当前节点排名 |
| `batch_size` | 128 | 训练批次大小 |
| `lr` | 5e-5 | 学习率 |
| `max_epochs` | 3 | 最大训练轮数 |
| `vision_model` | ViT-B-16 | 视觉模型 |
| `text_model` | RoBERTa-wwm-ext-base-chinese | 文本模型 |

### 修改配置

**批处理文件（.bat）：**
直接编辑文件中的 `set` 命令，例如：
```bat
set batch_size=64
set lr=1e-5
```

**PowerShell 文件（.ps1）：**
直接编辑文件中的变量，例如：
```powershell
$batch_size = 64
$lr = "1e-5"
```

## 🔧 与原脚本的主要区别

### 1. 环境变量设置
- ✅ 自动设置 `USE_LIBUV=0`（Windows PyTorch 分布式训练必需）
- ✅ 自动激活 conda 环境（批处理文件版本）

### 2. 路径格式
- ✅ 使用 Windows 路径分隔符 `\`
- ✅ 支持相对路径和绝对路径

### 3. 命令格式
- ✅ 使用 `python` 而不是 `python3`
- ✅ 使用 `--` 分隔符避免参数歧义

### 4. 额外功能
- ✅ 添加了错误处理和状态提示
- ✅ PowerShell 版本自动查找 conda 可执行文件
- ✅ 支持可选的梯度检查点（节省显存）

## 📝 注意事项

### 1. Conda 环境
- 批处理文件默认使用 `stable-diffusion-webui` 环境
- 如需修改，编辑脚本中的 `conda activate` 行

### 2. 单GPU训练
- 默认配置为单GPU训练（`GPUS_PER_NODE=1`）
- 如需多GPU训练，修改 `GPUS_PER_NODE` 参数

### 3. 数据路径
- 默认使用 `datapath` 作为数据根目录
- 可通过命令行参数指定其他路径

### 4. 显存优化
- 如需启用梯度检查点，取消注释 `grad_checkpointing` 相关行
- 可以减小 `batch_size` 以降低显存占用

## 🐛 常见问题

### Q: 提示找不到 conda？
**A:** PowerShell 版本会自动查找 conda，如果找不到，请手动激活环境后运行命令。

### Q: 训练时出现 USE_LIBUV 错误？
**A:** 脚本已自动设置该环境变量，如果仍有问题，请确保在运行前设置了 `USE_LIBUV=0`。

### Q: 如何修改训练参数？
**A:** 直接编辑脚本文件中的相应变量即可。

### Q: 支持多GPU训练吗？
**A:** 支持，修改 `GPUS_PER_NODE` 参数即可，但需要确保系统有足够的GPU。

## 📊 训练输出

训练日志和模型检查点将保存在：
```
${DATAPATH}/experiments/muge_finetune_vit-b-16_roberta-base_bs128_8gpu/
```

查看训练日志：
```powershell
Get-ChildItem datapath\experiments\muge_finetune_vit-b-16_roberta-base_bs128_8gpu\*.log | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 | 
    Get-Content -Tail 50 -Wait
```

## 🔄 与原脚本的对应关系

| Shell 脚本 | Windows 批处理 | PowerShell |
|-----------|----------------|------------|
| `bash script.sh` | `script.bat` | `.\script.ps1` |
| `${VAR}` | `%VAR%` | `$VAR` |
| `export VAR=value` | `set VAR=value` | `$VAR = "value"` |
| `python3` | `python` | `python` |
| `/path/to` | `\path\to` | `\path\to` |

---

**最后更新**：2024年12月

