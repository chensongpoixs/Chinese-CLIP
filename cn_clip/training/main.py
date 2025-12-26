"""
Chinese-CLIP 训练主程序
功能：实现CLIP模型的中文图文对比学习训练，支持分布式训练、混合精度、梯度检查点等优化技术
"""

from math import ceil
import os
import sys
import io

# ============================================================================
# 设置标准输出编码为UTF-8（Windows兼容性）
# ============================================================================
# Windows控制台默认使用cp1252编码，无法显示中文字符
# 这里设置标准输出和错误输出为UTF-8编码
if sys.platform == 'win32':
    # 设置标准输出编码
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    # 设置标准错误输出编码
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# Windows环境配置
# ============================================================================
# 设置 USE_LIBUV=0 以禁用 libuv 支持（Windows上PyTorch分布式训练必需）
os.environ["USE_LIBUV"] = "0"

# 设置分布式训练的主节点地址和端口（如果未通过命令行指定）
# 这些值通常由 torch.distributed.launch 自动设置，这里作为备用
# 
# 【Windows特殊处理】
# Windows上如果MASTER_ADDR是主机名（如"chensong"），可能无法正确解析
# 因此强制使用localhost或127.0.0.1，确保单机训练能够正常工作
# 如果环境变量中已经设置了MASTER_ADDR，则优先使用环境变量的值
# 但如果环境变量是主机名，则替换为localhost
if 'MASTER_ADDR' not in os.environ or os.environ.get('MASTER_ADDR', '').strip() == '':
    os.environ['MASTER_ADDR'] = 'localhost'  # 使用localhost而不是127.0.0.1，兼容性更好
elif os.environ.get('MASTER_ADDR', '').lower() not in ['localhost', '127.0.0.1', '0.0.0.0']:
    # 如果MASTER_ADDR是主机名（如"chensong"），在Windows上可能无法解析
    # 对于单机训练，强制使用localhost
    import platform
    if platform.system() == 'Windows':
        print(f"Warning: MASTER_ADDR is set to '{os.environ['MASTER_ADDR']}' which may not work on Windows.")
        print("Changing MASTER_ADDR to 'localhost' for single-node training.")
        os.environ['MASTER_ADDR'] = 'localhost'

# 设置端口（如果未通过命令行指定）
if 'MASTER_PORT' not in os.environ or os.environ.get('MASTER_PORT', '').strip() == '':
    os.environ['MASTER_PORT'] = '8514'   # 默认端口，确保这个端口没有被占用

# ============================================================================
# 标准库导入
# ============================================================================
import logging          # 日志记录
from pathlib import Path  # 路径处理
import json            # JSON配置文件解析
import time            # 时间处理
from time import gmtime, strftime  # 时间格式化
import importlib.util  # 动态模块导入（用于检查flash_attn是否安装）
import inspect         # 用于获取当前代码行号

# ============================================================================
# PyTorch相关导入
# ============================================================================
import torch
from torch import optim              # 优化器（AdamW等）
import torch.distributed as dist     # 分布式训练支持
import torch.backends.cudnn as cudnn # CUDA深度神经网络库后端
from torch.cuda.amp import GradScaler  # 混合精度训练的梯度缩放器

# ============================================================================
# Chinese-CLIP项目内部模块导入
# ============================================================================
from cn_clip.clip import load  # 模型权重加载函数
from cn_clip.clip.model import convert_weights, convert_state_dict, resize_pos_embed, CLIP  # CLIP模型定义
from cn_clip.training.train import train, evaluate  # 训练和评估函数
from cn_clip.training.data import get_data  # 数据加载器构建
from cn_clip.training.params import parse_args  # 命令行参数解析
from cn_clip.training.logger import setup_primary_logging, setup_worker_logging  # 日志系统设置
from cn_clip.training.scheduler import cosine_lr  # 余弦学习率调度器






import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available()) # 查看CUDA是否可用，如果True表示可以使用
print(torch.cuda.device_count()) # 查看可用的CUDA数量，0表示有一个
print(torch.version.cuda) # 查看CUDA的版本号



# ============================================================================
# 辅助函数定义
# ============================================================================

def convert_models_to_fp32(model):
    """
    将模型的所有参数和梯度转换为FP32格式
    
    参数:
        model: PyTorch模型
    说明:
        用于混合精度训练（AMP）或FP32训练时，确保模型参数为FP32格式
        参考: https://github.com/openai/CLIP/issues/83
    """
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def is_master(args):
    """
    判断当前进程是否为主进程（rank=0）
    
    参数:
        args: 训练参数对象，包含rank信息
    返回:
        bool: 如果是主进程返回True，否则返回False
    说明:
        在分布式训练中，只有主进程（rank=0）负责保存检查点、记录日志等操作
    """
    return args.rank == 0


def torch_version_str_compare_lessequal(version1, version2):
    """
    比较两个PyTorch版本号，判断version1是否小于等于version2
    
    参数:
        version1: 版本号字符串，如 "1.8.0"
        version2: 版本号字符串，如 "2.0.0"
    返回:
        bool: 如果version1 <= version2返回True，否则返回False
    说明:
        用于检查PyTorch版本兼容性，例如梯度检查点功能需要PyTorch >= 1.8.0
    """
    # 解析版本号，忽略"+"后的构建信息（如"1.8.0+cu111"）
    v1 = [int(entry) for entry in version1.split("+")[0].split(".")]
    v2 = [int(entry) for entry in version2.split("+")[0].split(".")]
    assert len(v1) == 3, "Cannot parse the version of your installed pytorch! ({})".format(version1)
    assert len(v2) == 3, "Illegal version specification ({}). Should be in 1.X.Y format.".format(version2)
    return sorted([v1, v2])[0] == v1

def main():
    """
    训练主函数
    
    功能流程：
    1. 解析命令行参数
    2. 初始化分布式训练环境
    3. 设置输出路径和日志系统
    4. 构建CLIP模型
    5. 配置模型精度和优化选项
    6. 包装为分布式数据并行模型
    7. 初始化数据集和数据加载器
    8. 初始化优化器和学习率调度器
    9. 记录和保存超参数
    10. 加载检查点（如果存在）
    11. CUDA优化设置
    12. 确定是否保存日志和检查点
    13. 加载教师模型（用于知识蒸馏，可选）
    14. 执行训练循环
    15. 保存检查点
    """
    # 获取当前文件名（用于调试和日志追踪）
    current_file = os.path.basename(__file__)
    
    # 获取当前行号（用于调试和日志追踪）
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ========== 开始执行 main() 函数 ==========")
    
    # ========================================================================
    # 步骤1: 解析命令行参数
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤1: 解析命令行参数...")
    args = parse_args()
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ 参数解析完成")

    # ========================================================================
    # 步骤2: 初始化分布式训练环境
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤2: 初始化分布式训练环境...")
    # 获取当前进程的本地GPU设备编号（由torch.distributed.launch自动设置）
    # LOCAL_RANK是当前进程在单机多GPU训练中的GPU编号（0, 1, 2, ...）
    # 对于单GPU训练，LOCAL_RANK=0（使用GPU 0）
    # 如果环境变量中没有LOCAL_RANK（不使用分布式训练），则默认为0
    args.local_device_rank =  int(os.environ.get("LOCAL_RANK", "0"))
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 获取本地GPU设备编号: LOCAL_RANK={args.local_device_rank}")
    
    # 设置当前进程使用的CUDA设备
    # 对于单GPU训练，使用GPU 0
    #torch.cuda.set_device(args.local_device_rank)
    args.device = "cuda" if torch.cuda.is_available() else "cpu";
    # 创建设备对象，指向指定的GPU
    #args.device = torch.device("cuda", args.local_device_rank)
    #args.device = torch.cuda.set_device("cuda:0");
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ CUDA设备设置完成: {args.device}")

    # ========================================================================
    # 初始化分布式进程组（Process Group）
    # ========================================================================
    # 
    # dist.init_process_group() 是PyTorch分布式训练的核心函数
    # 它的作用是建立多个进程之间的通信机制，使它们能够协同工作
    # 
    # 【工作原理】
    # 1. 每个GPU对应一个独立的进程（在分布式训练中）
    # 2. 这些进程需要能够相互通信，同步梯度、参数等
    # 3. init_process_group() 建立进程间的通信通道
    # 
    # 【参数说明】
    # backend (str): 通信后端，决定使用哪种通信协议
    #   - "nccl": NVIDIA Collective Communications Library
    #     * 专为NVIDIA GPU设计，性能最优
    #     * 支持多GPU、多节点训练
    #     * 仅支持CUDA设备，不支持CPU
    #     * 推荐用于NVIDIA GPU训练（当前使用）
    #   
    #   - "gloo": Facebook开发的跨平台后端
    #     * 支持CPU和GPU（CUDA）
    #     * 跨平台支持（Windows/Linux/Mac）
    #     * 性能较NCCL低，但兼容性更好
    #     * Windows上如果NCCL不可用，可以使用gloo
    #   
    #   - "mpi": Message Passing Interface
    #     * 需要系统安装MPI库
    #     * 主要用于HPC（高性能计算）环境
    # 
    # 【其他可选参数】（通常由torch.distributed.launch自动设置）
    # init_method (str): 初始化方法，指定如何建立进程间连接
    #   - "env://": 从环境变量读取（MASTER_ADDR, MASTER_PORT等）- 默认方式
    #   - "tcp://IP:PORT": 通过TCP连接指定主节点
    #   - "file:///path/to/file": 通过共享文件系统
    # 
    # rank (int): 当前进程的全局排名
    #   - 在所有节点所有GPU中的唯一编号（0, 1, 2, ..., world_size-1）
    #   - rank=0 通常是主进程（master），负责保存检查点、记录日志等
    #   - 通常从环境变量 RANK 自动读取
    # 
    # world_size (int): 总进程数（所有节点所有GPU的总数）
    #   - 例如：2个节点，每个节点4个GPU，则world_size=8
    #   - 通常从环境变量 WORLD_SIZE 自动读取
    # 
    # 【自动设置机制】
    # 当使用 torch.distributed.launch 启动训练时：
    #   - init_method 默认为 "env://"（从环境变量读取）
    #   - rank 和 world_size 从环境变量 RANK 和 WORLD_SIZE 自动读取
    #   - MASTER_ADDR 和 MASTER_PORT 由 --master_addr 和 --master_port 参数设置
    #   - LOCAL_RANK 由 --nproc_per_node 参数自动分配
    # 
    # 【环境变量说明】（由torch.distributed.launch自动设置）
    #   - MASTER_ADDR: 主节点IP地址（如 "localhost" 或 "192.168.1.100"）
    #   - MASTER_PORT: 主节点端口（如 "8514"）
    #   - RANK: 当前进程的全局排名
    #   - WORLD_SIZE: 总进程数
    #   - LOCAL_RANK: 当前进程在当前节点中的本地排名（0, 1, 2, ...）
    # 
    # 【注意事项】
    # 1. 所有进程必须调用相同的 init_process_group() 参数
    # 2. 必须确保所有进程能够访问 MASTER_ADDR 指定的地址
    # 3. 确保 MASTER_PORT 端口未被占用
    # 4. Windows上如果NCCL不可用，可以尝试使用 "gloo" 后端
    # 5. 单机多GPU训练时，MASTER_ADDR 通常设置为 "localhost"
    # 6. 多机多GPU训练时，MASTER_ADDR 必须设置为主节点的实际IP地址
    # 
    # 【单GPU训练配置】
    # 即使只有1个GPU，也可以使用分布式训练框架（保持代码一致性）
    # 单GPU训练的参数配置：
    #   - backend: "nccl"（NVIDIA GPU，Linux推荐）或 "gloo"（Windows兼容性更好）
    #   - init_method: "env://"（从环境变量读取，默认）
    #   - rank: 0（只有1个进程，所以rank=0）
    #   - world_size: 1（只有1个进程）
    #   - MASTER_ADDR: "localhost"（单机训练）
    #   - MASTER_PORT: 8514（任意可用端口）
    # 
    # 使用 torch.distributed.launch 启动单GPU训练时：
    #   python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=8514 -- ...
    #   环境变量会自动设置：
    #     - RANK=0（只有1个进程）
    #     - WORLD_SIZE=1（只有1个进程）
    #     - LOCAL_RANK=0（使用GPU 0）
    #     - MASTER_ADDR=localhost
    #     - MASTER_PORT=8514
    # 
    # 【当前配置】
    # - 使用 "nccl" 后端（NVIDIA GPU推荐，性能最优）
    # - 其他参数由 torch.distributed.launch 通过环境变量自动设置
    # - 这是单机单GPU、单机多GPU或多机多GPU训练的标准配置
    # 
    # 【单GPU训练的特殊说明】
    # 1. 使用 torch.distributed.launch 启动时，设置 --nproc_per_node=1
    # 2. 即使只有1个GPU，也会创建1个进程，rank=0, world_size=1
    # 3. Windows上如果NCCL不可用，可以改用 backend="gloo"
    # 4. 单GPU训练时，所有操作都在rank=0进程上执行
    # 5. 单GPU训练的性能几乎不受影响，分布式框架开销很小
    # 
    # 【故障排查】
    # 如果遇到连接问题：
    #   1. 检查 MASTER_ADDR 和 MASTER_PORT 是否正确
    #   2. 确保防火墙允许端口通信
    #   3. Windows上如果NCCL失败，尝试改用 backend="gloo"
    #   4. 确保所有进程同时启动（使用torch.distributed.launch）
    # ========================================================================
    # 单节点训练：尝试初始化单进程分布式进程组（可选）
    # ========================================================================
    # 注意：train.py 中使用了 dist.get_world_size(), dist.get_rank(), dist.all_gather() 等函数
    # 如果 Windows 上的 GLOO 后端不可用，我们将设置 aggregate=False 来避免使用分布式函数
    # 但 evaluate 函数中仍然使用了 dist.all_reduce()，需要特殊处理
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 尝试初始化单进程分布式进程组（单节点训练）...")
    
    # 检查是否已经初始化了分布式进程组
    dist_initialized = False
    if not dist.is_initialized():
        # 单节点单GPU训练：尝试初始化一个单进程的分布式进程组
        # Windows上 GLOO 后端可能不可用，如果失败则跳过初始化
        try:
            # 设置环境变量（如果未设置）
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = '127.0.0.1'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '29500'
            
            # 尝试使用 gloo 后端，使用 tcp 初始化方法
            current_line = inspect.currentframe().f_lineno
            print(f"[{current_file}:{current_line}] 尝试使用 GLOO 后端 (tcp://{os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')})...")
            
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}",
                rank=0,
                world_size=1
            )
            dist_initialized = True
            current_line = inspect.currentframe().f_lineno
            print(f"[{current_file}:{current_line}] ✓ 使用 GLOO 后端初始化单进程分布式进程组成功")
        except Exception as e:
            # 如果初始化失败，设置 aggregate=False 并跳过分布式初始化
            current_line = inspect.currentframe().f_lineno
            print(f"[{current_file}:{current_line}] 警告: 无法初始化分布式进程组: {e}")
            print(f"[{current_file}:{current_line}] 提示: Windows 上的 GLOO 后端可能不可用，将使用单节点模式（跳过聚合）")
            print(f"[{current_file}:{current_line}] 设置 aggregate=False 以避免使用分布式函数")
            args.aggregate = False  # 禁用聚合，避免使用分布式函数
            args.skip_aggregate = True
            dist_initialized = False
    else:
        dist_initialized = True
        current_line = inspect.currentframe().f_lineno
        print(f"[{current_file}:{current_line}] 分布式进程组已初始化，跳过")
    
    # 获取当前进程的全局排名（单节点训练时 rank=0）
    if dist_initialized:
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
    else:
        # 如果分布式未初始化，设置默认值
        args.rank = 0
        args.world_size = 1
    
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] args.rank: {args.rank}, args.world_size: {args.world_size}")  # 调试输出
    print("--------------------------------");
    current_line = inspect.currentframe().f_lineno
    #print(f"[{current_file}:{current_line}] args.world_size: {args.world_size}")  # 调试输出：总进程数
    print("--------------------------------");  
    
    # ========================================================================
    # 步骤3: 设置输出路径和日志系统
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤3: 设置输出路径和日志系统...")
    # 生成时间戳后缀，格式：YYYY-MM-DD-HH-MM-SS
    # 用于区分同一实验名称的不同运行实例，避免日志和检查点被覆盖
    time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    #print('')
    # 设置日志文件路径: {logsfan}/{name}/out_{时间戳}.log
    # 例如: datapath/experiments/muge_finetune_vit-b-16_roberta-base_bs128_8gpu/out_2024-12-25-16-30-00.log
    args.log_path = os.path.join(args.logsfan, args.name, "out_{}.log".format(time_suffix))
    
    # 设置检查点保存路径: {logsfan}/{name}/checkpoints/
    # 所有模型检查点将保存在此目录下
    args.checkpoint_path = os.path.join(args.logsfan, args.name, "checkpoints")
    
    # 只有主进程（rank=0）创建输出目录，避免多进程重复创建
    # 在分布式训练中，所有进程共享文件系统，只需主进程创建目录即可
    #if is_master(args):
    if True:
        for dirname in [args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)  # exist_ok=True表示目录已存在时不报错    

    # 验证精度设置（混合精度/FP16/FP32）
    # - 'amp': 自动混合精度（Automatic Mixed Precision），推荐使用，平衡速度和精度
    # - 'fp16': 纯FP16精度，速度最快但可能影响精度
    # - 'fp32': 纯FP32精度，精度最高但速度最慢
    assert args.precision in ['amp', 'fp16', 'fp32']

    # 设置日志级别
    # DEBUG: 详细调试信息（包括所有中间变量值）
    # INFO: 一般信息（训练进度、损失值等）
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    
    # 设置主进程日志系统
    # 主进程（rank=0）负责将日志写入文件
    # 返回一个队列，用于接收工作进程的日志消息
    log_queue = setup_primary_logging(args.log_path, args.log_level, args.rank)
    
    # 设置工作进程日志系统
    # 工作进程（rank>0）通过队列将日志消息发送给主进程
    # 主进程统一写入文件，避免多进程同时写文件造成冲突
    setup_worker_logging(args.rank, log_queue, args.log_level)
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ 日志系统设置完成")

    # ========================================================================
    # 步骤4: 构建CLIP模型
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤4: 构建CLIP模型...")
    # 加载视觉模型配置文件
    # 配置文件路径: cn_clip/clip/model_configs/{vision_model}.json
    # 例如: ViT-B-16 -> ViT-B-16.json, ViT-L/14 -> ViT-L-14.json
    # 配置文件包含：embed_dim, image_resolution, vision_layers, vision_width等
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file), f"Vision model config not found: {vision_model_config_file}"
    
    # 加载文本模型配置文件
    # 配置文件路径: cn_clip/clip/model_configs/{text_model}.json
    # 例如: RoBERTa-wwm-ext-base-chinese -> RoBERTa-wwm-ext-base-chinese.json
    # 配置文件包含：vocab_size, text_hidden_size, text_num_layers等
    text_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file), f"Text model config not found: {text_model_config_file}"
    
    # 合并视觉和文本模型配置
    # 将两个JSON配置文件的内容合并为一个字典，用于创建CLIP模型
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)  # 加载视觉模型配置（字典格式）
        
        # 处理vision_layers字段
        # 某些配置文件中vision_layers可能是字符串格式（如"[12, 12, 12, 12]"）
        # 需要转换为Python列表格式
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        
        # 合并文本模型配置到model_info
        # 文本模型的配置项会添加到model_info中，如果键名相同会被覆盖
        for k, v in json.load(ft).items():
            model_info[k] = v
    
    # 添加FlashAttention使用标志
    # FlashAttention是一种优化的注意力计算方式，可以加速训练并降低显存占用
    model_info['use_flash_attention'] = args.use_flash_attention

    # 创建CLIP模型实例
    # CLIP模型包含两个编码器：
    #   - visual: 视觉编码器（ViT或ResNet），用于编码图像
    #   - textual: 文本编码器（RoBERTa），用于编码文本
    #   两个编码器将图像和文本映射到同一个特征空间，用于计算相似度
    model = CLIP(**model_info)
    
    # 加载预训练权重（如果指定）
    # clip_weight_path: CLIP模型的预训练权重路径（包含视觉和文本编码器）
    # bert_weight_path: 仅文本编码器的预训练权重路径（可选，用于单独加载BERT权重）
    if args.clip_weight_path is not None:
        assert os.path.exists(args.clip_weight_path), "Pretrained CLIP weight not exists!"
    if args.bert_weight_path is not None:
        assert os.path.exists(args.bert_weight_path), "Pretrained BERT weight not exists!"
    
    # 加载权重到模型
    # load函数会：
    #   1. 加载CLIP权重（如果指定clip_weight_path）
    #   2. 加载BERT权重（如果指定bert_weight_path）
    #   3. 处理权重名称不匹配的情况（如添加/移除"module."前缀）
    #   4. 如果使用FlashAttention，转换权重格式
    load(model, clip_path=args.clip_weight_path, bert_path=args.bert_weight_path, use_flash_attention=args.use_flash_attention)
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ CLIP模型构建完成")

    # ========================================================================
    # 步骤5: 模型精度和优化设置
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤5: 配置模型精度和优化选项...")
    # 对于AMP或FP32训练，确保模型参数为FP32格式
    # 参考: https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)

    # 将模型移动到指定的GPU设备
    model.cuda(args.local_device_rank)
    
    # 对于FP16训练，将模型权重转换为FP16格式
    if args.precision == "fp16":
        convert_weights(model)

    # 梯度检查点：用计算时间换显存，适合显存不足的情况
    if args.grad_checkpointing:
        assert not torch_version_str_compare_lessequal(torch.__version__, "1.8.0"), \
            "Currently our grad_checkpointing is not compatible with torch version <= 1.8.0."
        model.set_grad_checkpointing()
        logging.info("Grad-checkpointing activated.")

    # FlashAttention：加速注意力计算，降低显存占用
    if args.use_flash_attention:
        assert importlib.util.find_spec("flash_attn"), "flash_attn is not installed. Install with: pip install flash-attn"
        logging.info("Using FlashAttention.")

    # 同步批归一化：在分布式训练中同步BN统计量
    if args.use_bn_sync:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 冻结视觉编码器：只训练文本编码器（用于某些迁移学习场景）
    if args.freeze_vision:
        for k, v in model.visual.named_parameters():
            v.requires_grad = False
        # 对于ResNet50，还需要冻结BN的running mean和variance
        if args.vision_model in ['RN50']:
            for m in model.visual.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()  # 设置为评估模式，不更新BN统计量
        logging.info("The visual encoder is freezed during training.")

    # ========================================================================
    # 步骤6: 单节点训练配置（不使用DistributedDataParallel）
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤6: 配置单节点训练（不使用DDP）...")
    
    # 单节点单GPU训练：直接将模型移动到GPU，不使用DistributedDataParallel
    # 这样可以避免DDP的开销，简化训练流程
    # 注意：train.py 中使用了 model.module，所以我们需要创建一个兼容的包装类
    # 或者直接使用模型（需要修改 train.py 中的 model.module 引用）
    
    # 将模型移动到指定GPU设备
    model = model.to(args.device)
    
    # 为了兼容 train.py 中可能使用 model.module 的情况
    # 创建一个简单的包装类，使 model.module 返回模型本身
    class ModelWrapper:
        """
        单节点训练模型包装类，用于兼容 train.py 中的 model.module 引用
        
        这个包装类模拟了 DistributedDataParallel 的行为，使得单节点训练时
        代码可以像使用 DDP 一样访问 model.module，但实际上直接使用原始模型
        
        关键点：
        - self.module 直接指向原始模型，这样 model.module.xxx 可以正常工作
        - __getattr__ 转发所有属性访问到原始模型
        - 实现了 PyTorch 模型需要的主要方法
        """
        def __init__(self, model):
            # 关键：self.module 直接指向原始模型，这样 model.module.xxx 可以正常工作
            self.module = model  # 兼容 train.py 中的 model.module 访问
            self._model = model  # 保存原始模型引用
        
        def __call__(self, *args, **kwargs):
            """前向传播：直接调用原始模型"""
            return self._model(*args, **kwargs)
        
        def __getattr__(self, name):
            """
            转发所有其他属性访问到原始模型
            
            这个方法会在 Python 找不到属性时被调用
            例如：model.visual, model.logit_scale 等都会通过这里转发到原始模型
            """
            # 避免递归：如果访问的是我们自己的属性，直接返回
            if name in ['module', '_model']:
                try:
                    return object.__getattribute__(self, name)
                except AttributeError:
                    pass
            
            # 转发到原始模型
            try:
                return getattr(self._model, name)
            except AttributeError:
                # 如果原始模型也没有这个属性，抛出清晰的错误
                raise AttributeError(f"'{type(self).__name__}' object and wrapped model both have no attribute '{name}'")
        
        def parameters(self):
            """返回模型参数"""
            return self._model.parameters()
        
        def named_parameters(self):
            """返回命名的模型参数"""
            return self._model.named_parameters()
        
        def state_dict(self, destination=None, prefix='', keep_vars=False):
            """返回模型状态字典"""
            return self._model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        def load_state_dict(self, state_dict, strict=True):
            """加载模型状态字典"""
            return self._model.load_state_dict(state_dict, strict=strict)
        
        def train(self, mode=True):
            """设置模型为训练模式"""
            self._model.train(mode)
            return self
        
        def eval(self):
            """设置模型为评估模式"""
            self._model.eval()
            return self
        
        def modules(self):
            """返回所有模块（用于兼容 train.py 中的检查）"""
            return self._model.modules()
        
        def children(self):
            """返回直接子模块"""
            return self._model.children()
        
        def named_children(self):
            """返回命名的直接子模块"""
            return self._model.named_children()
        
        def named_modules(self, memo=None, prefix='', remove_duplicate=True):
            """返回命名的所有模块"""
            return self._model.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate)
    
    # 包装模型以兼容 train.py 中的 model.module 引用
    model = ModelWrapper(model)
    
    # 如果使用FP16，转换权重
    if args.precision == "fp16":
        convert_weights(model._model)
    
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ 单节点训练配置完成（模型已移动到 {args.device}）")

    # ========================================================================
    # 步骤7: 初始化数据集和数据加载器
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤7: 初始化数据集和数据加载器...")
    # get_data函数会：
    #   1. 加载LMDB格式的训练数据集（如果指定train_data）
    #   2. 加载LMDB格式的验证数据集（如果指定val_data）
    #   3. 创建DataLoader，支持多进程数据加载（num_workers）
    #   4. 应用数据增强（如果启用use_augment）
    #   5. 返回包含'train'和'val'键的字典
    # epoch_id: 当前epoch编号，用于某些需要epoch相关数据增强的场景
    # max_txt_length: 文本最大长度（token数），超过此长度的文本会被截断
    data = get_data(args, epoch_id=0, max_txt_length=args.context_length)
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ 数据集和数据加载器初始化完成")

    # ========================================================================
    # 步骤8: 初始化优化器和学习率调度器
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤8: 初始化优化器和学习率调度器...")
    # 定义参数分组规则（用于不同的权重衰减策略）
    # 为什么需要分组？
    #   - BatchNorm/LayerNorm的scale和bias参数通常不使用权重衰减
    #   - 偏置（bias）参数通常不使用权重衰减
    #   - logit_scale（温度参数）通常不使用权重衰减
    #   这样可以提高训练稳定性和最终性能
    exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n : not exclude(n)

    # 获取模型所有参数并按规则分组
    # named_parameters: 返回(参数名, 参数张量)的迭代器
    # requires_grad=True: 只选择需要梯度的参数（排除冻结的参数）
    named_parameters = list(model.named_parameters())
    
    # 不使用权重衰减的参数组（BN/LN/bias/logit_scale）
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
    
    # 使用权重衰减的参数组（其他所有参数）
    rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

    if args.train_data is None:
        # 如果没有训练数据，则不创建优化器和调度器（仅用于评估）
        optimizer = None
        scheduler = None
    else:
        # 创建AdamW优化器，对不同参数组应用不同的权重衰减
        # AdamW是Adam的改进版本，使用解耦的权重衰减（decoupled weight decay）
        # 相比Adam，AdamW通常能获得更好的泛化性能
        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},  # BN/LN/bias/logit_scale不使用权重衰减
                {"params": rest_params, "weight_decay": args.wd},     # 其他参数使用权重衰减（默认0.001）
            ],
            lr=args.lr,                    # 初始学习率（peak learning rate）
            betas=(args.beta1, args.beta2), # Adam的动量参数
            #   beta1: 一阶矩估计的衰减率（默认0.9），控制梯度历史的影响
            #   beta2: 二阶矩估计的衰减率（默认0.98或0.999），控制梯度平方历史的影响
            eps=args.eps,                   # 数值稳定性参数（默认1e-6或1e-8），防止除零
        )
        
        # 计算总训练步数
        # num_batches: 一个epoch的批次数（考虑分布式训练，每个进程只看到部分数据）
        num_batches = data["train"].dataloader.num_batches
        
        if args.max_steps is not None:
            # 如果指定了最大步数（max_steps），根据步数计算需要的epoch数
            # 注意：需要考虑梯度累积频率（accum_freq）
            # 例如：如果max_steps=1000, accum_freq=2, num_batches=500
            #   则实际需要: ceil(1000 * 2 / 500) = ceil(4) = 4个epoch
            args.max_epochs = ceil(args.max_steps * args.accum_freq / num_batches)
        else:
            # 如果指定了最大epoch数（max_epochs），根据epoch数计算总步数
            # 例如：如果max_epochs=3, num_batches=500, accum_freq=2
            #   则总步数: (500 // 2) * 3 = 250 * 3 = 750步
            assert args.max_epochs is not None and args.max_epochs > 0
            args.max_steps = (num_batches // args.accum_freq) * args.max_epochs
        
        total_steps = args.max_steps
        
        # 创建余弦学习率调度器（带warmup）
        # cosine_lr函数会创建一个学习率调度器，学习率变化如下：
        #   1. Warmup阶段（前warmup步）：线性增长从0到lr
        #   2. Cosine衰减阶段（warmup步之后）：按余弦函数从lr衰减到0
        # 这种调度策略通常能获得更好的训练效果
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # 创建梯度缩放器（用于混合精度训练AMP）
    # GradScaler用于解决FP16训练中的梯度下溢问题
    # 工作原理：
    #   1. 在反向传播前，将损失值乘以一个缩放因子（scale）
    #   2. 反向传播后，梯度也会被缩放
    #   3. 如果梯度没有溢出，将梯度除以缩放因子恢复原值
    #   4. 如果梯度溢出，跳过本次更新并增大缩放因子
    # 注意：只有使用AMP（自动混合精度）时才需要scaler
    scaler = GradScaler() if args.precision == "amp" else None
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ 优化器和学习率调度器初始化完成")

    # ========================================================================
    # 步骤9: 记录和保存超参数
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤9: 记录和保存超参数...")
    # 主进程保存超参数到文件
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logsfan, args.name, "params_{}.txt".format(time_suffix))
        with open(params_file, "w", encoding="utf-8") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                f.write(f"{name}: {val}\n")

    # 所有进程记录超参数到日志
    if args.local_device_rank == 0:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
    logging.info(f"Use GPU: {args.local_device_rank} for training")
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ 超参数记录完成")

    # 关于mask_ratio的提示（FLIP策略仅支持ViT，不支持ResNet）
    if is_master(args) and args.mask_ratio > 0 and args.vision_model in ['RN50']:
        logging.info("Note: mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer. " + \
            "It will not function for ResNet backbone.")    

    # ========================================================================
    # 步骤10: 加载检查点（如果存在）
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤10: 加载检查点（如果存在）...")
    start_epoch = 0  # 起始epoch
    steps = 0        # 起始步数
    
    # 如果未指定resume路径，自动查找最新的检查点
    if args.resume is None:
        latest_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
        if os.path.isfile(latest_path):
            args.resume = latest_path
    
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logging.info(f"=> begin to load checkpoint '{args.resume}'")
            
            # 加载检查点（加载到CPU，避免GPU显存问题）
            checkpoint = torch.load(args.resume, map_location="cpu")
            
            # 过滤掉bert.pooler层的参数（某些版本可能不兼容）
            sd = {k: v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
            
            # 处理键名中的 "module." 前缀（从DDP模型保存的检查点）
            # 检查点可能包含 "module." 前缀，但我们的 ModelWrapper 模型没有这个前缀
            if sd and next(iter(sd.keys())).startswith("module."):
                # 移除 "module." 前缀
                sd = {k[len("module."):]: v for k, v in sd.items()}
                prefix_for_resize = ""  # 移除前缀后，不需要前缀
            else:
                prefix_for_resize = ""  # 没有前缀
            
            # 如果位置编码大小不匹配，通过插值调整
            resize_pos_embed(sd, model, prefix=prefix_for_resize)
            
            # 如果使用FlashAttention，转换状态字典格式
            if args.use_flash_attention:
                sd = convert_state_dict(sd)
            
            # 加载模型权重
            # 对于 ModelWrapper，需要加载到 _model
            if hasattr(model, '_model'):
                model._model.load_state_dict(sd, strict=False)
            else:
                model.load_state_dict(sd, strict=False)
            
            # 恢复epoch和步数信息，并重新加载对应epoch的数据集
            if not args.reset_data_offset:
                start_epoch = checkpoint["epoch"]
                steps = checkpoint["step"]
                data = get_data(args, 
                                epoch_id=start_epoch, 
                                max_txt_length=args.context_length)
            
            # 恢复优化器状态（如果未重置优化器）
            if not args.reset_optimizer and optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
                logging.info("=> optimizer state is restored from the checkpoint")
            
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']} @ {steps} steps)"
            )
            current_line = inspect.currentframe().f_lineno
            print(f"[{current_file}:{current_line}] ✓ 检查点加载完成: epoch {checkpoint['epoch']} @ {steps} steps")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
            current_line = inspect.currentframe().f_lineno
            print(f"[{current_file}:{current_line}] 未找到检查点，从头开始训练")
    else:
        current_line = inspect.currentframe().f_lineno
        print(f"[{current_file}:{current_line}] 未指定检查点路径，从头开始训练")

    # ========================================================================
    # 步骤11: CUDA优化设置
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤11: 配置CUDA优化设置...")
    cudnn.benchmark = True        # 启用CUDNN自动调优，加速训练（但可能降低可复现性）
    cudnn.deterministic = False   # 允许非确定性算法，提高性能
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ CUDA优化设置完成")

    # ========================================================================
    # 步骤12: 确定是否保存日志和检查点
    # ========================================================================
    # 只有主进程（rank=0）才保存日志和检查点，避免多进程重复保存
    args.should_save = (args.logsfan is not None and args.logsfan != '' and args.logsfan.lower() != 'none') and is_master(args)
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤12: 确定保存设置 (should_save={args.should_save})...")

    # ========================================================================
    # 步骤13: 加载教师模型（用于知识蒸馏）
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    if args.distillation:
        print(f"[{current_file}:{current_line}] 步骤13: 加载教师模型（知识蒸馏）...")
    else:
        print(f"[{current_file}:{current_line}] 步骤13: 跳过教师模型加载（未启用知识蒸馏）")
    if args.distillation:
        try:
            from modelscope.models import Model
        except:
            raise ImportError("modelscope is not installed. Please install it by `pip install modelscope`.")

        teacher_model_dict = {
            "damo/multi-modal_team-vit-large-patch14_multi-modal-similarity" : {"model": "image_model"},
            "damo/multi-modal_rleg-vit-large-patch14" : {"model": "encode_image"},
            "damo/multi-modal_clip-vit-huge-patch14_zh" : {"clip_model": "encode_image"},
            "damo/multi-modal_clip-vit-large-patch14_zh" : {"clip_model": "encode_image"},
        }
        assert args.teacher_model_name in teacher_model_dict, "Error: Valid teacher model name has not been built."

        try:
            teacher_model = Model.from_pretrained(args.teacher_model_name)
        except Exception as e:
            if "Unexpected key(s) in state_dict" in str(e):
                error_message = (
                    "An error occurred while loading the model: {}\n"
                    "Maybe you should update modelscope. ".format(e)
                )
                raise RuntimeError(error_message)

        for k, v in teacher_model.state_dict().items():
            v.requires_grad = False
        
        # mapping different extract_features function to same name
        mapping = teacher_model_dict[args.teacher_model_name]
        if "model" in mapping and hasattr(teacher_model, "model"):
            model_instance = getattr(teacher_model, "model")
            if hasattr(model_instance, mapping["model"]):
                setattr(teacher_model, "get_feature", getattr(model_instance, mapping["model"]))
        elif "clip_model" in mapping and hasattr(teacher_model, "clip_model"):
            model_instance = getattr(teacher_model, "clip_model")
            if hasattr(model_instance, mapping["clip_model"]):
                setattr(teacher_model, "get_feature", getattr(model_instance, mapping["clip_model"]))

        # 单节点训练：直接将教师模型移动到GPU，不使用DDP
        teacher_model = teacher_model.to(args.device)
        # 使用相同的 ModelWrapper 包装以兼容 train.py 中的 teacher_model.module 引用
        teacher_model = ModelWrapper(teacher_model)
        logging.info(f"Teacher model loaded from {args.teacher_model_name}")
        current_line = inspect.currentframe().f_lineno
        print(f"[{current_file}:{current_line}] ✓ 教师模型加载完成: {args.teacher_model_name}")
    else:
        teacher_model = None

    # ========================================================================
    # 步骤14: 训练循环
    # ========================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤14: 开始训练循环 (从epoch {start_epoch} 到 {args.max_epochs})...")
    print(f"[{current_file}:{current_line}] ========== 所有初始化完成，开始训练 ==========")
    # 从start_epoch开始训练，直到max_epochs
    # start_epoch可能是0（从头训练）或从检查点恢复的epoch编号
    for epoch in range(start_epoch, args.max_epochs):
        current_line = inspect.currentframe().f_lineno
        print(f"[{current_file}:{current_line}] ========== 开始训练 Epoch {epoch + 1}/{args.max_epochs} ==========")
        
        # 记录当前epoch开始（只有主进程记录）
        if is_master(args) == 0:
            logging.info(f'Start epoch {epoch + 1}')
        
        # 执行训练
        current_line = inspect.currentframe().f_lineno
        print(f"[{current_file}:{current_line}] 开始执行训练 (epoch {epoch + 1})...")
        # train函数会：
        #   1. 遍历训练数据加载器
        #   2. 前向传播计算损失（对比学习损失）
        #   3. 反向传播计算梯度
        #   4. 更新模型参数
        #   5. 记录训练指标（损失、准确率等）
        #   6. 返回本epoch的步数
        # 如果启用知识蒸馏，会传入teacher_model用于计算蒸馏损失
        if args.distillation:
            # 知识蒸馏训练：使用教师模型指导学生模型训练
            num_steps_this_epoch = train(model, data, epoch, optimizer, scaler, scheduler, args, steps, teacher_model)
        else:
            # 标准训练：只使用对比学习损失
            num_steps_this_epoch = train(model, data, epoch, optimizer, scaler, scheduler, args, steps)
        
        # 累计总步数（用于学习率调度和日志记录）
        steps += num_steps_this_epoch
        current_line = inspect.currentframe().f_lineno
        print(f"[{current_file}:{current_line}] ✓ Epoch {epoch + 1} 训练完成，总步数: {steps}")

        # 执行验证（如果满足验证条件）
        current_line = inspect.currentframe().f_lineno
        print(f"[{current_file}:{current_line}] 检查是否需要执行验证...")
        # 验证条件：
        #   1. 指定了验证数据集（val_data不为None）
        #   2. 指定了验证间隔（valid_epoch_interval不为None）
        #   3. 当前epoch满足验证间隔（(epoch + 1) % valid_epoch_interval == 0）
        if args.val_data is not None and args.valid_epoch_interval is not None and ((epoch + 1) % args.valid_epoch_interval) == 0:
            assert "val" in data, "Error: Valid dataset has not been built."
            
            # evaluate函数会：
            #   1. 在验证集上计算图像和文本特征
            #   2. 计算检索指标（R@1, R@5, R@10等）
            #   3. 记录验证结果到日志
            current_line = inspect.currentframe().f_lineno
            print(f"[{current_file}:{current_line}] 开始执行验证 (epoch {epoch + 1})...")
            if not args.use_flash_attention:
                evaluate(model, data, epoch, args, steps)
            else:
                # FlashAttention需要FP16精度，使用autocast上下文管理器
                with torch.cuda.amp.autocast():
                    evaluate(model, data, epoch, args, steps)
            current_line = inspect.currentframe().f_lineno
            print(f"[{current_file}:{current_line}] ✓ 验证完成 (epoch {epoch + 1})")

        # 如果还有下一个epoch，为下一个epoch重新加载数据集和数据加载器
        # 这样可以支持每个epoch使用不同的数据增强策略
        # 例如：某些数据增强策略可能需要根据epoch动态调整
        if epoch + 1 < args.max_epochs:
            data = get_data(args, epoch_id=epoch + 1, max_txt_length=args.context_length)

        # ====================================================================
        # 步骤15: 保存检查点
        # ====================================================================
        # 只有主进程且本epoch有训练步数时才保存检查点
        current_line = inspect.currentframe().f_lineno
        if args.should_save and num_steps_this_epoch > 0:
            print(f"[{current_file}:{current_line}] 步骤15: 保存检查点 (epoch {epoch + 1})...")
        if args.should_save and num_steps_this_epoch > 0:
            # 保存定期检查点（根据save_epoch_frequency或最后一个epoch）
            # 保存条件：
            #   1. 是最后一个epoch（epoch + 1 == max_epochs）
            #   2. 或者满足保存频率（(epoch + 1) % save_epoch_frequency == 0）
            # 定期检查点文件名：epoch{epoch+1}.pt，例如：epoch1.pt, epoch2.pt, epoch3.pt
            if (epoch + 1) == args.max_epochs or (
                args.save_epoch_frequency > 0 and ((epoch + 1) % args.save_epoch_frequency) == 0
            ):
                t1 = time.time()
                save_path = os.path.join(args.checkpoint_path, f"epoch{epoch + 1}.pt")
                
                # 保存检查点内容：
                #   - epoch: 当前epoch编号，用于恢复训练时确定起始epoch
                #   - step: 当前总步数，用于恢复训练时确定起始步数
                #   - name: 实验名称，用于标识检查点所属的实验
                #   - state_dict: 模型权重，包含所有参数的当前值
                #   - optimizer: 优化器状态，包含动量、学习率调度等信息
                # 注意：如果使用FlashAttention，需要转换state_dict格式
                torch.save(
                    {
                        "epoch": epoch + 1,      # 当前epoch数
                        "step": steps,           # 当前总步数
                        "name": args.name,       # 实验名称
                        "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(model.state_dict()),  # 模型权重
                        "optimizer": optimizer.state_dict(),  # 优化器状态（包含Adam的动量等）
                    },
                    save_path,
                )
                logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, steps, time.time() - t1))
                current_line = inspect.currentframe().f_lineno
                print(f"[{current_file}:{current_line}] ✓ 定期检查点已保存: {save_path}")
            
            # 保存最新检查点（每个epoch都保存，用于自动恢复训练）
            # 文件名固定为：epoch_latest.pt
            # 如果训练中断，可以通过加载epoch_latest.pt自动恢复训练
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": steps,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, steps, time.time() - t1))
            current_line = inspect.currentframe().f_lineno
            print(f"[{current_file}:{current_line}] ✓ 最新检查点已保存: {save_path}")


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == "__main__":
    main()
