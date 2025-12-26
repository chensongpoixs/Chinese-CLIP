import torch
import torch.distributed as dist
import os

def setup_distributed():
    """设置分布式训练环境"""
    # 尝试不同的端口
    port = int(os.environ.get('MASTER_PORT', 29500))
    while True:
        try:
            dist.init_process_group(
                backend='nccl'  ,
                init_method=f'tcp://localhost:{port}',
                world_size=int(os.environ['WORLD_SIZE']),
                rank=int(os.environ['RANK'])
            )
            break
        except RuntimeError as e:
            if "Address already in use" in str(e):
                port += 1
                os.environ['MASTER_PORT'] = str(port)
            else:
                raise

def main():
    # 检查是否是主进程
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}/{dist.get_world_size()} 初始化成功")
    
    # 你的训练代码...
# python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 --use_env 
if __name__ == "__main__":
    # 单机多卡启动命令示例
    # python -m torch.distributed.launch --nproc_per_node=2 your_script.py
    main()