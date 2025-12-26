'''
test tran 
@date: 2025-
'''



import os
print(os.getcwd())


import torch;

device =   "cuda" if torch.cuda.is_available() else "cpu"; # 检查是否有可用的GPU,否则使用CPU



print(device);


GPUS_PER_NODE=1 # 卡数
WORKER_CNT=1 # 机器数
MASTER_ADDR="127.0.0.1"
MASTER_PORT=18534 # 同台机器同时起多个任务，请分别分配不同的端口号
RANK=0

# 刚刚创建过的目录，存放了预训练参数和预处理好的数据集
DATAPATH="datapath"

# 指定LMDB格式的训练集和验证集路径（存放了LMDB格式的图片和图文对数据）
train_data=f"{DATAPATH}/datasets/MUGE/lmdb/train"
val_data=f"{DATAPATH}/datasets/MUGE/lmdb/valid"
num_workers=4 # 训练集pytorch dataloader的进程数，设置为>0，以减小训练时读取数据的时间开销
valid_num_workers=4 # 验证集pytorch dataloader的进程数，设置为>0，以减小验证时读取数据的时间开销

# 指定刚刚下载好的Chinese-CLIP预训练权重的路径
resume=f"{DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt"
reset_data_offset="--reset-data-offset" # 从头读取训练数据
reset_optimizer="--reset-optimizer" # 重新初始化AdamW优化器

# 指定输出相关配置
output_base_dir=f"{DATAPATH}/experiments/"
name="muge_finetune_vit-b-16_roberta-base_bs48_1gpu" # finetune超参、日志、ckpt将保存在../datapath/experiments/muge_finetune_vit-b-16_roberta-base_bs48_1gpu/
save_step_frequency=999999 # disable it
save_epoch_frequency=1 # 每轮保存一个finetune ckpt
log_interval=10 # 日志打印间隔步数
report_training_batch_acc="--report-training-batch-acc" # 训练中，报告训练batch的in-batch准确率

# 指定训练超参数
context_length=52 # 序列长度，这里指定为Chinese-CLIP默认的52
warmup=100 # warmup步数
batch_size=48 # 训练单卡batch size
valid_batch_size=48 # 验证单卡batch size
lr=3e-6 # 学习率，因为这里我们使用的对比学习batch size很小，所以对应的学习率也调低一些
wd=0.001 # weight decay
max_epochs=1 # 训练轮数，也可通过--max-steps指定训练步数
valid_step_interval=1000 # 验证步数间隔
valid_epoch_interval=1 # 验证轮数间隔
vision_model="ViT-B-16" # 指定视觉侧结构为ViT-B/16
text_model="RoBERTa-wwm-ext-base-chinese" # 指定文本侧结构为RoBERTa-base
use_augment="--use-augment" # 对图像使用数据增强
grad_checkpointing="--grad-checkpointing" # 激活重计算策略，用更多训练时间换取更小的显存开销






#run_command = "export PYTHONPATH=${PYTHONPATH}:d:/Work/AI/stable_diffusion/stable-diffusion-webui/Chinese-CLIP/cn_clip;" + \
cmd = f"""
python -m torch.distributed.launch --nproc_per_node={GPUS_PER_NODE} --nnodes={WORKER_CNT}   --node_rank={RANK}  \
      --master_addr={MASTER_ADDR} --master_port={MASTER_PORT}     cn_clip/training/main.py \
      --train-data={train_data} \
      --val-data={val_data} \
      --num-workers={num_workers} \
      --valid-num-workers={valid_num_workers} \
      --resume={resume} \
      {reset_data_offset} \
      {reset_optimizer} \
      --logsfan={output_base_dir} \
      --name={name} \
      --save-step-frequency={save_step_frequency} \
      --save-epoch-frequency={save_epoch_frequency} \
      {report_training_batch_acc} \
      --context-length={context_length} \
      --warmup={warmup} \
      --batch-size={batch_size} \
      --valid-batch-size={valid_batch_size} \
      --valid-step-interval={valid_step_interval} \
      --valid-epoch-interval={valid_epoch_interval} \
      --lr={lr} \
      --wd={wd} \
      --max-epochs={max_epochs} \
      --vision-model={vision_model} \
      {use_augment} \
      {grad_checkpointing} \
      --text-model={text_model}
""".lstrip()
print(cmd)


cmd;



print('Path ===   ', os.getcwd());


# print('=====', test_cmd);

import os

if os.environ.get('LOCAL_RANK') == 'YOUR_RANK': # Replace with your actual rank
    print("Running on rank:", os.environ.get('LOCAL_RANK'))
else:
    print("Running on other rank")



print(os.environ.get('LOCAL_RANK'))


#os.environ['LOCAL_RANK'] = 9;

# import os
# import argparse
# def main(args):
#     local_rank = args.local_rank
#     print(local_rank, os.environ['LOCAL_RANK'])


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", type=int)
#     args = parser.parse_args()
#     main(args)


'''
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=2       --master_addr=localhost --master_port=8514 cn_clip/training/main.py       
--train-data=datapath/datasets/MUGE/lmdb/train       --val-data=datapath/datasets/MUGE/lmdb/valid       --num-workers=4       --valid-num-workers=4       
--resume=datapath/pretrained_weights/clip_cn_vit-b-16.pt       --reset-data-offset       --reset-optimizer       --logs=datapath/experiments/       
--name=muge_finetune_vit-b-16_roberta-base_bs48_1gpu       --save-step-frequency=999999       --save-epoch-frequency=1       --log-interval=10       
--report-training-batch-acc       --context-length=52       --warmup=100       --batch-size=48       --valid-batch-size=48       
--valid-step-interval=1000       --valid-epoch-interval=1       --lr=3e-06       --wd=0.001       --max-epochs=1       --vision-model=ViT-B-16       
--use-augment       --grad-checkpointing       --text-model=RoBERTa-wwm-ext-base-chinese

'''





