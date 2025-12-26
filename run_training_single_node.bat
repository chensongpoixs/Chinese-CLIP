@echo off
chcp 65001 >nul
REM 单节点训练脚本
REM 激活conda环境并运行训练

call conda activate Chinese-Clip
if errorlevel 1 (
    echo 错误: 无法激活conda环境 Chinese-Clip
    pause
    exit /b 1
)

echo 已激活conda环境: Chinese-Clip
echo 开始运行训练...
echo.

python cn_clip/training/main.py ^
    --train-data=datapath/datasets/MUGE/lmdb/train ^
    --val-data=datapath/datasets/MUGE/lmdb/valid ^
    --num-workers=0 ^
    --valid-num-workers=0 ^
    --resume=datapath/pretrained_weights/clip_cn_vit-b-16.pt ^
    --reset-data-offset ^
    --reset-optimizer ^
    --logsfan=datapath/experiments/ ^
    --name=muge_finetune_vit-b-16_roberta-base_bs48_1gpu ^
    --save-step-frequency=999999 ^
    --save-epoch-frequency=1 ^
    --report-training-batch-acc ^
    --context-length=52 ^
    --warmup=100 ^
    --batch-size=48 ^
    --valid-batch-size=48 ^
    --valid-step-interval=1000 ^
    --valid-epoch-interval=1 ^
    --lr=3e-06 ^
    --wd=0.001 ^
    --max-epochs=10 ^
    --vision-model=ViT-B-16 ^
    --use-augment ^
    --grad-checkpointing ^
    --text-model=RoBERTa-wwm-ext-base-chinese ^
    --skip-aggregate

if errorlevel 1 (
    echo.
    echo 训练过程中出现错误！
    pause
    exit /b 1
)

echo.
echo 训练完成！
pause

