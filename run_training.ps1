# 在conda的stable-diffusion-webui环境中运行训练命令
# 使用方法：在PowerShell中运行 .\run_training.ps1

# 查找conda可执行文件
$condaExe = $null
$condaPaths = @(
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\Anaconda3\Scripts\conda.exe",
    "C:\ProgramData\Miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\AppData\Local\Continuum\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\AppData\Local\Continuum\miniconda3\Scripts\conda.exe"
)

foreach ($path in $condaPaths) {
    if (Test-Path $path) {
        $condaExe = $path
        break
    }
}

# 如果找到了conda，使用conda run运行命令
if ($condaExe) {
    Write-Host "找到conda: $condaExe"
    Write-Host "正在stable-diffusion-webui环境中运行训练命令..."
    # 设置环境变量以禁用libuv（Windows上需要）
    $env:USE_LIBUV = "0"
    & $condaExe run -n stable-diffusion-webui python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=8514 -- cn_clip/training/main.py --train-data=datapath/datasets/MUGE/lmdb/train --val-data=datapath/datasets/MUGE/lmdb/valid --num-workers=4 --valid-num-workers=4 --resume=datapath/pretrained_weights/clip_cn_vit-b-16.pt --reset-data-offset --reset-optimizer --logs=datapath/experiments/ --name=muge_finetune_vit-b-16_roberta-base_bs48_1gpu --save-step-frequency=999999 --save-epoch-frequency=1 --log-interval=10 --report-training-batch-acc --context-length=52 --warmup=100 --batch-size=48 --valid-batch-size=48 --valid-step-interval=1000 --valid-epoch-interval=1 --lr=3e-06 --wd=0.001 --max-epochs=1 --vision-model=ViT-B-16 --use-augment --grad-checkpointing --text-model=RoBERTa-wwm-ext-base-chinese
} else {
    Write-Host "未找到conda，请使用以下方法之一："
    Write-Host "1. 打开Anaconda Prompt，激活stable-diffusion-webui环境，然后运行："
    Write-Host "   set USE_LIBUV=0"
    Write-Host "   python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=8514 -- cn_clip/training/main.py --train-data=datapath/datasets/MUGE/lmdb/train --val-data=datapath/datasets/MUGE/lmdb/valid --num-workers=4 --valid-num-workers=4 --resume=datapath/pretrained_weights/clip_cn_vit-b-16.pt --reset-data-offset --reset-optimizer --logs=datapath/experiments/ --name=muge_finetune_vit-b-16_roberta-base_bs48_1gpu --save-step-frequency=999999 --save-epoch-frequency=1 --log-interval=10 --report-training-batch-acc --context-length=52 --warmup=100 --batch-size=48 --valid-batch-size=48 --valid-step-interval=1000 --valid-epoch-interval=1 --lr=3e-06 --wd=0.001 --max-epochs=1 --vision-model=ViT-B-16 --use-augment --grad-checkpointing --text-model=RoBERTa-wwm-ext-base-chinese"
    Write-Host "2. 或者运行批处理文件: .\run_training.bat"
}

