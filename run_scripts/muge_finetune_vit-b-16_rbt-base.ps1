# ============================================================================
# Chinese-CLIP MUGE Finetune Training Script for Windows PowerShell
# ============================================================================
# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# Usage: .\run_scripts\muge_finetune_vit-b-16_rbt-base.ps1 [DATAPATH]
# If DATAPATH is not specified, it will use "datapath" as default
# ============================================================================

param(
    [string]$DATAPATH = "datapath"
)

# Set environment variable for Windows PyTorch distributed training
$env:USE_LIBUV = "0"

# ============================================================================
# Training Configuration
# ============================================================================

# Number of GPUs per GPU worker
$GPUS_PER_NODE = 1
# Number of GPU workers, for single-worker training, please set to 1
$WORKER_CNT = 1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
$MASTER_ADDR = "localhost"
# The port for communication
$MASTER_PORT = 8514
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
$RANK = 0

# Set PYTHONPATH
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD\cn_clip"

# ============================================================================
# Data Options
# ============================================================================
$train_data = "$DATAPATH\datasets\MUGE\lmdb\train"
$val_data = "$DATAPATH\datasets\MUGE\lmdb\valid"

# ============================================================================
# Restore Options
# ============================================================================
$resume = "$DATAPATH\pretrained_weights\clip_cn_vit-b-16.pt"
$reset_data_offset = "--reset-data-offset"
$reset_optimizer = "--reset-optimizer"
# To disable reset_optimizer, set to empty string:
# $reset_optimizer = ""

# ============================================================================
# Output Options
# ============================================================================
$output_base_dir = "$DATAPATH\experiments\"
$name = "muge_finetune_vit-b-16_roberta-base_bs128_8gpu"
$save_step_frequency = 999999
$save_epoch_frequency = 1
$log_interval = 1
$report_training_batch_acc = "--report-training-batch-acc"
# To disable report_training_batch_acc, set to empty string:
# $report_training_batch_acc = ""

# ============================================================================
# Training Hyper-parameters
# ============================================================================
$context_length = 52
$warmup = 100
$batch_size = 128
$valid_batch_size = 128
$accum_freq = 1
$lr = "5e-5"
$wd = 0.001
$max_epochs = 3
$valid_step_interval = 150
$valid_epoch_interval = 1
$vision_model = "ViT-B-16"
$text_model = "RoBERTa-wwm-ext-base-chinese"
$use_augment = "--use-augment"
# To disable use_augment, set to empty string:
# $use_augment = ""

# ============================================================================
# Additional Options (not in original script but useful for Windows)
# ============================================================================
$num_workers = 4
$valid_num_workers = 4
# Uncomment below to enable gradient checkpointing (saves memory):
# $grad_checkpointing = "--grad-checkpointing"
$grad_checkpointing = ""

# ============================================================================
# Find conda executable
# ============================================================================
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

# ============================================================================
# Run Training Command
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Starting Chinese-CLIP Training" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Data Path: $DATAPATH"
Write-Host "Model: $vision_model + $text_model"
Write-Host "Batch Size: $batch_size"
Write-Host "Learning Rate: $lr"
Write-Host "Max Epochs: $max_epochs"
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

if ($condaExe) {
    Write-Host "Using conda: $condaExe" -ForegroundColor Green
    Write-Host "Activating environment: stable-diffusion-webui" -ForegroundColor Green
    Write-Host ""
    
    $trainingArgs = @(
        "run", "-n", "stable-diffusion-webui",
        "python", "-m", "torch.distributed.launch",
        "--nproc_per_node=$GPUS_PER_NODE",
        "--nnodes=$WORKER_CNT",
        "--node_rank=$RANK",
        "--master_addr=$MASTER_ADDR",
        "--master_port=$MASTER_PORT",
        "--",
        "cn_clip/training/main.py",
        "--train-data=$train_data",
        "--val-data=$val_data",
        "--num-workers=$num_workers",
        "--valid-num-workers=$valid_num_workers",
        "--resume=$resume",
        $reset_data_offset,
        $reset_optimizer,
        "--logs=$output_base_dir",
        "--name=$name",
        "--save-step-frequency=$save_step_frequency",
        "--save-epoch-frequency=$save_epoch_frequency",
        "--log-interval=$log_interval",
        $report_training_batch_acc,
        "--context-length=$context_length",
        "--warmup=$warmup",
        "--batch-size=$batch_size",
        "--valid-batch-size=$valid_batch_size",
        "--valid-step-interval=$valid_step_interval",
        "--valid-epoch-interval=$valid_epoch_interval",
        "--accum-freq=$accum_freq",
        "--lr=$lr",
        "--wd=$wd",
        "--max-epochs=$max_epochs",
        "--vision-model=$vision_model",
        $use_augment,
        $grad_checkpointing,
        "--text-model=$text_model"
    )
    
    # Remove empty strings from arguments
    $trainingArgs = $trainingArgs | Where-Object { $_ -ne "" }
    
    & $condaExe $trainingArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "============================================================================" -ForegroundColor Red
        Write-Host "Training failed with error code $LASTEXITCODE" -ForegroundColor Red
        Write-Host "============================================================================" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
    Write-Host ""
    Write-Host "============================================================================" -ForegroundColor Green
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "============================================================================" -ForegroundColor Green
} else {
    Write-Host "Conda not found. Please activate your conda environment manually and run:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "python -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE --nnodes=$WORKER_CNT --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT -- cn_clip/training/main.py --train-data=$train_data --val-data=$val_data --num-workers=$num_workers --valid-num-workers=$valid_num_workers --resume=$resume $reset_data_offset $reset_optimizer --logs=$output_base_dir --name=$name --save-step-frequency=$save_step_frequency --save-epoch-frequency=$save_epoch_frequency --log-interval=$log_interval $report_training_batch_acc --context-length=$context_length --warmup=$warmup --batch-size=$batch_size --valid-batch-size=$valid_batch_size --valid-step-interval=$valid_step_interval --valid-epoch-interval=$valid_epoch_interval --accum-freq=$accum_freq --lr=$lr --wd=$wd --max-epochs=$max_epochs --vision-model=$vision_model $use_augment $grad_checkpointing --text-model=$text_model" -ForegroundColor Yellow
}

