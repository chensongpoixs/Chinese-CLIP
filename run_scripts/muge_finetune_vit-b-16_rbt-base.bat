@echo off
REM ============================================================================
REM Chinese-CLIP MUGE Finetune Training Script for Windows
REM ============================================================================
REM Guide:
REM This script supports distributed training on multi-gpu workers (as well as single-worker training). 
REM Please set the options below according to the comments. 
REM For multi-gpu workers training, these options should be manually set for each worker. 
REM After setting the options, please run the script on each worker.
REM Usage: run_scripts\muge_finetune_vit-b-16_rbt-base.bat [DATAPATH]
REM If DATAPATH is not specified, it will use "datapath" as default
REM ============================================================================

REM Set environment variable for Windows PyTorch distributed training
set USE_LIBUV=0

REM Activate conda environment (modify the environment name if needed)
call conda activate stable-diffusion-webui

REM ============================================================================
REM Training Configuration
REM ============================================================================

REM Number of GPUs per GPU worker
set GPUS_PER_NODE=1
REM Number of GPU workers, for single-worker training, please set to 1
set WORKER_CNT=1
REM The ip address of the rank-0 worker, for single-worker training, please set to localhost
set MASTER_ADDR=127.0.0.1
REM The port for communication
set MASTER_PORT=8514
REM The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
set RANK=0

REM Set PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%CD%\cn_clip

REM Data path - use command line argument or default to "datapath"
if "%1"=="" (
    set DATAPATH=datapath
) else (
    set DATAPATH=%1
)

REM ============================================================================
REM Data Options
REM ============================================================================
set train_data=%DATAPATH%\datasets\MUGE\lmdb\train
set val_data=%DATAPATH%\datasets\MUGE\lmdb\valid

REM ============================================================================
REM Restore Options
REM ============================================================================
set resume=%DATAPATH%\pretrained_weights\clip_cn_vit-b-16.pt
set reset_data_offset=--reset-data-offset
set reset_optimizer=--reset-optimizer
REM To disable reset_optimizer, comment the line above and uncomment below:
REM set reset_optimizer=

REM ============================================================================
REM Output Options
REM ============================================================================
set output_base_dir=%DATAPATH%\experiments\
set name=muge_finetune_vit-b-16_roberta-base_bs128_8gpu
set save_step_frequency=999999
set save_epoch_frequency=1
set log_interval=1
set report_training_batch_acc=--report-training-batch-acc
REM To disable report_training_batch_acc, comment the line above and uncomment below:
REM set report_training_batch_acc=

REM ============================================================================
REM Training Hyper-parameters
REM ============================================================================
set context_length=52
set warmup=100
set batch_size=128
set valid_batch_size=128
set accum_freq=1
set lr=5e-5
set wd=0.001
set max_epochs=3
set valid_step_interval=150
set valid_epoch_interval=1
set vision_model=ViT-B-16
set text_model=RoBERTa-wwm-ext-base-chinese
set use_augment=--use-augment
REM To disable use_augment, comment the line above and uncomment below:
REM set use_augment=

REM ============================================================================
REM Additional Options (not in original script but useful for Windows)
REM ============================================================================
set num_workers=4
set valid_num_workers=4
REM Uncomment below to enable gradient checkpointing (saves memory):
REM set grad_checkpointing=--grad-checkpointing
set grad_checkpointing=

REM ============================================================================
REM Run Training Command
REM ============================================================================
echo ============================================================================
echo Starting Chinese-CLIP Training
echo ============================================================================
echo Data Path: %DATAPATH%
echo Model: %vision_model% + %text_model%
echo Batch Size: %batch_size%
echo Learning Rate: %lr%
echo Max Epochs: %max_epochs%
echo ============================================================================
echo.

python -m torch.distributed.launch --nproc_per_node=%GPUS_PER_NODE% --nnodes=%WORKER_CNT% --node_rank=%RANK% --master_addr=%MASTER_ADDR% --master_port=%MASTER_PORT% -- cn_clip/training/main.py --train-data=%train_data% --val-data=%val_data% --num-workers=%num_workers% --valid-num-workers=%valid_num_workers% --resume=%resume% %reset_data_offset% %reset_optimizer% --logsfan=%output_base_dir% --name=%name% --save-step-frequency=%save_step_frequency% --save-epoch-frequency=%save_epoch_frequency% --log-interval=%log_interval% %report_training_batch_acc% --context-length=%context_length% --warmup=%warmup% --batch-size=%batch_size% --valid-batch-size=%valid_batch_size% --valid-step-interval=%valid_step_interval% --valid-epoch-interval=%valid_epoch_interval% --accum-freq=%accum_freq% --lr=%lr% --wd=%wd% --max-epochs=%max_epochs% --vision-model=%vision_model% %use_augment% %grad_checkpointing% --text-model=%text_model%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================================
    echo Training failed with error code %ERRORLEVEL%
    echo ============================================================================
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================================
echo Training completed successfully!
echo ============================================================================
pause

