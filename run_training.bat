@echo off
set USE_LIBUV=0
call conda activate stable-diffusion-webui
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=8514 -- cn_clip/training/main.py --train-data=datapath/datasets/MUGE/lmdb/train --val-data=datapath/datasets/MUGE/lmdb/valid --num-workers=4 --valid-num-workers=4 --resume=datapath/pretrained_weights/clip_cn_vit-b-16.pt --reset-data-offset --reset-optimizer --logs=datapath/experiments/ --name=muge_finetune_vit-b-16_roberta-base_bs48_1gpu --save-step-frequency=999999 --save-epoch-frequency=1 --log-interval=10 --report-training-batch-acc --context-length=52 --warmup=100 --batch-size=48 --valid-batch-size=48 --valid-step-interval=1000 --valid-epoch-interval=1 --lr=3e-06 --wd=0.001 --max-epochs=1 --vision-model=ViT-B-16 --use-augment --grad-checkpointing --text-model=RoBERTa-wwm-ext-base-chinese
pause


