import argparse
import inspect
import os


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B-32", "ViT-B-16", "ViT-H-14"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    elif model_name in ["ViT-L-14", "ViT-L-14-336"]:
        return {"lr": 4.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    """
    解析命令行参数
    
    功能：
    1. 创建参数解析器
    2. 定义所有训练相关的命令行参数
    3. 解析命令行参数
    4. 应用模型默认参数（如果某些参数未指定）
    5. 返回解析后的参数对象
    """
    # 获取当前文件名（用于调试和日志追踪）
    current_file = os.path.basename(__file__)
    
    # 获取当前行号（用于调试和日志追踪）
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ========== 开始执行 parse_args() 函数 ==========")
    
    # ============================================================================
    # 步骤1: 创建参数解析器
    # ============================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤1: 创建 ArgumentParser...")
    parser = argparse.ArgumentParser()
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ ArgumentParser 创建完成")
    
    # ============================================================================
    # 步骤2: 定义所有命令行参数
    # ============================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤2: 定义命令行参数...")
    
    # ============================================================================
    # 分布式训练参数（由torch.distributed.launch自动传递）
    # ============================================================================
    # local-rank: 当前进程在当前节点中的本地GPU排名（0, 1, 2, ...）
    # 这个参数由 torch.distributed.launch 自动传递，必须定义否则会报错
    # 即使不使用，也要定义这个参数以避免 "unrecognized arguments" 错误
    #parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training (automatically set by torch.distributed.launch)")
    # parser.add_argument("--local-rank", "--local_rank", type=int)
    # ============================================================================
    # 数据相关参数
    # ============================================================================
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to the LMDB directory with training data split",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to the LMDB directory with validation data split, default to None which disables validation",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="The number of workers for training dataloader."
    )
    parser.add_argument(
        "--valid-num-workers", type=int, default=1, help="The number of workers for validation dataloader (if making validation)."
    )
    parser.add_argument(
        "--logsfan",
        type=str,
        default="./logs/",
        help="Where to store logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train_clip",
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="How often to log loss info."
    )
    parser.add_argument(
        "--report-training-batch-acc", default=False, action="store_true", help="Whether to report training batch accuracy."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training per GPU."
    )
    parser.add_argument(
        "--valid-batch-size", type=int, default=64, help="Batch size for validation per GPU."
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Number of steps to train for (in higher priority to --max_epochs)."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=32, help="Number of full epochs to train for (only works if --max_steps is None)."
    )
    parser.add_argument(
        "--valid-step-interval", type=int, default=None, help="The step interval for validation (default to None which disables validation between steps)."
    )
    parser.add_argument(
        "--valid-epoch-interval", type=int, default=1, help="The epoch interval for validation (default to 1, set None to disable validation between epochs)."
    )
    parser.add_argument(
        "--context-length", type=int, default=52, help="The maximum length of input text (include [CLS] & [SEP] tokens). Default to 52."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=500, help="Number of steps to warmup for."
    )
    parser.add_argument("--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync."
    )
    parser.add_argument("--use-augment",
        default=False,
        action="store_true",
        help="Whether to use image augment."
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-epoch-frequency", type=int, default=1, help="How often to save checkpoints by epochs."
    )
    parser.add_argument(
        "--save-step-frequency", type=int, default=-1, help="How often to save checkpoints by steps."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        default=False,
        help="If resumed from a checkpoint, whether to reset the optimizer states.",
    )
    parser.add_argument(
        "--reset-data-offset",
        action="store_true",
        default=False,
        help="If resumed from a checkpoint, whether to reset the dataset offset to the beginning.",
    )    
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--mask-ratio",
        default=0,
        type=float,
        help="Random mask ratio of patches during finetuning. Default to zero which does not mask any patches.",
    )
    parser.add_argument(
        "--clip-weight-path",
        default=None,
        type=str,
        help="The path of openai pretrained weight, used to initialize the image encoder, should be set to None if you do not use pretrained CLIP",
    )    
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        default=False,
        help="Freeze the weight of vision encoder.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )    
    parser.add_argument(
        "--bert-weight-path",
        default=None,
        type=str,
        help="The path of bert pretrained weight, used to initialize the text encoder, should be set to None if you do not use pretrained BERT",
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--use-flash-attention",
        default=False,
        action="store_true",
        help="Enable flash attention."
    )
    parser.add_argument(
        "--accum-freq",
        type=int,
        default=1,
        help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    # arguments for distributed training
    parser.add_argument(
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=123, 
        help="Random seed."
    )
    # arguments for distillation
    parser.add_argument(
        "--distillation",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--teacher-model-name",
        type=str,
        default=None,
        help="The name of teacher model."
    )
    parser.add_argument(
        "--kd_loss_weight",
        type=float,
        default=0.5,
        help="Weight of KD loss."
    )
    # ============================================================================
    # 步骤3: 解析命令行参数
    # ============================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤3: 解析命令行参数...")
    args = parser.parse_args()
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ 命令行参数解析完成")
    
    # ============================================================================
    # 步骤4: 处理聚合参数
    # ============================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤4: 处理聚合参数 (aggregate = not skip_aggregate)...")
    args.aggregate = not args.skip_aggregate
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ 聚合参数设置完成: aggregate={args.aggregate}")

    # ============================================================================
    # 步骤5: 应用模型默认参数
    # ============================================================================
    # If some params are not passed, we use the default values based on model name.
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤5: 应用模型默认参数 (vision_model={args.vision_model})...")
    default_params = get_default_params(args.vision_model)
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 获取到的默认参数: {default_params}")
    
    for name, val in default_params.items():
        if getattr(args, name) is None:
            current_line = inspect.currentframe().f_lineno
            print(f"[{current_file}:{current_line}] 应用默认参数: {name}={val}")
            setattr(args, name, val)
    
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ 默认参数应用完成")
    
    # ============================================================================
    # 步骤6: 打印所有解析后的参数
    # ============================================================================
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] 步骤6: 打印所有解析后的参数...")
    print(f"[{current_file}:{current_line}] " + "=" * 80)
    print(f"[{current_file}:{current_line}] 所有训练参数列表:")
    print(f"[{current_file}:{current_line}] " + "-" * 80)
    
    # 按参数名排序打印，便于查看
    for name in sorted(vars(args).keys()):
        val = getattr(args, name)
        # 对于较长的字符串（如路径），截断显示
        if isinstance(val, str) and len(val) > 100:
            val_display = val[:100] + "..."
        else:
            val_display = val
        print(f"[{current_file}:{current_line}]   {name:30s} = {val_display}")
    
    print(f"[{current_file}:{current_line}] " + "-" * 80)
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ✓ 参数打印完成")
    print(f"[{current_file}:{current_line}] " + "=" * 80)
    
    current_line = inspect.currentframe().f_lineno
    print(f"[{current_file}:{current_line}] ========== parse_args() 函数执行完成 ==========")

    return args
