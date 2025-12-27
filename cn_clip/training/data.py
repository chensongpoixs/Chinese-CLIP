from math import ceil
import os
import logging
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass

import lmdb
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
import torch.distributed as dist

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from timm.data import create_transform

from cn_clip.clip import _tokenizer
from cn_clip.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        self.lmdb_path = lmdb_path

        # assert LMDB directories exist
        assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split);
        lmdb_pairs = os.path.join(lmdb_path, "pairs");
        assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(lmdb_pairs, split);
        lmdb_imgs = os.path.join(lmdb_path, "imgs");
        assert os.path.isdir(lmdb_imgs), "The LMDB directory {} of {} image base64 strings does not exist!".format(lmdb_imgs, split);

        # open LMDB files
        self.env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False);
        self.txn_pairs = self.env_pairs.begin(buffers=True);
        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False);
        self.txn_imgs = self.env_imgs.begin(buffers=True);

        # fetch number of pairs and images
        self.number_samples = int(self.txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'));
        self.number_images = int(self.txn_imgs.get(key=b'num_images').tobytes().decode('utf-8'));
        logging.info("{} LMDB file contains {} images and {} pairs.".format(split, self.number_images, self.number_samples));

        super(LMDBDataset, self).__init__()

        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples;
        self.global_batch_size = 1; # will be modified to the exact global_batch_size after calling pad_dataset()

        self.split = split;
        self.max_txt_length = max_txt_length;        

        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            transform = create_transform(
                             input_size=resolution,
                             scale=(0.9, 1.0),
                             is_training=True,
                             color_jitter=None,
                             auto_augment='original',
                             interpolation='bicubic',
                             mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711),
                         );
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:]);
        else:
            transform = Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]);
        return transform

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.number_samples

        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode('utf-8')).tobytes())
        image_id, text_id, raw_text = pair

        image_b64 = self.txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64))) # already resized
        image = self.transform(image)

        text = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        eos_index = text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        return image, text, eos_index


def pad_dataset(dataset, global_batch_size):
    # edit dataset.__len__() of the dataset
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: object  # 可以是 DistributedSampler 或 RandomSampler
    dataset: LMDBDataset
    epoch_id: int


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    dataset = LMDBDataset(
        db_path, 
        split="train" if is_train else "val",
        max_txt_length=max_txt_length,
        use_augment=args.use_augment if is_train else False,
        resolution=fetch_resolution(args.vision_model),
    ) 

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    batch_size = args.batch_size if is_train else args.valid_batch_size
    global_batch_size = batch_size * 1; # torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    # Update in 22.12.11: We have changed the **validation** dataset sampler during finetuning
    # from sequential to shuffled (in a determistic order between experiments and epochs). 
    # This is to avoid there being one text matching multiple images (or vice versa) in a local batch
    # which will affect the correctness of computing the validation in-batch accuracy.
    
    # 如果分布式已初始化，使用 DistributedSampler；否则使用 RandomSampler（单节点训练）
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
        sampler.set_epoch(epoch_id if is_train else 0)
    else:
        # 单节点训练：使用 RandomSampler
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch_id)  # 确保每个epoch的随机性不同
        sampler = RandomSampler(dataset, generator=generator)

    # ============================================================================
    # 创建 DataLoader 数据加载器
    # ============================================================================
    # DataLoader 的作用：
    #   1. 批量加载数据：将数据集中的样本按批次组织
    #   2. 数据采样：使用指定的采样器（sampler）决定数据的访问顺序
    #   3. 多进程加载：可选地使用多个工作进程并行加载数据（加速数据预处理）
    #   4. 数据预取：提前加载下一批数据，减少GPU等待时间
    #
    # 参数说明：
    #   dataset (LMDBDataset): 
    #       - 要加载的数据集对象，包含LMDB数据库连接和数据转换逻辑
    #       - 注意：LMDB的Environment对象无法被pickle，因此num_workers必须为0
    #
    #   batch_size (int):
    #       - 每个批次包含的样本数量
    #       - 训练时使用 args.batch_size，验证时使用 args.valid_batch_size
    #       - 较大的batch_size可以提高GPU利用率，但需要更多显存
    #
    #   pin_memory (bool):
    #       - 是否将数据固定到内存（固定内存，page-locked memory）
    #       - True: 启用固定内存，数据从CPU传输到GPU更快（推荐在GPU训练时使用）
    #       - False: 不固定内存，节省内存但传输稍慢
    #       - 这里设置为False，因为LMDB数据可能较大，固定内存会占用更多RAM
    #
    #   num_workers (int):
    #       - 用于数据加载的子进程数量
    #       - 0: 使用主进程加载数据（单进程模式）
    #       - >0: 使用多个子进程并行加载数据，可以加速数据预处理
    #       - 注意：由于LMDBDataset在__init__中打开了Environment对象，
    #         这些对象无法被pickle序列化，因此num_workers必须设置为0
    #       - 训练时使用 args.num_workers，验证时使用 args.valid_num_workers
    #
    #   sampler (DistributedSampler | RandomSampler):
    #       - 数据采样器，决定数据访问的顺序
    #       - DistributedSampler: 分布式训练时使用，确保每个进程看到不同的数据子集
    #       - RandomSampler: 单节点训练时使用，随机打乱数据顺序
    #       - 如果使用sampler，则shuffle参数会被忽略（默认为False）
    #
    # 其他常用参数（此处未使用）：
    #   - shuffle (bool): 是否在每个epoch开始时打乱数据（使用sampler时无效）
    #   - drop_last (bool): 是否丢弃最后一个不完整的批次（默认False）
    #   - collate_fn (callable): 自定义的批次合并函数（默认使用torch的默认函数）
    #   - prefetch_factor (int): 每个worker预取的批次数（默认2）
    #   - persistent_workers (bool): 是否保持worker进程存活（默认False）
    # ============================================================================
    dataloader = DataLoader(
        dataset,                                    # 数据集对象（LMDBDataset）
        batch_size=batch_size,                      # 批次大小（训练或验证）
        pin_memory=False,                           # 不固定内存（节省RAM，适合LMDB大数据集）
        num_workers=args.num_workers if is_train else args.valid_num_workers,  # 工作进程数（必须为0，因为LMDB Environment无法pickle）
        sampler=sampler,                            # 数据采样器（DistributedSampler或RandomSampler）
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(args, epoch_id=0, max_txt_length=64):
    data = {}

    if args.train_data:
        data["train"] = get_dataset(
            args, 
            is_train=True,  
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id)

    if args.val_data:
        data["val"] = get_dataset(
            args, 
            is_train=False, 
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id)

    return data
