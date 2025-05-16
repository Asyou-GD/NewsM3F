import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import mmengine
import torch
from mmengine.dataset import Compose
from mmengine import print_log
from mmpretrain.datasets import BaseDataset
from mmpretrain.registry import DATASETS
from transformers import AutoTokenizer

import h5py





@DATASETS.register_module()
class NewsMultiLabelDataset(BaseDataset):
    """News multi-label dataset for textual and image classification.

    This dataset:
      1) Reads a list of JSON filenames from `ann_file`.
      2) Each JSON is located in `data_root`.
      3) Builds multi-label vectors for both level1 and level2 categories.
      4) Processes images using `load_pipeline`.
      5) Returns a dictionary to be consumed by the `pipeline`.
    """

    def __init__(self,
                 data_root: str,
                 ann_file: Union[str, list],
                 pipeline: list,
                 load_pipeline: list,
                 level1_categories: List[str],
                 level2_categories: List[str],
                 use_img: bool = True,
                 num_imgs: int = 3,
                 img_size: int = 384,
                 main_feat_cfg: Optional[dict] = None,
                 test_mode: bool = False,
                 max_refetch: int = 10,
                 **kwargs):
        """
        Args:
            data_root (str): Directory where all the JSON files live.
            ann_file (str or list): Path to txt file containing JSON filenames (one per line), 
                or a list of filenames directly.
            pipeline (list): A sequence of data transforms for textual data.
            load_pipeline (list): A sequence of data transforms for image data.
            level1_categories (List[str]): List of all level1 categories in English.
            level2_categories (List[str]): List of all level2 categories in English.
            num_imgs (int): Number of images to load per sample. Defaults to 1.
            test_mode (bool): Whether it's a test dataset. Default: False.
            max_refetch (int): Number of retries if data is invalid. Default: 10.
            **kwargs: Additional keyword arguments passed to BaseDataset.
        """
        self.data_root = data_root
        self.level1_categories = level1_categories
        self.level2_categories = level2_categories
        self.level1_to_idx = {cat: i for i, cat in enumerate(self.level1_categories)}
        self.level2_to_idx = {cat: i for i, cat in enumerate(self.level2_categories)}
        self.max_refetch = max_refetch
        self.num_imgs = num_imgs
        self.test_mode = test_mode
        self.main_feat_cfg = main_feat_cfg
        self.img_size = img_size
        tokenizer_name = self.main_feat_cfg.get('tokenizer', None)
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
        self.use_img = use_img
        # Initialize image processing pipeline
        self.load_pipeline = Compose(load_pipeline) if load_pipeline is not None else None
        # Initialize parent class (BaseDataset) — this will call self.load_data_list()
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            test_mode=test_mode,
            **kwargs
        )


    def load_data_list(self) -> List[Dict[str, Any]]:

        if isinstance(self.ann_file, str):
            lines = mmengine.list_from_file(self.ann_file)
            # Each line is something like "ap-000002.json"
            data_list = [{'filename': line.strip()} for line in lines if line.strip()]
        # If ann_file itself is a list (already provided in config)
        elif isinstance(self.ann_file, list):
            data_list = [{'filename': fn} for fn in self.ann_file]
        else:
            raise TypeError("ann_file must be either a string path or a list of filenames.")
        return data_list

    def prepare_data(self, idx: int) -> Dict[str, Any]:
        data_info = self.get_data_info(idx)

        filename = data_info['filename']
        json_path = os.path.join(self.data_root, filename)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        translated_text = data.get("text", "")

        # level1_labels_en = data.get("标注提交tag一级分类_英文", [])
        # level2_labels_en = data.get("标注提交tag二级分类_英文", [])

        level1_labels_en = data.get("label_level1", [])
        level2_labels_en = data.get("label_level2", [])

        label1_vec = torch.zeros(len(self.level1_categories), dtype=torch.float)
        for lbl in level1_labels_en:
            if lbl=='Culturet':
                lbl='Culture and Entertainment'
            if lbl in self.level1_to_idx:
                label1_vec[self.level1_to_idx[lbl]] = 1.0
            else:
                print('一级类目不在标签树内:',lbl)
        label2_vec = torch.zeros(len(self.level2_categories), dtype=torch.float)
        for lbl in level2_labels_en:
            if lbl in self.level2_to_idx:
                label2_vec[self.level2_to_idx[lbl]] = 1.0
            else:
                print('二级类目不在标签树内:',lbl)

        image_paths = data.get("image_path", [])  # This should be a list
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        imgs = []
        img_attn_masks = []
        if self.use_img == True:
            for img_path in image_paths[:self.num_imgs]:
                if img_path:
                    try:
                        # img_path = ''
                        tmp_result = self.load_pipeline(dict(url=img_path))
                        img = np.ascontiguousarray(tmp_result['img'].transpose(2, 0, 1))
                        imgs.append(torch.from_numpy(img))
                        img_attn_masks.append(1)
                        # img_attn_masks.append(0)
                        # print('img:',img_path, img.shape, img)
                    except Exception as e:
                        # logging.warning(f"Failed to load image {img_path}: {e}")
                        imgs.append(torch.zeros(3, self.img_size, self.img_size))  # Assuming default image size
                        img_attn_masks.append(0)
                else:
                    imgs.append(torch.zeros(3, self.img_size, self.img_size))  # Placeholder for missing image
                    img_attn_masks.append(0)
        else:
            imgs.append(torch.zeros(3, self.img_size, self.img_size))  # Placeholder for missing image
            img_attn_masks.append(0)
        # If fewer images than num_imgs, pad with zeros
        while len(imgs) < self.num_imgs:
            imgs.append(torch.zeros(3, self.img_size, self.img_size))  # Adjust channels and size as needed
            img_attn_masks.append(0)
        
        imgs = torch.stack(imgs, dim=0)  # Shape: (num_imgs, C, H, W)
        img_attn_masks = torch.tensor(img_attn_masks, dtype=torch.long)

        if self.tokenizer is not None:
            tokenized_data = self.tokenizer(
                translated_text,
                max_length=self.main_feat_cfg['max_text_length'],
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).data  # .data is an OrderedDict with 'input_ids', 'token_type_ids', etc.
        else:
            # If no tokenizer, just keep raw text
            tokenized_data = {'input_ids': None, 'attention_mask': None}

        # 6) Prepare output dictionary
        out_data = {
            'filename': filename,
            'input_main_string': tokenized_data,  # Tokenized text
            'labels_level1': label1_vec,
            'labels_level2': label2_vec,
            'img_attn_masks': img_attn_masks
        }
        # print(out_data)
        # 7) Pass through the pipeline
        results = self.pipeline(out_data)
        results['inputs'] = imgs

        return results

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        if not self._fully_initialized:
            print_log(
                'Please call full_init() method manually to accelerate the speed.',
                logger='current',
                level=logging.WARNING
            )
            self.full_init()

        for _ in range(self.max_refetch + 1):
            try:
                data = self.prepare_data(idx)
            except Exception as e:
                logging.warning(f"Error processing index {idx}: {e}")
                idx = self._rand_another()
                continue
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise RuntimeError(
            f'Cannot find valid data after {self.max_refetch} tries! '
            'Please check your data or pipeline.'
        )



@DATASETS.register_module()
class NewsMultiLabelDataset_jsonl(BaseDataset):
    """兼容 *.jsonl 与 *.json 文件目录 的多标签新闻数据集"""

    def __init__(self,
                 data_root: str,
                 ann_file: Union[str, list],
                 pipeline: list,
                 load_pipeline: list,
                 level1_categories: List[str],
                 level2_categories: List[str],
                 use_img: bool = True,
                 num_imgs: int = 3,
                 img_size: int = 384,
                 main_feat_cfg: Optional[dict] = None,
                 test_mode: bool = False,
                 max_refetch: int = 10,
                 **kwargs):

        # ------- 0) 基础属性 -------
        self.data_root = data_root
        self.level1_categories = level1_categories
        self.level2_categories = level2_categories
        self.level1_to_idx = {c: i for i, c in enumerate(level1_categories)}
        self.level2_to_idx = {c: i for i, c in enumerate(level2_categories)}
        self.max_refetch = max_refetch
        self.num_imgs = num_imgs
        self.test_mode = test_mode
        self.img_size = img_size
        self.use_img = use_img

        # ------- 1) tokenizer -------
        self.main_feat_cfg = main_feat_cfg or {}
        tok_name = self.main_feat_cfg.get('tokenizer')
        self.tokenizer = (AutoTokenizer.from_pretrained(tok_name)
                          if tok_name else None)

        # ------- 2) 图像 pipeline -------
        self.load_pipeline = Compose(load_pipeline) if load_pipeline else None

        # ------- 3) 判断 ann_file 是否为 jsonl -------
        self.is_jsonl = isinstance(ann_file, str) and ann_file.endswith('.jsonl')
        self.jsonl_cache: List[str] | None = None  # 保存每一行的字符串

        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            data_root=data_root,
            test_mode=test_mode,
            **kwargs
        )

    # ------------------------------------------------------------------
    # BaseDataset 接口
    # ------------------------------------------------------------------
    def load_data_list(self) -> List[Dict[str, Any]]:
        """构造 `self.data_list`，每个元素只存索引或文件名."""
        if self.is_jsonl:
            # 读取整个 jsonl 文件到内存，便于随机访问
            with open(self.ann_file, 'r', encoding='utf-8') as f:
                self.jsonl_cache = f.readlines()

            # data_list 只保存行号，后续 prepare_data 用行号取内容
            return [{'line_idx': idx} for idx in range(len(self.jsonl_cache))]

        # --------- 旧的多 json 文件目录格式 ----------
        if isinstance(self.ann_file, str):
            # txt 里每行都是 json 文件名
            names = mmengine.list_from_file(self.ann_file)
            return [{'filename': n.strip()} for n in names if n.strip()]

        if isinstance(self.ann_file, list):
            return [{'filename': fn} for fn in self.ann_file]

        raise TypeError('ann_file must be path str / list / jsonl str.')

    # ------------------------------------------------------------------
    def prepare_data(self, idx: int) -> Dict[str, Any]:
        """真正读数据 + 处理. 支持两种存储形式."""
        data_info = self.get_data_info(idx)

        # ----------- 1. 解析 JSON 内容 -----------
        if self.is_jsonl:
            raw_line: str = self.jsonl_cache[data_info['line_idx']]
            data = json.loads(raw_line)
        else:
            # 原来按文件名加载
            json_path = os.path.join(self.data_root, data_info['filename'])
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # ----------- 2. 字段抽取 -----------
        text = data.get('text', '')
        level1_labels_en = data.get('label_level1', [])
        level2_labels_en = data.get('label_level2', [])
        image_paths = data.get('image_path', [])
        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        # ----------- 3. 标签转 one‑hot -----------
        label1_vec = torch.zeros(len(self.level1_categories))
        for lbl in level1_labels_en:
            if lbl in self.level1_to_idx:
                label1_vec[self.level1_to_idx[lbl]] = 1
            else:
                print(f'label {lbl} not in level1')
        label2_vec = torch.zeros(len(self.level2_categories))
        for lbl in level2_labels_en:
            if lbl in self.level2_to_idx:
                label2_vec[self.level2_to_idx[lbl]] = 1
            else:
                print(f'label {lbl} not in level2')
        # ----------- 4. 图像处理 -----------
        imgs, img_attn_masks = [], []
        if self.use_img:
            for p in image_paths[:self.num_imgs]:
                full_path = os.path.join(self.data_root, p)
                try:
                    tmp = self.load_pipeline(dict(url=full_path))
                    img = np.ascontiguousarray(tmp['img'].transpose(2, 0, 1))
                    imgs.append(torch.from_numpy(img))
                    img_attn_masks.append(1)
                except Exception:
                    imgs.append(torch.zeros(3, self.img_size, self.img_size))
                    img_attn_masks.append(0)
        # 补齐
        while len(imgs) < self.num_imgs:
            imgs.append(torch.zeros(3, self.img_size, self.img_size))
            img_attn_masks.append(0)
        imgs = torch.stack(imgs)            # (N,3,H,W)
        img_attn_masks = torch.tensor(img_attn_masks, dtype=torch.long)

        # ----------- 5. 文本 token -----------
        if self.tokenizer:
            tok = self.tokenizer(
                text,
                max_length=self.main_feat_cfg.get('max_text_length', 512),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).data
        else:
            tok = {'input_ids': None, 'attention_mask': None}

        # ----------- 6. 封装返回 -----------
        out = {
            'filename': data.get('news_id', ''),     # 可改成唯一 id
            'input_main_string': tok,
            'labels_level1': label1_vec,
            'labels_level2': label2_vec,
            'img_attn_masks': img_attn_masks
        }
        out = self.pipeline(out)
        out['inputs'] = imgs
        return out



