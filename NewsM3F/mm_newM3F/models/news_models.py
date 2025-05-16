import copy
import math
from functools import partial
from typing import Optional, List, Tuple
import einops
import torch
from mmcv.cnn.bricks.transformer import build_transformer_layer
from mmengine import digit_version
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from peft import LoraConfig, get_peft_model, get_peft_config
from torch import nn
from transformers import ChineseCLIPModel, ChineseCLIPConfig
from transformers.models.chinese_clip.modeling_chinese_clip import ChineseCLIPVisionTransformer, ChineseCLIPTextModel
from mmpretrain.models import ImageClassifier, ClsDataPreprocessor, LinearClsHead
from mmpretrain.registry import MODELS, HOOKS
from mmpretrain.structures import DataSample
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2Model, SiglipVisionModel
from transformers import Swinv2Config, Swinv2Model
 
from transformers import ConvNextConfig, ConvNextModel
from transformers import AutoModel


@MODELS.register_module()
class NewsIMGQWEN2TEXTFeatFusionClassifier(ImageClassifier):
    def __init__(
            self,
            backbone,
            text_backbone,
            head1,
            head2,
            pgvr_cfg: Optional[dict] = None,
            consistency_loss_cfg: Optional[dict] = None,
            is_pooling_feats=False,
            discrete_int_extractor=None,
            continuous_float_extractor=None,
            freeze_backbone=False,
            *args,
            **kwargs,
    ):
        vision_project = backbone.pop('vision_project', None)

        super().__init__(backbone, *args, **kwargs)
        self.is_pooling_feats = is_pooling_feats

        self.text_backbone = MODELS.build(text_backbone)
        if vision_project is not None:
            self.vision_projector = MODELS.build(vision_project)

        if discrete_int_extractor is not None:
            self.discrete_int_extractor = nn.Embedding(**discrete_int_extractor)
        if continuous_float_extractor is not None:
            self.continuous_float_extractor = MODELS.build(continuous_float_extractor)

        if freeze_backbone:
            self.backbone.eval()
            self.text_backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.text_backbone.parameters():
                param.requires_grad = False

        if is_main_process():
            self.print_trainable_parameters()
        self.head1 = MODELS.build(head1)
        self.head2 = MODELS.build(head2)

        if pgvr_cfg is not None:
            self.pgvr = MODELS.build(pgvr_cfg)
        else:
            self.pgvr = None

        if consistency_loss_cfg is not None:
            self.consistency_loss = MODELS.build(consistency_loss_cfg)
        else:
            self.consistency_loss = None

    def switch_to_deploy(self):
        self.backbone.merge_lora_weights()
        self.text_backbone.merge_lora_weights()

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        trainable_names = []
        for name, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
                trainable_names.append(name)
        trainable_names = set([x.split('.')[0] for x in trainable_names])
        print(f"Trainable parameter names: {trainable_names}")
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def extract_feat(
            self,
            inputs: torch.Tensor,
            data_samples: Optional[List[DataSample]] = None
    ):
        feat_dict = dict()
        batch_size = inputs.shape[0]
        extra_feat_keys = [k for k in data_samples[0].keys() if 'input_' in k]

        # for img feat
        img_attn_masks = [x.get('img_attn_masks') for x in data_samples]
        img_attn_masks = torch.stack(img_attn_masks).to(inputs.device)
        imgs = einops.rearrange(inputs, 'b n c h w -> (b n) c h w')
        # imgs = imgs.to(dtype=torch.bfloat16)

        img_feat = self.backbone(imgs, output_hidden_states=True)[2][26]

        # if self.is_pooling_feats:
        #     img_feat = img_feat.pooler_output.unsqueeze(1)  # 16, 1, 1024
        # else:
        #     img_feat = img_feat.last_hidden_state  # 16, 257, 1024
        # import pdb; pdb.set_trace()
        img_feat = einops.rearrange(img_feat, '(b n) c d -> b n c d', b=batch_size)

        if hasattr(self, 'vision_projector'):
            img_feat = self.vision_projector(img_feat)
            # import pdb; pdb.set_trace()
        feat_dict['img'] = img_feat
        feat_dict['img_attn_masks'] = img_attn_masks
        self._img_rep_each = img_feat.mean(dim=2)  
        
    # for text feat
        for key in [x for x in extra_feat_keys if '_string' in x]:
            input_ids = torch.cat([x.get(key)['input_ids'] for x in data_samples]).to(inputs.device)
            attention_mask = torch.cat([x.get(key)['attention_mask'] for x in data_samples]).to(inputs.device)
            # token_type_ids = torch.cat([x.get(key)['token_type_ids'] for x in data_samples]).to(inputs.device)
            token_type_ids = None 

            text_input_dict = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            
            text_feat = self.text_backbone(**text_input_dict)
            if self.is_pooling_feats:
                text_feat = text_feat[0].unsqueeze(1)
                attention_mask_ = torch.ones((batch_size, 1)).to(inputs.device)
            else:
                text_feat = text_feat[1]
                attention_mask_ = attention_mask
            feat_dict[key+'_attn_masks'] = attention_mask_
            feat_dict[key] = text_feat

        feats = self.neck(feat_dict)
        return feats


    def loss(self, inputs, data_samples):
        feats = self.extract_feat(inputs, data_samples)
        losses1 = self.head1.loss(feats, data_samples)
        losses2 = self.head2.loss(feats, data_samples)

        if self.pgvr is not None:
            img_repr = self._img_rep_each          # [B, N, D]
            p, w = self.pgvr(img_repr)
            y_parent = torch.stack([d.labels_level1 for d in data_samples]).to(p.device)
            loss_pgvr = self.pgvr.ranking_loss(p, w, y_parent)
        else:
            loss_pgvr = torch.tensor(0., device=inputs.device)

        if self.consistency_loss is not None:
            # 假设 head1 预测一级 logits，head2 预测二级 logits
            logits_parent = self.head1(feats)          # [B, C1]
            logits_child  = self.head2(feats)          # [B, C2]

            # 把 softmax / sigmoid 之后的概率拿出来
            
            p_parent = torch.sigmoid(logits_parent)
            p_child  = torch.sigmoid(logits_child)

            # 从 data_samples 里取二级标签 (0/1)
            # 下面的 key 名根据你实际 DataSample 里字段来改

            y_child = torch.stack([d.labels_level2 for d in data_samples]).to(p_child.device)

            loss_consis = self.consistency_loss(p_parent, p_child, y_child)
        else:
            loss_consis = logits_parent.new_tensor(0.0)

        losses = {}
        for key in losses1:
            losses[f'head1_{key}'] = losses1[key]
        for key in losses2:
            losses[f'head2_{key}'] = losses2[key]
        losses['loss_pgvr'] = loss_pgvr
        losses['loss_consistency'] = loss_consis

        return losses

    def predict(self, inputs, data_samples, **kwargs):
        feats = self.extract_feat(inputs, data_samples)
        cls = self.head1.predict(feats, data_samples, **kwargs)

        cls = self.head2.predict(feats, data_samples, **kwargs)
        return cls


@MODELS.register_module()
class PrototypeGuidedVisualRouting(BaseModule):
    """Prototype‑Guided Visual Routing
    
    Args:
        num_classes (int):   类别数
        feat_dim (int):      输入特征维度
        use_cosine (bool):   True=余弦相关性，False=点积
        loss_weight (float): ranking loss 的权重
    """
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        use_cosine: bool = True,
        loss_weight: float = 1.0,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_cosine = use_cosine
        self.loss_weight = loss_weight

        # class prototypes，shape = [C, D]
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))

        # routing head：为每幅图得到 w_i^c
        self.routing_head = nn.Linear(feat_dim, num_classes)

        # 初始化
        nn.init.xavier_uniform_(self.prototypes)
        nn.init.xavier_uniform_(self.routing_head.weight)
        nn.init.zeros_(self.routing_head.bias)

    # ------------- forward 改成同时支持 2D / 3D 特征 -----------------
    def forward(self, feat: torch.Tensor):
        """
        feat : [B, D]  或 [B, N, D]
        返回：
            p, w : 同形状 [..., C]
        """
        original_shape = feat.shape[:-1]          # 可能是 (B,) 或 (B,N)
        feat_flat = feat.reshape(-1, self.feat_dim)   # [(B*N), D]

        if self.use_cosine:
            p_flat = F.normalize(feat_flat, dim=-1) @ F.normalize(self.prototypes, dim=-1).t()
        else:
            p_flat = feat_flat @ self.prototypes.t()

        w_flat = self.routing_head(feat_flat)          # [(B*N), C]

        # 把平铺结果 reshape 回去
        p = p_flat.view(*original_shape, self.num_classes)
        w = w_flat.view(*original_shape, self.num_classes)
        return p, w
    # ----------------------------------------------------------------

    # ----------------- ranking_loss：沿 “图片” 维度 -------------------
    def ranking_loss(self, p, w, y_parent):
        """
        p, w : [B, N, C]   -- N 是一条新闻里的图片数
        y_parent : [B, C]  -- 一级标签 one‑hot
        """
        assert p.dim() == 3, "PGVR 现在只处理 [B,N,C] 形式"

        B, N, C = p.shape
        # i / j 维度是图片
        p_i = p.unsqueeze(2)            # [B, N, 1, C]
        p_j = p.unsqueeze(1)            # [B, 1, N, C]
        w_i = w.unsqueeze(2)
        w_j = w.unsqueeze(1)

        indicator = (p_i > p_j).float()                 # 1(p_i^c > p_j^c)
        margin    = F.relu(w_j - w_i)                   # max(0, w_j^c - w_i^c)

        # 去掉 i==j
        eye = torch.eye(N, device=p.device).view(1, N, N, 1)
        indicator = indicator * (1 - eye)
        margin    = margin    * (1 - eye)

        # 只在真标签的类上做监督
        anchor_mask = y_parent.view(B, 1, 1, C)         # [B,1,1,C]
        loss_mat = indicator * margin * anchor_mask     # [B,N,N,C]

        total_pairs = (indicator * anchor_mask).sum().clamp(min=1.0)
        loss = loss_mat.sum() / total_pairs
        return loss * self.loss_weight



@MODELS.register_module()
class HierConsistencyLoss(nn.Module):
    """层次一致性损失

    Args:
        child_to_parent (List[int]):  len = #child_classes，
            第 *j* 个元素给出子类 *j* 的父类在一级 logits 里的索引。
        reduction (str):  'mean' / 'sum'，默认 'mean'
        loss_weight (float):  损失系数
    """
    def __init__(
        self,
        child_to_parent: List[int],
        reduction: str = 'mean',
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.register_buffer(
            'parent_idx', torch.tensor(child_to_parent, dtype=torch.long)
        )
        assert reduction in ('mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        p_parent: torch.Tensor,    # [B, C_parent]
        p_child:  torch.Tensor,    # [B, C_child]
        y_child:  torch.Tensor,    # [B, C_child]  (0 / 1)
    ):
        """
        Returns
        -------
        loss : Tensor, scalar
        """
        # 取出与每个子类对应的父类置信度
        p_parent_aligned = p_parent[:, self.parent_idx]      # shape = [B, C_child]

        diff = F.relu(p_child - p_parent_aligned)            # 仅当 child > parent 时 >0
        loss = diff * y_child.float()                        # 只看真标签子类

        if self.reduction == 'mean':
            loss = loss.sum() / (y_child.sum().clamp(min=1.0))
        else:  # 'sum'
            loss = loss.sum()

        return loss * self.loss_weight
