# mmpretrain/models/heads/multi_label_cls_head.py

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample, label_to_onehot

@MODELS.register_module()
class MultiLabelClsHead(BaseModule):
    """Classification head for multi-label task with customizable label_key.

    Args:
        loss (dict): Config of classification loss. Defaults to
            dict(type='CrossEntropyLoss', use_sigmoid=True).
        thr (float, optional): Threshold for predictions. Defaults to None.
        topk (int, optional): Top-k predictions to consider as positive. Defaults to None.
        label_key (str, optional): The attribute name in DataSample which stores
            the ground-truth labels for this head. Defaults to 'gt_label'.
        init_cfg (dict, optional): Initialization config. Defaults to None.

    Notes:
        If both ``thr`` and ``topk`` are set, ``thr`` is used to determine
        positive predictions. If neither is set, ``thr=0.5`` is used by default.
    """

    def __init__(self,
                 loss: Dict = dict(type='CrossEntropyLoss', use_sigmoid=True),
                 thr: Optional[float] = None,
                 topk: Optional[int] = None,
                 label_key: str = 'gt_label',
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)

        if not isinstance(loss, torch.nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss

        # Set default threshold if both thr and topk are None
        if thr is None and topk is None:
            thr = 0.5

        self.thr = thr
        self.topk = topk
        self.label_key = label_key

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """Process before the final classification head."""
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """Forward process."""
        pre_logits = self.pre_logits(feats)
        return pre_logits

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): Features from the backbone.
            data_samples (List[DataSample]): Annotation data.
            **kwargs: Additional arguments for loss module.

        Returns:
            dict[str, Tensor]: Dictionary of loss components.
        """
        cls_score = self(feats)
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample], **kwargs) -> dict:
        """Unpack data samples and compute loss with device-checking logic."""
        num_classes = cls_score.size(-1)

        # 1) 如果 data_sample 里有 'gt_score'，直接拿
        if 'gt_score' in data_samples[0]:
            target = torch.stack([ds.gt_score.float() for ds in data_samples])
        else:
            labels = [getattr(ds, self.label_key, []) for ds in data_samples]

            # 2) 判断是否已是 multi-hot
            if (
                isinstance(labels[0], torch.Tensor)
                and labels[0].dtype in [torch.float, torch.int, torch.long]
            ):
                if labels[0].dim() == 1 and labels[0].size(0) == num_classes:
                    # 已经是 multi-hot
                    target = torch.stack(labels).float()
                else:
                    # 是索引列表
                    target = torch.stack([
                        label_to_onehot(lbl, num_classes) for lbl in labels
                    ]).float()
            else:
                # 不是 tensor，转 tensor
                target = torch.stack([
                    label_to_onehot(lbl, num_classes) for lbl in labels
                ]).float()

        # 3) 自动修正 device 不匹配
        if target.device != cls_score.device:
            old_dev = target.device
            new_dev = cls_score.device
            # print(f"[WARNING] device mismatch: target is on {old_dev}, "
            #       f"but cls_score is on {new_dev}. Moving target to {new_dev}.")
            target = target.to(new_dev)

        # 4) 计算 loss
        losses = {}
        loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        return losses

    def predict(self,
                feats: Tuple[torch.Tensor],
                data_samples: List[DataSample] = None) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): Features from the backbone.
            data_samples (List[DataSample], optional): Annotation data.

        Returns:
            List[DataSample]: Data samples with predictions.
        """
        cls_score = self(feats)
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions

    def _get_predictions(self, cls_score: torch.Tensor,
                         data_samples: List[DataSample]):
        """Post-process the output of head."""
        pred_scores = torch.sigmoid(cls_score)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(cls_score.size(0))]

        for data_sample, score in zip(data_samples, pred_scores):
            if self.thr is not None:
                # A label is predicted positive if its score >= thr
                label = torch.where(score >= self.thr)[0]
            else:
                # Use top-k predictions as positive
                _, label = score.topk(self.topk)
            # try:
            # data_sample.set_pred_score(score).set_pred_label(label)
        data_sample.set_pred_score(score).set_pred_label(label)

            # except:
            #     import pdb;pdb.set_trace()
        return data_samples
