# mmpretrain/models/heads/multi_label_linear_head.py

from typing import List, Dict, Optional  # 确保导入 List
import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .multi_label_cls_head import MultiLabelClsHead

@MODELS.register_module()
class MultiLabelLinearClsHead(MultiLabelClsHead):
    """Linear classification head for multi-label task, supporting custom label keys.

    Args:
        num_classes (int): Number of classes at this classification head.
        in_channels (int): Input feature dimension.
        loss (Dict): Config for the classification loss. Defaults to
            dict(type='BCEWithLogitsLoss', loss_weight=1.0).
        thr (float, optional): Threshold for predictions. Defaults to None.
        topk (int, optional): Top-k predictions to consider as positive. Defaults to None.
        label_key (str, optional): The attribute name in DataSample which stores
            the ground-truth labels for this head. Defaults to 'gt_label'.
        prefix (str, optional): Prefix for prediction fields. Defaults to ''.
        init_cfg (dict, optional): Initialization config. Defaults to
            dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss: Dict = dict(type='BCEWithLogitsLoss', loss_weight=1.0),
                 thr: Optional[float] = None,
                 topk: Optional[int] = None,
                 label_key: str = 'gt_label',
                 prefix: str = '',
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01)):
        # Pass label_key to parent class
        super(MultiLabelLinearClsHead, self).__init__(
            loss=loss, thr=thr, topk=topk, label_key=label_key, init_cfg=init_cfg)

        assert num_classes > 0, f'num_classes ({num_classes}) must be a positive integer.'
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.prefix = prefix

        # Define the final linear layer
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Process before the final classification head."""
        return feats[-1]

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Forward process."""
        pre_logits = self.pre_logits(feats)
        cls_score = self.fc(pre_logits)
        return cls_score

    def _get_predictions(self, cls_score: torch.Tensor,
                         data_samples: List['DataSample']):
        """Post-process the output of head."""
        pred_scores = torch.sigmoid(cls_score)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(cls_score.size(0))]

        for data_sample, score in zip(data_samples, pred_scores):
            if self.thr is not None:
                label = torch.where(score >= self.thr)[0]
            else:
                _, label = score.topk(self.topk)

            # 设置带前缀的字段名
            if self.prefix:
                pred_score_field = f'pred_score_{self.prefix}'
                pred_label_field = f'pred_label_{self.prefix}'
                num_classes_field = f'num_classes_{self.prefix}'
            else:
                pred_score_field = 'pred_score'
                pred_label_field = 'pred_label'
                num_classes_field = 'num_classes'

            data_sample.set_field(score, pred_score_field, dtype=torch.Tensor)
            data_sample.set_field(label, pred_label_field, dtype=torch.Tensor)
            data_sample.set_field(self.num_classes, num_classes_field, field_type='metainfo')

        return data_samples
