from collections import Counter
from typing import Sequence, List, Union, Optional, Any

import numpy as np
import torch
from mmengine.evaluator import DumpResults
from mmengine.evaluator.metric import _to_cpu, BaseMetric

from typing import Sequence, Dict, Any
from sklearn.metrics import f1_score
import logging
from mmpretrain.evaluation import SingleLabelMetric
from mmpretrain.registry import METRICS
import pandas as pd
from torchmetrics.functional.classification import multilabel_recall_at_fixed_precision, multiclass_recall_at_fixed_precision,binary_recall_at_fixed_precision, binary_precision, binary_recall

import csv
import os
from sklearn.metrics import precision_recall_curve

from torchmetrics.functional import f1_score




@METRICS.register_module()
class TwoLevelF1Metric(BaseMetric):

    def __init__(
        self,
        num_classes_level1: int,
        num_classes_level2: int,
        threshold_level1: float = 0.5,
        threshold_level2: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes_level1 = num_classes_level1
        self.num_classes_level2 = num_classes_level2
        self.threshold_level1 = threshold_level1
        self.threshold_level2 = threshold_level2

        # 缓存区
        self.pred_scores_level1: List[torch.Tensor] = []
        self.gt_labels_level1: List[torch.Tensor] = []
        self.pred_scores_level2: List[torch.Tensor] = []
        self.gt_labels_level2: List[torch.Tensor] = []


    def process(
        self,
        data_batch: Dict[str, Any],
        predictions: Sequence[Dict[str, Any]],
    ) -> None:

        for pred, sample in zip(predictions, data_batch["data_samples"]):
            # -------- level-1 --------
            score_l1 = pred.get("pred_score_level1")
            if score_l1 is None:
                logging.warning("Prediction missing 'pred_score_level1'; skipped.")
                continue
            score_l1 = score_l1.squeeze(0) if score_l1.dim() == 2 else score_l1
            self.pred_scores_level1.append(score_l1.detach().cpu())

            gt_l1 = sample.get("labels_level1")
            if gt_l1 is None:
                logging.warning("Data sample missing 'labels_level1'; skipped.")
                continue
            self.gt_labels_level1.append(gt_l1.float().detach().cpu())

            # -------- level-2 --------
            score_l2 = pred.get("pred_score_level2")
            if score_l2 is None:
                logging.warning("Prediction missing 'pred_score_level2'; skipped.")
                continue
            score_l2 = score_l2.squeeze(0) if score_l2.dim() == 2 else score_l2
            self.pred_scores_level2.append(score_l2.detach().cpu())

            gt_l2 = sample.get("labels_level2")
            if gt_l2 is None:
                logging.warning("Data sample missing 'labels_level2'; skipped.")
                continue
            self.gt_labels_level2.append(gt_l2.float().detach().cpu())

        # 避免 BaseMetric 的 “空 results” 警告
        self.results.append(1)

    def compute_metrics(self, results: Sequence[dict]) -> Dict[str, float]:
        """跨设备聚合后计算 F1 指标."""
        if len(self.pred_scores_level1) == 0:
            logging.warning("No predictions to evaluate.")
            return {k: 0.0 for k in (
                "f1_macro_l1", "f1_micro_l1", "f1_weighted_l1", "f1_samples_l1",
                "f1_macro_l2", "f1_micro_l2", "f1_weighted_l2", "f1_samples_l2",
            )}

        # 堆叠 [N, C]
        ps_l1 = torch.stack(self.pred_scores_level1)
        gt_l1 = torch.stack(self.gt_labels_level1).int()
        ps_l2 = torch.stack(self.pred_scores_level2)
        gt_l2 = torch.stack(self.gt_labels_level2).int()

        pred_l1 = (ps_l1 >= self.threshold_level1).int()
        pred_l2 = (ps_l2 >= self.threshold_level2).int()

        metrics: Dict[str, float] = {}

        # ------------ macro / micro / weighted ------------
        for tag, pred, gt, n_cls in (
            ("l1", pred_l1, gt_l1, self.num_classes_level1),
            ("l2", pred_l2, gt_l2, self.num_classes_level2),
        ):
            for avg in ("macro", "micro", "weighted"):
                metrics[f"f1_{avg}_{tag}"] = float(
                    f1_score(
                        pred,
                        gt,
                        task="multilabel",
                        num_labels=n_cls,
                        average=avg,
                    ).item()
                )


        def samples_f1(pred: torch.Tensor, gt: torch.Tensor) -> float:
            tp = (pred & gt).sum(dim=1).float()
            fp = (pred & (~gt)).sum(dim=1).float()
            fn = ((~pred) & gt).sum(dim=1).float()
            denom = 2 * tp + fp + fn
            f1 = torch.where(denom == 0, torch.zeros_like(denom), 2 * tp / denom)
            return f1.mean().item()

        metrics["f1_samples_l1"] = samples_f1(pred_l1.bool(), gt_l1.bool())
        metrics["f1_samples_l2"] = samples_f1(pred_l2.bool(), gt_l2.bool())

        return metrics

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """清空缓存，供下一个 epoch 使用。"""
        self.pred_scores_level1.clear()
        self.gt_labels_level1.clear()
        self.pred_scores_level2.clear()
        self.gt_labels_level2.clear()
        self.results.clear()