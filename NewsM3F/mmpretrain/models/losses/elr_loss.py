import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import math
from mmpretrain.registry import MODELS


@MODELS.register_module()
class Elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=28, beta=0.3):
        super(Elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda()   ### soft label 估计矩阵
        self.beta = beta

    def forward(self, cls_score, label,  **kwargs):
        # import pdb; pdb.set_trace()


        y_pred = F.sigmoid(cls_score).float()
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        sample_idx = kwargs['sample_idx']
        self.target = self.target.to(y_pred.device)
        label = label.to(torch.float)
        self.target[sample_idx] = self.beta * self.target[sample_idx] + self.beta * label +(1-2*self.beta) * y_pred_ ### soft label 估计矩阵

        # numerical_labels = torch.argmax(label, dim=1)

        # bce_loss = self.num_classes * F.binary_cross_entropy(y_pred, label, weight=torch.gather(self.target[sample_idx],1, numerical_labels.unsqueeze(1)))
        bce_loss =  self.num_classes * F.binary_cross_entropy(y_pred, label)
        # bce_loss = F.binary_cross_entropy_with_logits(cls_score, label,  pos_weight=torch.gather(self.target[sample_idx],1, numerical_labels.unsqueeze(1)))
        # elr_reg = 
        elr_reg = self.num_classes * (-(self.target[sample_idx] * y_pred.log())).mean() 
        # import pdb; pdb.set_trace()
        # print('bce_loss:', bce_loss, 'elr_reg:', elr_reg)
        final_loss = bce_loss + 3*elr_reg
        # final_loss = bce_loss
        return  final_loss

