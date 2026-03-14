# -*- coding: utf-8 -*-
# %%
import os

import torch
import torch.nn as tn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import random
import inspect

import joblib

from torch import Tensor
from itertools import chain
from pathlib import Path

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemprop import data, featurizers, models, nn, utils
from chemprop.data import MoleculeDatapoint, make_split_indices, split_data_by_indices
from chemprop.nn.predictors import BinaryClassificationFFN
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn.metrics import BinaryAUROC, BCELoss
from chemprop.featurizers import MoleculeFeaturizerRegistry

# %%
from chemprop.nn.predictors import BinaryClassificationFFN
from chemprop.nn.metrics import BinaryAUROC
from chemprop.nn.metrics import BCELoss
import torch.nn as tn
from torch import Tensor
import torch.nn.functional as F

# %%
class CloneableBCELoss(tn.Module):
    def __init__(self, gamma=1.5, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds, targets, mask=None, weights=None, lt_mask=None, gt_mask=None):
        preds = preds.view(-1)
        targets = targets.view(-1)

        if mask is not None:
            mask = mask.view(-1).bool()
            preds = preds[mask]
            targets = targets[mask]

        # Focal loss 계산
        BCE_loss = F.binary_cross_entropy_with_logits(preds, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean()

    def clone(self):
        return CloneableBCELoss(gamma=self.gamma, alpha=self.alpha)



# %%
class CustomBinaryClassificationFFN(BinaryClassificationFFN):
    def __init__(self, input_dim: int, hidden_size: int = 1900, num_layers: int = 2, dropout: float = 0.1, pos_weight: float = 1.0):
        super().__init__()

        layers = []

        # 입력층
        layers.append(tn.Linear(input_dim, hidden_size))
        layers.append(tn.ReLU())
        layers.append(tn.Dropout(dropout))

        # 히든층
        for _ in range(num_layers - 1):
            layers.append(tn.Linear(hidden_size, hidden_size))
            layers.append(tn.ReLU())
            layers.append(tn.Dropout(dropout))

        # 출력층 (binary classification → 1 unit)
        layers.append(tn.Linear(hidden_size, 1))

        self.ffn = tn.Sequential(*layers)

        # Sigmoid 제거: raw logits 출력
        self.output_transform = tn.Identity()

        # pos_weight 반영된 loss 함수
        self.criterion = CloneableBCELoss(pos_weight)

        # AUC metric (raw logits로 계산 가능)
        self.metric = BinaryAUROC()

    def forward(self, Z: Tensor) -> Tensor:
        return self.output_transform(self.ffn(Z))

    def train_step(self, Z: Tensor) -> Tensor:
        return self.forward(Z)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

