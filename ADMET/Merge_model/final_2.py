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

from chemprop.nn.predictors import RegressionFFN
from chemprop.nn.metrics import EvidentialLoss
import torch.nn as tn
import torch.nn.functional as F
import torch
from torch import Tensor
from chemprop import nn as chemnn   # for nn.Identity

class CustomRegressionFFN(tn.Module):
    n_targets = 1
    _T_default_metric = nn.metrics.MSE

    def __init__(self, input_dim: int, hidden_size: int = 800, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        layers += [tn.Linear(input_dim, hidden_size), tn.ReLU(), tn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [tn.Linear(hidden_size, hidden_size), tn.ReLU(), tn.Dropout(dropout)]
        self.backbone = tn.Sequential(*layers)
        self.head = tn.Linear(hidden_size, 1)

        self.output_transform = tn.Identity()
        self.criterion = nn.metrics.MSE()
        self.metric = nn.metrics.RMSE()

        self.hparams = {
            "predictor": "CustomRegressionFFN",
            "input_dim": input_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    def forward(self, Z):
        out = self.head(self.backbone(Z))   # (batch, 1)
        return out                          

    def train_step(self, Z):
        return self.forward(Z)


class CustomEvidentialFFN(tn.Module):
    n_targets = 4
    _T_default_criterion = EvidentialLoss

    def __init__(self, input_dim: int, hidden_size: int = 300, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        layers += [tn.Linear(input_dim, hidden_size), tn.ReLU(), tn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [tn.Linear(hidden_size, hidden_size), tn.ReLU(), tn.Dropout(dropout)]
        self.ffn = tn.Sequential(*layers, tn.Linear(hidden_size, self.n_targets))

        self.output_transform = tn.Identity()
        self.criterion = EvidentialLoss()
        self.metric = nn.metrics.RMSE()

        self.hparams = {
            "predictor": "CustomEvidentialFFN",
            "input_dim": input_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    def forward(self, Z: Tensor) -> Tensor:
        Y = self.ffn(Z)
        mean, v, alpha, beta = torch.chunk(Y, self.n_targets, dim=1)

        v = F.softplus(v)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta)

        mean = self.output_transform(mean)
        if not isinstance(self.output_transform, tn.Identity):
            beta = self.output_transform.transform_variance(beta)

        return torch.stack((mean, v, alpha, beta), dim=2)

    train_step = forward
