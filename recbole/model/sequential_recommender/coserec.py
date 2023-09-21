# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.cl4rec import CL4Rec


class CoSeRec(CL4Rec):
    def __init__(self, config, dataset):
        super(CoSeRec, self).__init__(config, dataset)
