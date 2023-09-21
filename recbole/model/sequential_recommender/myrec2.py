# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
MyRec2
################################################

Reference: None
Reference: None
"""

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.sasrec import SASRec


class MyRec2(SASRec):
    r"""
    TODO
    """

    def __init__(self, config, dataset):
        super(MyRec2, self).__init__(config, dataset)
