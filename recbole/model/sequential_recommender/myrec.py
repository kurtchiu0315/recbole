# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
MyRec
################################################

Reference: None
Reference: None
"""

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.cl4rec import CL4Rec


class MyRec(CL4Rec):
    r"""
    TODO
    """

    def __init__(self, config, dataset):
        super(MyRec, self).__init__(config, dataset)
