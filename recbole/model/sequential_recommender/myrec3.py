# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
MyRec3
################################################

Reference: None

Reference: None
"""

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.duorec import DuoRec


class MyRec3(DuoRec):
    r"""
    TODO
    """

    def __init__(self, config, dataset):
        super(MyRec3, self).__init__(config, dataset)

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # for mask
        
        # parameters initialization
        self.apply(self._init_weights)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
        losses = [loss]
        if self.cl_type in ['us', 'un', 'us_x']:
#             aug_item_seq2, aug_item_seq_len2 = interaction['aug2'], interaction['aug_len2']
#             un_aug_seq_output = self.forward(aug_item_seq2, aug_item_seq_len2)
            un_aug_seq_output = self.forward(item_seq, item_seq_len)
        
        if self.cl_type in ['us', 'su', 'us_x']:
            aug_item_seq1, aug_item_seq_len1 = interaction['aug1'], interaction['aug_len1']
            su_aug_seq_output = self.forward(aug_item_seq1, aug_item_seq_len1)

        if self.cl_type in ['us', 'un']:
            logits, labels = self.info_nce(seq_output, un_aug_seq_output)
            cl_loss = self.cl_lambda * self.cl_loss_fct(logits, labels)
            losses.append(cl_loss)

        if self.cl_type in ['us', 'su']:
            logits, labels = self.info_nce(seq_output, su_aug_seq_output)
            cl_loss = self.cl_lambda * self.cl_loss_fct(logits, labels)
            losses.append(cl_loss)

        if self.cl_type == 'us_x':
            logits, labels = self.info_nce(un_aug_seq_output, su_aug_seq_output)
            cl_loss = self.cl_lambda * self.cl_loss_fct(logits, labels)
            losses.append(cl_loss)

        return tuple(losses)
