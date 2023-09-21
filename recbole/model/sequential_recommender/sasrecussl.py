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
from torch import nn

from recbole.model.sequential_recommender.sasrec import SASRec


class SASRecUSSL(SASRec):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(SASRecUSSL, self).__init__(config, dataset)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.ussl_weight = config['ussl_weight']

        # load dataset info
        self.n_users = dataset.user_num

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size, padding_idx=0)
        self.ussl_norm = nn.Linear(self.hidden_size, self.hidden_size)
        self.default_mask = self.mask_correlated_samples(self.batch_size)
        self.ssl_loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)
    
    def mask_correlated_samples(self, batch_size):
        N = batch_size
        mask = torch.ones((2 * N, 2 * N)).bool()
        mask = mask.fill_diagonal_(0)
        mask *= ~ torch.diagflat(torch.ones(N), offset=N).bool()
        mask *= ~ torch.diagflat(torch.ones(N), offset=-N).bool()
        return mask

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_emb = self.user_embedding(user).squeeze(1)
        seq_output = self.forward(item_seq, item_seq_len)
        ussl_loss = self.cal_ssl_loss(user_emb, seq_output)
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = seq_output + user_emb
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

        return loss, self.ussl_weight * ussl_loss
    
    def cal_ssl_loss(self, seq1, seq2, tau=1):
        cur_batch_size = seq1.shape[0]
        seq2 = self.ussl_norm(seq2)
        logits, labels = self.info_nce(seq1, seq2, temp=tau, batch_size=cur_batch_size)
        ssl_loss = self.ssl_loss_fct(logits, labels)
        return ssl_loss

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
        # [2B H]
        z = torch.cat((z_i, z_j), dim=0)
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        # [2B, 1]
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.default_mask

        # [2B, 2(B-1)]
        negative_samples = sim[mask].reshape(N, -1)
        # [2B, 2B-1]
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        # the first column stores positive pair scores
        labels = torch.zeros(N, dtype=torch.long, device=z_i.device)
        return logits, labels

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        user_emb = self.user_embedding(user).squeeze(1)
        seq_output += user_emb
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        user_emb = self.user_embedding(user).squeeze(1)
        seq_output += user_emb
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
