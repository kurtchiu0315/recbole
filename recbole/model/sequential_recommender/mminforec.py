# -*- coding: utf-8 -*-
# @Time    : 2020/9/19 21:49
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

r"""
MMInfoRec = S3Rec + FDSA
################################################
S3Rec
################################################

Reference:
    Kun Zhou and Hui Wang et al. "S^3-Rec: Self-Supervised Learning
    for Sequential Recommendation with Mutual Information Maximization"
    In CIKM 2020.

Reference code:
    https://github.com/RUCAIBox/CIKM2020-S3Rec

"""

import random

import torch
from torch import nn
import numpy as np

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer, VanillaAttention
from recbole.model.loss import BPRLoss


class MMInfoRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MMInfoRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        
        self.num_hidden_layers_gru = config['num_hidden_layers_gru']
        self.enc = config['enc'] # [att, meancc, attcc]
        self.loss_fuse_dropout_prob = config['loss_fuse_dropout_prob']
        self.pred_step = config['pred_step']
        self.mil = config['mil']
        self.mb_dropout_prob = config['mb_dropout_prob']
        self.mem = config['mem']
        self.tau = config['tau']

        # 從 FDSA 借過來的～
        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])
        
        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']
        self.loss_list = config['loss_list']
        self.cl_lambda = config['cl_lambda']
        
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        
        if self.enc == 'att':
            self.ar_att_hidden_sz = self.hidden_size
        else:
            self.ar_att_hidden_sz = 2 * self.hidden_size
            
        self.position_embedding = nn.Embedding(self.max_seq_length, self.ar_att_hidden_sz)
    
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size, 
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        
        # self.feature_att_layer -> self.g_enc: fuse item embedding and attribute embedding
        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset, self.hidden_size, self.selected_features, self.pooling_mode, self.device
        )
        self.feature_att_layer = VanillaAttention(self.hidden_size, self.hidden_size)
        
        # self.feature_trm_encoder -> self.ar_att: sequence encoder
        self.feature_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.ar_att_hidden_sz,
            inner_size=self.ar_att_hidden_sz,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        
        # self.ar: autoregressive module for multi-step prediction
        self.ar = nn.GRU(self.ar_att_hidden_sz, self.ar_att_hidden_sz, self.num_hidden_layers_gru, batch_first=False)
        
        self.LayerNorm = nn.LayerNorm(self.ar_att_hidden_sz, eps = self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.fuse_dropout = nn.Dropout(self.loss_fuse_dropout_prob)
        self.all_fuse_emb = None

        # memory
        self.mb = nn.Embedding(self.mem, self.ar_att_hidden_sz)
        self.mb_fc = nn.Linear(self.ar_att_hidden_sz, self.mem)
        self.softmax = nn.Softmax(dim=-1)
        self.mb_dp = nn.Dropout(self.mb_dropout_prob)
        
        self.loss_fct_dict = {'BPR': BPRLoss(), 
                              'CE': nn.CrossEntropyLoss(), 
                              'MILNCE': self.MILNCE}

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == 'MILNCE':
            pass
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE', 'MILNCE']!")

        self.apply(self._init_weights)
        self.other_parameter_name = ['feature_embed_layer']
        
    def init_hidden(self, batch_size, device):
        return torch.zeros((self.num_hidden_layers_gru, 
                            batch_size, 
                            self.ar_att_hidden_sz), 
                            requires_grad=True).to(device)
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def MILNCE(self, sim_score):
        nominator = torch.log((torch.exp(self.tau * sim_score)  + 1e-24 ).sum(dim=1))
        denominator = torch.logsumexp(self.tau * sim_score, dim=1)
        loss = -(nominator - denominator)
        return loss.mean()
    
    def mem_read(self, seq_output):
        
        # print(f'seq_output.shape: {seq_output.shape}') # [B L H]: [512, 5, 128]
        seq_output_mb = self.mb_fc(seq_output) # [B L mem]
        # print(f'seq_output_mb.shape: {seq_output_mb.shape}') # [512, 5, 64]
        seq_output_mb = self.softmax(seq_output_mb) # [B L mem]
        # print(f'seq_output_mb.shape (after softmax): {seq_output_mb.shape}') # [512, 5, 64]
        # print(f'self.mb.weight.shape: {self.mb.weight.shape}') # [mem H]: [64, 128]
        seq_output_mem = seq_output_mb.matmul(self.mb.weight) # [B L H]
        # print(f'seq_output_mem.shape: {seq_output_mem.shape}') # [B L H]: [512, 5, 128]
        
        return self.mb_dp(seq_output_mem + seq_output)
    
    def fusing_embedding(self, item_seq):
        
        item_emb = self.item_embedding(item_seq) # [B L H]
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding, dense_embedding = sparse_embedding['item'], dense_embedding['item']

        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)

        # [B L num_features H]
        feature_table = torch.cat(feature_table, dim = -2)
        
        if self.enc == 'att':
            feat_emb = torch.cat([torch.unsqueeze(item_emb, dim = 2), # [B L 1 H]
                                  feature_table], # [B L num_features H]
                                 dim = -2) # [B, L, 1 + num_features, H]
            fuse_emb, attn_weight = self.feature_att_layer(feat_emb) # fuse_emb [B L H], weight [B L num_features]
            
        elif self.enc == 'meancc':
            fuse_emb = torch.cat([item_emb, # [B L H]
                                  feature_table.sum(dim=2) / self.num_feature_field + 1e-24], # [B L H]
                                 dim = -1) # [B, L, 2 * H]
        elif self.enc == 'attcc':
            feat_emb = torch.cat([torch.unsqueeze(item_emb, dim = 2), # [B L 1 H]
                                  feature_table], # [B L num_features H]
                                 dim = -2) # [B, L, 1 + num_features, H]
            fuse_emb, attn_weight = self.feature_att_layer(feat_emb) # fuse_emb [B L H], weight [B L num_features]
            fuse_emb = torch.cat([item_emb, # [B L H]
                                   fuse_emb], # [B L H]
                                 dim = -1) # [B, L, 2 * H]
        
        return fuse_emb # [B, L, 2 * H] or [B, L, H]

    def get_attention_mask(self, sequence):
        attention_mask = (sequence > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(sequence.device)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    

    def forward(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device = item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        
#         item_emb = self.item_embedding(item_seq)
#         input_emb = item_emb + position_embedding
#         input_emb = self.LayerNorm(input_emb)
#         input_emb = self.dropout(input_emb)

        fuse_emb = self.fusing_embedding(item_seq) # [B, L, 2 * H] or [B, L, H]
        input_emb = fuse_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.feature_trm_encoder(input_emb, attention_mask, output_all_encoded_layers=True)
        seq_output = trm_output[-1]  # [B, L, 2 * H]
        
        if self.mem > 0:
            seq_output = self.mem_read(seq_output)  # [B, L, 2 * H]
        
        return seq_output, fuse_emb
    
    
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, fuse_emb = self.forward(item_seq)
        pos_items = interaction[self.POS_ITEM_ID]
        
        # print(f'pos_items.shape: {pos_items.shape}') # [512]
        
        losses = []
        
        if 'BPR' in self.loss_list:
            seq_output_tmp = self.gather_indexes(seq_output, item_seq_len - 1)
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_seq = pos_items.unsqueeze(0) # [1 n_items]
            fuse_pos_items_emb = self.fusing_embedding(pos_items_seq).squeeze(0) # [B, H * 2]
            # print(f'fuse_pos_items_emb.shape: {fuse_pos_items_emb.shape}') # [512, 128]
            neg_items_seq = neg_items.unsqueeze(0) # [1 n_items]
            fuse_neg_items_emb = self.fusing_embedding(neg_items_seq).squeeze(0) # [B, H * 2]
            # pos_items_emb = self.item_embedding(pos_items) 
            # neg_items_emb = self.item_embedding(neg_items) 
            pos_score = torch.sum(seq_output_tmp * fuse_pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output_tmp * fuse_neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct_dict['BPR'](pos_score, neg_score)
            losses.append(loss)
        
        if 'CE' in self.loss_list:  # self.loss_type = 'CE'
            
            seq_output_tmp = self.gather_indexes(seq_output, item_seq_len - 1)
            
            # test_items_emb = self.item_embedding.weight # [n_items H]: [1350, 64]
            # print(f'test_items_emb: {test_items_emb.shape}')
            
            test_items_seq = torch.tensor([i for i in range(self.n_items)], device=item_seq.device).unsqueeze(0) # [1 n_items]
            # print(f'test_items_seq: {test_items_seq.shape}')
            fuse_test_items_emb = self.fusing_embedding(test_items_seq).squeeze(0) 
            # [1 n_items H]: [1, 1350, 128] -> [n_items H]: [1350, 128]
            # print(f'fuse_test_items_emb: {fuse_test_items_emb.shape}')
            
            logits = torch.matmul(seq_output_tmp, fuse_test_items_emb.transpose(0, 1))  # [B, n_items]
            loss = self.loss_fct_dict['CE'](logits, pos_items)
            losses.append(loss)
        
        if 'MILNCE' in self.loss_list:
            
            # multi step prediction, the first step is already calculated as 'pred'
            hidden = self.init_hidden(seq_output.shape[1], item_seq.device) # [1, L, H * 2] or [1 L H]
            pred_out = self.gather_indexes(seq_output, item_seq_len - 1)
            pred = [pred_out]

            # print(f'seq_output[-1].shape: {seq_output[-1].shape}') # [B, L, 2 * H]: [512, 5, 128]
            # print(f'hidden.shape: {hidden.shape}') # [1, L, 2 * H]: [1, 5, 128]

            seq_output_tmp = seq_output
            for _ in range(self.pred_step - 1):

                out, hidden = self.ar(seq_output_tmp, hidden)
                # print(f'out: {out.shape}') # [B, L, 2 * H]: [512, 5, 128]
                # print(f'hidden: {hidden.shape}') # [1, L, 2 * H]: [1, 5, 128]

                # with memory
                if self.mem > 0:
                    seq_output_tmp = self.mem_read(out)
                else:
                    seq_output_tmp = out

                pred_out = self.gather_indexes(seq_output_tmp, item_seq_len - 1)
                pred.append(pred_out)

            # multiple samples
            pos_items_seq = pos_items.unsqueeze(0) # [1 n_items]
            fuse_pos_items_embs = self.fusing_embedding(pos_items_seq).squeeze(0) # [B, H * 2]
            for i in range(self.mil - 1):
                fuse_pos_items_emb = self.fusing_embedding(pos_items_seq).squeeze(0) # [B, 2 * H] or [B, H]
                fuse_pos_items_embs = torch.cat((fuse_pos_items_embs, fuse_pos_items_emb), dim = 0)

            # print(f'fuse_pos_items_embs.shape: {fuse_pos_items_embs.shape}') # [2048, 128]
            # print(f'fuse_embs.shape: {fuse_embs.shape}') # [2048, 128]
            # print(f'torch.cat(pred, dim=0).shape: {torch.cat(pred, dim = 0).shape}') # [1024, 128]
            sim_score = torch.mm(torch.cat(pred, dim=0), self.fuse_dropout(fuse_pos_items_embs).t()) # [1024, 2048]

            loss = self.loss_fct_dict['MILNCE'](sim_score)
            losses.append(self.cl_lambda * loss)
        
        # print(f'loss.shape: {loss.shape}') # [] -> 純數
        
        return tuple(losses)
    
    def info_nce(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        cur_batch_size = z_i.size(0)
        N = 2 * cur_batch_size
        if cur_batch_size != self.batch_size:
            mask = self.mask_correlated_samples(cur_batch_size)
        else:
            mask = self.default_mask
        z = torch.cat((z_i, z_j), dim=0)  # [2B H]
    
        if self.similarity_type == 'cos':
            sim = self.sim(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.tau
        elif self.similarity_type == 'dot':
            sim = self.sim(z, z.T) / self.tau

        sim_i_j = torch.diag(sim, cur_batch_size)
        sim_j_i = torch.diag(sim, -cur_batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # [2B, 1]
        negative_samples = sim[mask].reshape(N, -1)  # [2B, 2(B-1)]

        logits = torch.cat((positive_samples, negative_samples), dim=1)  # [2B, 2B-1]
        # the first column stores positive pair scores
        labels = torch.zeros(N, dtype=torch.long, device=z_i.device)
        return logits, labels
    
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN] 
        test_item = interaction[self.ITEM_ID] # [eval_batch_size]: 2048
        seq_output, _ = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        
        # test_item_emb = self.item_embedding(test_item) # [eval_batch_size H]
        test_item_seq = test_item.unsqueeze(0) # [1 eval_batch_size H]
        fuse_test_item_emb = self.fusing_embedding(test_item_seq).squeeze(0) # [1 eval_batch_size H] -> [eval_batch_size H]
        
        scores = torch.mul(seq_output, fuse_test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _ = self.forward(item_seq)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)
        
        # test_items_emb = self.item_embedding.weight # [n_items H]
        test_items_seq = torch.tensor([i for i in range(self.n_items)], device=item_seq.device).unsqueeze(0) # [1 n_items]
        fuse_test_items_emb = self.fusing_embedding(test_items_seq).squeeze(0) # [1 n_items H] -> [n_items H]
        
        scores = torch.matmul(seq_output, fuse_test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
