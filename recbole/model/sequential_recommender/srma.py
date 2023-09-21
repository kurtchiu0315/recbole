# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.layers import AugTransformerEncoder


class SRMA(SASRec):

    def __init__(self, config, dataset):
        super(SRMA, self).__init__(config, dataset)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.tau = config['tau']
        self.cl_lambda = config['cl_lambda']
        self.cl_loss_type = config['cl_loss_type']
        self.similarity_type = config['similarity_type']
        
        # for SRMA
        self.model_aug = config['model_aug'] # T or F
        self.aug_layer_type = config['aug_layer_type'] # [single, inter]
        self.layer_drop_num = config['layer_drop_num'] # number of layers to drop
        self.n_aug_layer = config['n_aug_layer'] # number of augmented layers
        self.layer_drop_thres = config['layer_drop_thres']
        self.cl_type = config['cl_type'] # [same, sasrec-static]
        self.en_weight = config['en_weight'] # weight of encoder complementing
        
        # aug_trm_encoder: trm_encoder  w/ model_augmentation
        self.aug_trm_encoder = AugTransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            aug_layer_type=self.aug_layer_type,
            layer_drop_num=self.layer_drop_num,
            n_aug_layer=self.n_aug_layer,
            layer_drop_thres=self.layer_drop_thres 
        )

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # for mask
        self.default_mask = self.mask_correlated_samples(self.batch_size)

        if self.similarity_type == 'dot':
            self.sim = torch.mm
        elif self.similarity_type == 'cos':
            self.sim = F.cosine_similarity

        if self.cl_loss_type == 'infonce':
            self.cl_loss_fct = nn.CrossEntropyLoss()
        
        # parameters initialization
        self.apply(self._init_weights)

    def forward(self, item_seq, item_seq_len, isTrain = False):
        
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        
        if self.model_aug:
            trm_output = self.aug_trm_encoder(input_emb, 
                                              extended_attention_mask, 
                                              output_all_encoded_layers=True,
                                              isTrain=isTrain)
        else:
            trm_output = self.trm_encoder(input_emb, 
                                          extended_attention_mask, 
                                          output_all_encoded_layers=True)
        
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        
        return output  # [B H]
    
    def mask_correlated_samples(self, batch_size):
        N = batch_size
        mask = torch.ones((2 * N, 2 * N)).bool()
        mask = mask.fill_diagonal_(0)
        mask *= ~ torch.diagflat(torch.ones(N), offset=N).bool()
        mask *= ~ torch.diagflat(torch.ones(N), offset=-N).bool()
        return mask

    def calculate_loss(self, interaction):
    
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len, isTrain = True)
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
        
        aug_item_seq1, aug_len1 = interaction['aug1'], interaction['aug_len1']
        aug_item_seq2, aug_len2 = interaction['aug2'], interaction['aug_len2']
        seq_output1 = self.forward(aug_item_seq1, aug_len1, isTrain = True)
        seq_output2 = self.forward(aug_item_seq2, aug_len2, isTrain = True)
        
        if self.cl_type == 'same':
            
            logits, labels = self.info_nce(seq_output1, seq_output2)
            cl_loss = self.cl_loss_fct(logits, labels)
            losses.append(cl_loss * self.cl_lambda)
            
        elif self.cl_type == 'sasrec-distinct':
            
            for aug_seq_output in [seq_output1, seq_output2]:
                
                logits, labels = self.info_nce(seq_output, aug_seq_output)
                cl_loss = self.cl_loss_fct(logits, labels)
                losses.append(cl_loss * self.en_weight)
        
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

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
