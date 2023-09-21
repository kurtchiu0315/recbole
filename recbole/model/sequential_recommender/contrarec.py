# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.sequential_recommender.bert4rec import BERT4Rec


class ContraRec(SASRec):
    
    def __init__(self, config, dataset):
        
        super(ContraRec, self).__init__(config, dataset)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.tau = config['tau']
        self.cl_lambda = config['cl_lambda']
        self.cl_loss_type = config['cl_loss_type']
        self.similarity_type = config['similarity_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # for mask
        self.default_mask = self.mask_correlated_samples(self.batch_size)

        if self.similarity_type == 'dot':
            self.sim = torch.mm
        elif self.similarity_type == 'cos':
            self.sim = F.cosine_similarity

        if self.cl_loss_type == 'infonce':
            self.cl_loss_fct = nn.CrossEntropyLoss()
        elif self.cl_loss_type == 'revised_infonce':
            self.cl_loss_fct = self.revised_info_nce_loss
        
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
        
        loss = super().calculate_loss(interaction)
        
        aug_item_seq1, aug_len1 = interaction['aug1'], interaction['aug_len1']
        aug_item_seq2, aug_len2 = interaction['aug2'], interaction['aug_len2']
        
        seq_output1 = self.forward(aug_item_seq1, aug_len1)
        seq_output2 = self.forward(aug_item_seq2, aug_len2)
        
        if self.cl_loss_type == 'infonce':
            logits, labels = self.info_nce(seq_output1, seq_output2)
            cl_loss = self.cl_loss_fct(logits, labels)
        else:
            pos_items = interaction[self.POS_ITEM_ID]
            cl_loss = self.cl_loss_fct(seq_output1, seq_output2, pos_items)
        
        return loss, self.cl_lambda * cl_loss
   
    def revised_info_nce_loss(self, z_i, z_j, labels):
        """
        Revise info_nce() from CL4Rec
        """
        
        labels = labels.contiguous().view(-1, 1) # [B, 1]
        mask = torch.eq(labels, labels.transpose(0, 1)).float().to(z_i.device) # [B, B]
        
        z = torch.cat((z_i, z_j), dim=0) # [2B H]
    
        if self.similarity_type == 'cos': 
            sim = self.sim(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.tau
        elif self.similarity_type == 'dot':
            sim = self.sim(z, z.T) / self.tau
        
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()
        
        mask = mask.repeat(2, 2) # [2B, 2B]
        logits_mask = torch.scatter( # [2B, 2B]
            torch.ones_like(mask), 1,
            torch.arange(mask.shape[0]).view(-1, 1).to(z_i.device), 0
        ) 
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask # [2B, 2B]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10) # [2B, 2B]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10) # [2B, 2B]

        # loss
        loss = - self.tau * mean_log_prob_pos
        
        return loss.mean()
    
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
        
        z = torch.cat((z_i, z_j), dim=0) # [2B H]
    
        if self.similarity_type == 'cos':
            sim = self.sim(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.tau
        elif self.similarity_type == 'dot':
            sim = self.sim(z, z.T) / self.tau
        
        sim_i_j = torch.diag(sim, cur_batch_size)  
        sim_j_i = torch.diag(sim, -cur_batch_size) 

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # [2B, 1]
        negative_samples = sim[mask].reshape(N, -1)  

        logits = torch.cat((positive_samples, negative_samples), dim=1) # [2B, 2B-1]
        # the first column stores positive pair scores
        labels = torch.zeros(N, dtype=torch.long, device=z_i.device) # [2B]
        
        return logits, labels

    def full_sort_predict(self, interaction):
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        
        return scores
