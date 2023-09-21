import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import random
import numpy as np
from recbole.model.sequential_recommender.sasrec import SASRec


class DSSRec(SASRec):
    
    def __init__(self, config, dataset):
        super(DSSRec, self).__init__(config, dataset)
        self.disentangled_encoder = DSSEncoder(config)
        self.lamb =  config['lamb'] or 0.1
        self.train_batch_size = config['train_batch_size']
        self.num_intents = config['num_intents'] or 1


    def SAS_encoder(self, item_seq, output_all_encoded_layers=False):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers)
        if output_all_encoded_layers:
            return  torch.stack(trm_output, dim=0)  # [layers B L H]
        else:
            return  trm_output[-1]  # [B L H]



    def __seq2itemloss(self,
                       inp_subseq_encodings,  next_item_emb):
        sqrt_hidden_size = np.sqrt(self.hidden_size)

        next_item_emb = torch.transpose(next_item_emb, 1, 2)  # [B, D, 1]
        dot_product = torch.matmul(inp_subseq_encodings, next_item_emb)  # [B, K, 1]

        reduce_scale = torch.tensor([0.0]).to(inp_subseq_encodings.device)
        with torch.no_grad():
            if dot_product.max() > 20:
                reduce_scale = dot_product.max() - 20

        exp_normalized_dot_product = torch.exp(dot_product / sqrt_hidden_size - reduce_scale)
        numerator = torch.max(exp_normalized_dot_product, dim=1)[0]  # [B, 1]

        inp_subseq_encodings_trans = inp_subseq_encodings.transpose(0, 1)  # [K, B, D]
        next_item_emb_trans = next_item_emb.squeeze(-1).transpose(0, 1)  # [D, B]
        # sum of dot products of given input sequence encoding for each intent with all next item embeddings
        dot_products = torch.matmul(inp_subseq_encodings_trans,
                                    next_item_emb_trans) /sqrt_hidden_size  # [K, B, B]
                                    
        dot_products = torch.exp(dot_products - reduce_scale)  # [K, B, B]
        dot_products = dot_products.sum(-1)
        dot_products = dot_products.transpose(0, 1)  # [B, K]
        # sum across all intents
        denominator = dot_products.sum(-1).unsqueeze(-1)  # [B, 1]
        seq2item_loss_k = -(torch.log(0.0001 + numerator) - torch.log(0.0001 + denominator))  # [B, 1]
        seq2item_loss = torch.sum(seq2item_loss_k)


        return seq2item_loss


    def __seq2seqloss(self, inp_subseq_encodings, label_subseq_encodings):
     
        sqrt_hidden_size = np.sqrt(self.hidden_size)

        reduce_scale = torch.tensor([0.0]).to(inp_subseq_encodings.device)
       
        product = torch.mul(inp_subseq_encodings, label_subseq_encodings) # [B, K, D]
 
        log_numerator = torch.sum(product, dim=-1) / sqrt_hidden_size  # [B, K]
        with torch.no_grad():
            if log_numerator.max() > 20:
                reduce_scale = log_numerator.max() - 20
        numerator = torch.exp(log_numerator - reduce_scale)  # [B, K]
        
        
        inp_subseq_encodings_trans = inp_subseq_encodings.transpose(0, 1)  # [K, B, D]
        inp_subseq_encodings_trans_expanded = inp_subseq_encodings_trans.unsqueeze(1)  # [K, 1, B, D]
        label_subseq_encodings_trans = label_subseq_encodings.transpose(0, 1).transpose(1, 2)  # [K, D, B]
        dot_products = torch.matmul(inp_subseq_encodings_trans_expanded, label_subseq_encodings_trans)  # [K, K, B, B]
        dot_products = torch.exp(dot_products / sqrt_hidden_size - reduce_scale)
        dot_products = (dot_products.sum(-1)).sum(1)   # [K, K, B] -> [K, B]
        denominator = dot_products.transpose(0, 1)  # [B, K]

        seq2seq_loss_k = -(torch.log(0.0001 + numerator) - torch.log(0.0001 + denominator))
        seq2seq_loss_k = torch.flatten(seq2seq_loss_k)
        
        thresh_th =  int(np.floor( self.lamb * self.train_batch_size  *  self.num_intents))
        if seq2seq_loss_k.shape[0]-1 < thresh_th:
            thresh_th = random.randint(0, seq2seq_loss_k.shape[0]-1)
  
        thresh = torch.kthvalue(seq2seq_loss_k, thresh_th)[0]
     
        conf_indicator = seq2seq_loss_k <= thresh
        conf_seq2seq_loss_k = torch.mul(seq2seq_loss_k, conf_indicator)
        seq2seq_loss = torch.sum(conf_seq2seq_loss_k)
    
        
        return seq2seq_loss

    

    
    def pretrain(self, interaction):
        next_items = interaction['next_items']
        next_items_emb = self.item_embedding(next_items)  # [B, 1, D]
        inp_subseqs = interaction['inp_subseqs']
        label_subseqs = interaction['label_subseqs']
     
        input_subseq_encoding = self.SAS_encoder(inp_subseqs)
        label_subseq_encoding = self.SAS_encoder(label_subseqs)

        disent_inp_subseq_encodings = self.disentangled_encoder(True,
                                                               input_subseq_encoding)
        disent_label_seq_encodings = self.disentangled_encoder(False,
                                                               label_subseq_encoding)
        # seq2item loss
        seq2item_loss = self.__seq2itemloss(disent_inp_subseq_encodings, next_items_emb)
        # seq2seq loss
        seq2seq_loss = self.__seq2seqloss(disent_inp_subseq_encodings, disent_label_seq_encodings)

        return seq2item_loss , seq2seq_loss


    def calculate_loss(self, interaction):
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type in ['BPR', 'BCE']:
            neg_items = self.construct_neg_items(pos_items)
            
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            if self.loss_type == 'BPR':
                loss = self.loss_fct(pos_score, neg_score)
            else:
                loss = self.loss_fct(pos_score, torch.ones_like(pos_score)) + \
                       self.loss_fct(neg_score, torch.zeros_like(neg_score))
    

            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss



class DSSEncoder(nn.Module):
    def __init__(self, config):
        super(DSSEncoder, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_intents = config['num_intents']
        self.layer_norm_eps = config['layer_norm_eps']
        self.max_seq_len = config['MAX_ITEM_LIST_LENGTH']

        self.prototypes = nn.ParameterList([nn.Parameter(torch.randn(self.hidden_size) *
                                                         (1 / np.sqrt(self.hidden_size)))
                                            for _ in range( self.num_intents)])

        self.layernorm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.layernorm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.layernorm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.layernorm5 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.w = nn.Linear(self.hidden_size, self.hidden_size)

        self.b_prime = nn.Parameter(torch.zeros(self.hidden_size))
 
        # individual alpha for each position
        self.alphas = nn.Parameter(torch.zeros(self.max_seq_len, self.hidden_size))

        self.beta_input_seq = nn.Parameter(torch.randn(self.num_intents, self.hidden_size) *
                                           (1 / np.sqrt(self.hidden_size)))

        self.beta_label_seq = nn.Parameter(torch.randn(self.num_intents, self.hidden_size) *
                                           (1 / np.sqrt(self.hidden_size)))

    def _intention_clustering(self, z):
        """
        Method to measure how likely the primary intention at position i
        is related with kth latent category
        :param z:
        :return:
        """
        z = self.layernorm1(z)
        exp_normalized_numerators = list()
        i = 0
        for prototype_k in self.prototypes:
            prototype_k = self.layernorm2(prototype_k)  # [D]
            numerator = torch.matmul(z, prototype_k)  # [B, S]
            exp_normalized_numerator = torch.exp(numerator / np.sqrt(self.hidden_size))  # [B, S]
            exp_normalized_numerators.append(exp_normalized_numerator)
            if i == 0:
                denominator = exp_normalized_numerator
            else:
                denominator = torch.add(denominator, exp_normalized_numerator)
            i = i + 1

        all_attentions_p_k_i = [torch.div(k, denominator)
                                for k in exp_normalized_numerators]  # [B, S] K times
        all_attentions_p_k_i = torch.stack(all_attentions_p_k_i, -1)  # [B, S, K]

        return all_attentions_p_k_i

    def _intention_weighting(self, z) :
        """
        Method to measure how likely primary intention at position i
        is important for predicting user's future intentions
        :param z:
        :return:
        """
        keys_tilde_i = self.layernorm3(z + self.alphas)  # [B, S, D]
        keys_i = keys_tilde_i + torch.relu(self.w(keys_tilde_i))  # [B, S, D]
        query = self.layernorm4(self.b_prime + self.alphas[-1, :] + z[:, -1, :])  # [B, D]
        query = torch.unsqueeze(query, -1)  # [B, D, 1]

        numerators = torch.matmul(keys_i, query)  # [B, S, 1]
        exp_normalized_numerators = torch.exp(numerators / np.sqrt(self.hidden_size))
        sum_exp_normalized_numerators = exp_normalized_numerators.sum(1).unsqueeze(-1)  # [B, 1] to [B, 1, 1]
        all_attentions_p_i = exp_normalized_numerators / sum_exp_normalized_numerators  # [B, S, 1]
        all_attentions_p_i = all_attentions_p_i.squeeze(-1)  # [B, S]

        return all_attentions_p_i

    def _intention_aggr(self, z, attention_weights_p_k_i, attention_weights_p_i, is_input_seq):
        """
        Method to aggregate intentions collected at all positions according
        to both kinds of attention weights
        :param z:
        :param attention_weights_p_k_i:
        :param attention_weights_p_i:
        :param is_input_seq: bool
        :return:
        """
        attention_weights_p_i = attention_weights_p_i.unsqueeze(-1)  # [B, S, 1]
        attention_weights = torch.mul(attention_weights_p_k_i, attention_weights_p_i)  # [B, S, K]
        attention_weights_transpose = attention_weights.transpose(1, 2)  # [B, K, S]
        if is_input_seq:
            disentangled_encoding = self.beta_input_seq + torch.matmul(attention_weights_transpose, z)
        else:
            disentangled_encoding = self.beta_label_seq + torch.matmul(attention_weights_transpose, z)

        disentangled_encoding = self.layernorm5(disentangled_encoding)
    
        return disentangled_encoding  # [B, K, D]

    def forward(self, is_input_seq, z):
  
        attention_weights_p_k_i = self._intention_clustering(z)  # [B, S, K]
        attention_weights_p_i = self._intention_weighting(z)  # [B, S]

        disentangled_encoding = self._intention_aggr(z,
                                                     attention_weights_p_k_i,
                                                     attention_weights_p_i,
                                                     is_input_seq)

        return disentangled_encoding