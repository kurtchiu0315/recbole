import torch
import torch.nn.functional as F
from torch import nn
import random

from recbole.model.sequential_recommender.cl4rec import CL4Rec


class DADACLRec(CL4Rec):
  
    def __init__(self, config, dataset):
        super(DADACLRec, self).__init__(config, dataset)
        self.scl_lambda = config['scl_lambda']
        self.aug_for_train = config['aug_for_train']
        if config['consider_su_aug']:
            self.scl_lambda = 0


    def calculate_loss(self, interaction, aug_chosen, aug_chosen_len):
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
            logits = torch.matmul( seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        losses = [loss]

        if self.aug_for_train:
            test_item_emb = self.item_embedding.weight
            aug_item_seq, aug_len = aug_chosen, aug_chosen_len
            aug_output = self.forward(aug_item_seq, aug_len, converted_to_emb=True)
            logits = torch.matmul( aug_output, test_item_emb.transpose(0, 1))
            losses.append(
                self.cl_lambda*self.loss_fct(logits, pos_items)
            )
        else:
            cl_loss, aug_output = self.CL_loss(interaction, aug_chosen, aug_chosen_len, seq_output=seq_output, return_aug_output=True)
            losses.append(cl_loss)
            if self.scl_lambda > 0:
                su_aug, su_aug_len = interaction['su_aug'], interaction['su_aug_len']
                su_aug_output = self.forward(su_aug, su_aug_len)
                logits, labels = self.info_nce(su_aug_output, aug_output)

                losses.append(
                    self.scl_lambda * self.cl_loss_fct(logits, labels)
                )

        return tuple(losses)
    
    def CL_loss(self, interaction, aug_chosen, aug_chosen_len, seq_output=None, return_aug_output=False):
        if seq_output == None:
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            seq_output = self.forward(item_seq, item_seq_len)
    
        aug_item_seq, aug_len = aug_chosen, aug_chosen_len
        aug_output = self.forward(aug_item_seq, aug_len, converted_to_emb=True)

        logits, labels = self.info_nce(seq_output, aug_output)

        loss = self.cl_lambda * self.cl_loss_fct(logits, labels)
        if return_aug_output:
            return loss, aug_output
        else:
            return loss 
            
    

    def forward(self, item_seq, item_seq_len, converted_to_emb=False):
        if not converted_to_emb:
            return super().forward(item_seq, item_seq_len)
        else:
            position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
            position_ids = position_ids.unsqueeze(0).expand(item_seq.shape[0:2])
            position_embedding = self.position_embedding(position_ids)

            item_emb = item_seq
            input_emb = item_emb + position_embedding
            input_emb = self.LayerNorm(input_emb)
            input_emb = self.dropout(input_emb)

            auxiliary_item_seq = torch.zeros(item_seq.shape[0:2], device=item_seq.device)
            for index, length in enumerate(item_seq_len):
                auxiliary_item_seq[index][0:length] = 1
            extended_attention_mask = self.get_attention_mask(auxiliary_item_seq)

            trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
            output = trm_output[-1]
            output = self.gather_indexes(output, item_seq_len - 1)
            return output  # [B H]
    
    

class PolicyChooser(nn.Module):
    def __init__(self, aug_num, config):
        super(PolicyChooser, self).__init__()

        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.policy_trm_num_layers = config['policy_trm_num_layers']
        self.MAX_ITEM_LIST_LENGTH = config['MAX_ITEM_LIST_LENGTH']

        self.policy_trm_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.MAX_ITEM_LIST_LENGTH, 
                nhead=1 ,
                dim_feedforward=self.inner_size,
                dropout=self.hidden_dropout_prob
            ),
            num_layers=self.policy_trm_num_layers
        )
        self.fc_before_trm = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//4),
            nn.Dropout(self.hidden_dropout_prob),
            nn.ReLU(),
            nn.Linear(self.hidden_size//4, 1)
        )

        self.fc_after_trm = nn.Sequential(
            nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(aug_num* self.MAX_ITEM_LIST_LENGTH, aug_num)
        )

        self.apply(self._init_weights)

        

    def forward(self, data, item_embeddings):
        self.fc_before_trm.to(data.device)
        self.policy_trm_encoder.to(data.device)
        self.fc_after_trm.to(data.device)

        # data: [B, aug_num, MAXLEN]
        out = self.fc_before_trm(item_embeddings(data))
        out = out.squeeze(-1)
        # out: [B, aug_num, MAXLEN]

        out = self.policy_trm_encoder(out)
        out = out.reshape(len(data), -1)
        out = self.fc_after_trm(out)
        # out: [B, aug_num]

        return out
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=1.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def _init_weights_biasver(self, module):
        
        mean_bias = random.random()
        std_bias = random.random()

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0+mean_bias, std=1.0+std_bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.fill_(0+mean_bias)
            module.weight.data.fill_(1.0+mean_bias)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.fill_(0+mean_bias)

