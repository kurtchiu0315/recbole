r"""
BiCAT
"""

import torch
from torch import nn

from recbole.model.sequential_recommender.sasrec import SASRec



class BiCAT(SASRec):
    def __init__(self, config, dataset, ):
        super(BiCAT, self).__init__(config, dataset)
        self.num_prior = config['num_prior'] # K

        self.position_embedding = nn.Embedding(self.max_seq_length + self.num_prior, self.hidden_size)
        self.lamb =  config['lamb'] or 0.5
        self.alpha = config['alpha'] or 10

        self.loss_type = config['loss_type']
        self.bceloss =  nn.BCEWithLogitsLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()

        if self.loss_type == 'BCE':
            self.main_loss = self.bceloss
        else:
            self.main_loss = self.ce_loss


    def sas_calculate_loss(self, interaction):
        return  super().calculate_loss(interaction)

    def prev_item_generator_training(self, interaction):

        # item_seq[1:] -> POS_ITEM
        # rev_item_seq[0:-1] -> first_item
        # e.g., For item_seq == [1,2,3,4,5,6,7,8,9], POS_ITEM == 10
        # train_phase:
        # [2,3,4,5,6,7,8,9] -> 10
        # [9,8,7,6,5,4,3,2] -> 1

        losses = []
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        can_use_to_train = (item_seq_len > 1) 
        # Since some seq len is 1, we can not use 0 items to predict the last 1 item 

        rev_item_seq = torch.flip(item_seq, dims=[1])

        first_items = (rev_item_seq[:,-1])[can_use_to_train]
        v_n = (interaction[self.POS_ITEM_ID])[can_use_to_train]

        if self.loss_type == 'BCE':
            # change labels to 1-hot encoding
            first_items = self.label_to_one_hot(first_items, max_value=self.n_items)
            v_n = self.label_to_one_hot(v_n, max_value=self.n_items)
        

        # L_R (pseudo_first <-> first_items)
        pseudo_first =  self.forward(
            (rev_item_seq[:,0:-1])[can_use_to_train], 
            item_seq_len[can_use_to_train]-1
        )
        l_r = self.main_loss(
            torch.matmul(pseudo_first, self.item_embedding.weight.transpose(0, 1)), 
            first_items
        )

        # L_F (v_n_hat <-> v_n)
        v_n_hat = self.forward(
            (item_seq[:, 1:])[can_use_to_train] , 
            item_seq_len[can_use_to_train]-1
        )
        l_f = self.main_loss(
            torch.matmul(v_n_hat, self.item_embedding.weight.transpose(0, 1)),
            v_n
        )
        losses = (l_r, self.lamb*l_f)

        return losses

    @torch.no_grad()
    def augment_seq_generator(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        rev_item_seq = torch.flip(item_seq,dims=[1])

        logits = torch.matmul(
            self.forward(rev_item_seq, item_seq_len), 
            self.item_embedding.weight.transpose(0, 1)
        )

        pseudo_first = logits.argmax(dim=1)
        new_gen_seq = torch.cat([pseudo_first.reshape(-1,1), item_seq], dim = 1)
        new_gen_seq_len = interaction[self.ITEM_SEQ_LEN] + 1
        label = interaction[self.POS_ITEM_ID]

        return {
            'new_gen_seq': new_gen_seq,
            'new_gen_seq_len': new_gen_seq_len,
            'label': label
        }
    

    
    def label_to_one_hot(self, vector, max_value = None):
        if max_value:
            one_hot = torch.zeros((len(vector), max_value)).to(vector.device)
        else:
            one_hot = torch.zeros((len(vector), max(vector) + 1)).to(vector.device)
        
        one_hot[torch.arange(one_hot.shape[0]), vector] = 1
        return one_hot
        

    def Loss_KL(self, logits1, logits2):
        # logits1: augmented seq out
        # logits2: original seq out
        L1 = 0.5* self.kl_loss(
            torch.log_softmax(logits1, dim=1), 
            torch.log_softmax(logits2, dim=1), 
        )

        L2 = self.kl_loss(
            torch.log_softmax(logits2, dim=1), 
            torch.log_softmax(logits1, dim=1), 
        )

        return L1+L2


    def calculate_loss(self, interaction):
        seq_aug = interaction[self.ITEM_SEQ]
        seq_aug_len = interaction[self.ITEM_SEQ_LEN]

        seq_orig = (interaction[self.ITEM_SEQ][:,self.num_prior:]).detach().clone()
        seq_orig_len = seq_aug_len.detach().clone() - self.num_prior
        
        label = interaction[self.POS_ITEM_ID]
      
        aug_out = self.forward(seq_aug, seq_aug_len)
        orig_out = self.forward(seq_orig, seq_orig_len)

        aug_out_distribution = torch.matmul(aug_out, self.item_embedding.weight.transpose(0, 1))
        orig_out_distribution = torch.matmul(orig_out, self.item_embedding.weight.transpose(0, 1))

        if self.loss_type == 'BCE':
            label = self.label_to_one_hot(label, max_value=self.n_items)

        loss = self.main_loss(
            aug_out_distribution,
            label
        )

        Kloss = self.Loss_KL(aug_out_distribution, orig_out_distribution)
        return (loss, self.alpha * Kloss) 
        

        
    


