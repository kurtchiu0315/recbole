import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.cl4rec import CL4Rec


class EC4SRec(CL4Rec):
    r"""
    TODO
    """

    def __init__(self, config, dataset):
        super(EC4SRec, self).__init__(config, dataset)

        self.training_phase = 'pretrain'
        self.sl_lambda = config['sl_lambda']
        self.scl_lambda = config['scl_lambda']
        self.do_CL = False

    def calculate_loss(self, interaction, get_importance_mode=False):
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

        if get_importance_mode:
            return loss
    
        losses = [loss]

        if self.do_CL:
            aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = \
                interaction['aug1'], interaction['aug_len1'], interaction['aug2'], interaction['aug_len2']

            aug_output1 = self.forward(aug_item_seq1, aug_len1)
            logits, labels = self.info_nce(seq_output, aug_output1)
            if self.cl_loss_type == 'dcl': # decoupled contrastive learning
                cl_loss1 = self.calculate_decoupled_cl_loss(logits, labels)
            else: # original infonce
                cl_loss1 = self.cl_loss_fct(logits, labels)
            losses.append(self.cl_lambda * cl_loss1)


            aug_output2 = self.forward(aug_item_seq2, aug_len2)
            logits, labels = self.info_nce(seq_output, aug_output2)
            if self.cl_loss_type == 'dcl': # decoupled contrastive learning
                cl_loss2 = self.calculate_decoupled_cl_loss(logits, labels)
            else: # original infonce
                cl_loss2 = self.cl_loss_fct(logits, labels)
            losses.append(self.sl_lambda * cl_loss2)


            logits, labels = self.info_nce(aug_output1, aug_output2)
            if self.cl_loss_type == 'dcl': # decoupled contrastive learning
                cl_loss3 = self.calculate_decoupled_cl_loss(logits, labels)
            else: # original infonce
                cl_loss3 = self.cl_loss_fct(logits, labels)
            losses.append(self.scl_lambda * cl_loss3)

        return tuple(losses)
