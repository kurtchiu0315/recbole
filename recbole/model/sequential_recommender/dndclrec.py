import torch
import torch.nn.functional as F
from torch import nn
import random

from recbole.model.sequential_recommender.cl4rec import CL4Rec
from recbole.data.interaction import Interaction

class DNDCLRec(CL4Rec):
    def __init__(self, config, dataset):
        super(DNDCLRec, self).__init__(config, dataset)
        self.celoss_no_reduction = nn.CrossEntropyLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(log_target=True, reduction="none")
        self.alpha = config['alpha']
        self.hard = config['hard'] or 2
        self.W_sim = nn.Sequential(
            nn.Linear(config['hidden_size']*3, 2),

        )
        self.test_mode = False
        self.view_difficulty = config['view_difficulty'] or 10.0

    def calculate_loss(self, interaction, aug_chosen, aug_chosen_len, reward_mode=False):
        losses = []
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # item_emb = self.item_embedding(item_seq)
        seq_output = self.forward(item_seq, item_seq_len)
        aug_output, aug_view = self.forward( aug_chosen, aug_chosen_len, has_converted_to_emb=True)
        pos_items = interaction[self.POS_ITEM_ID]
       
        if reward_mode:
            seq_output, seq_view = self.forward(self.item_embedding(item_seq), item_seq_len, has_converted_to_emb=True)
            # return self.Reward(seq_output, aug_output, seq_view, aug_view)
            result = self.performance_reward(seq_output, pos_items, aug_output, seq_view, aug_view)
                
            return result["losses"],  result["performance"]
        else:
            if self.test_mode:
                self.Test(item_seq, item_seq_len, aug_chosen, aug_chosen_len)
            
            test_item_emb = self.item_embedding.weight
            x_prob = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            # x_aug_prob = torch.matmul(aug_output, test_item_emb.transpose(0, 1))

            # cofidence_x = torch.gather(x_prob.softmax(dim=1), dim=1, index=pos_items.view(-1,1)).squeeze(1)
            # cofidence_x_aug = torch.gather(x_aug_prob.softmax(dim=1), dim=1, index=pos_items.view(-1,1)).squeeze(1)
           
            # weight = torch.pow(cofidence_x, self.alpha) * torch.pow((cofidence_x-cofidence_x_aug).relu(), 1-self.alpha)
            rec_loss = self.loss_fct(x_prob, pos_items)
            losses.append(rec_loss)
            # losses.append(rec_loss)
            # aug_rec_loss =  self.celoss_no_reduction(x_aug_prob, pos_items)
            # aug_rec_loss =  weight.mean()*self.celoss_no_reduction(x_aug_prob, pos_items)
            # losses.append(aug_rec_loss.mean())
            
            cl_loss = self.cl_loss_fct(*self.info_nce(seq_output, aug_output))
            # cl_loss =  self.celoss_no_reduction(*self.info_nce(seq_output, aug_output))
            # with torch.no_grad():
            #     aug_x_prob = torch.matmul(aug_output, test_item_emb.transpose(0, 1)).softmax(dim=1)
            #     performace = F.cosine_similarity(x_prob, aug_x_prob) > 0.3
            # cl_loss = torch.cat([performace.detach()]*2, dim=0) * cl_loss
            losses.append(self.cl_lambda * cl_loss.mean())

            # kl_loss = 0.5*self.kl_loss(seq_output.log_softmax(dim=1), aug_output.log_softmax(dim=1)) + \
            #     0.5*self.kl_loss(aug_output.log_softmax(dim=1), seq_output.log_softmax(dim=1)) 

            # losses.append(kl_loss.mean())
            return tuple(losses) 

    def Reward(self, seq_output, aug_output, seq_view, aug_view):
        # with torch.no_grad():
        #     cl_loss = self.celoss_no_reduction(*self.info_nce(seq_output, aug_output))
        
        # kl_loss = self.kl_loss(seq_view.log_softmax(dim=1), aug_view.log_softmax(dim=1)) + \
        #         self.kl_loss(aug_view.log_softmax(dim=1), seq_view.log_softmax(dim=1)) 

        # loss = 10.0 * (cl_loss * torch.hstack([kl_loss.mean(dim=1),kl_loss.mean(dim=1)]))

        cl_loss = self.cl_loss_fct(*self.info_nce(seq_output, aug_output))
        # cl_loss2 = 0.01* self.cl_loss_fct(*self.info_nce(seq_view, aug_view))
        M = 0.5*(seq_view.log_softmax(dim=1) + aug_view.log_softmax(dim=1))
        JS_div = self.kl_loss(seq_view.log_softmax(dim=1), M) + self.kl_loss(aug_view.log_softmax(dim=1), M) 
        # kl_loss = self.kl_loss(seq_view.log_softmax(dim=1), aug_view.log_softmax(dim=1))
        
        # return self.cl_lambda*cl_loss, -self.cl_lambda*cl_loss2
        
        return self.cl_lambda*cl_loss, -self.view_difficulty*JS_div.mean()
    
    
    def performance_reward(self, seq_output, pos_items, aug_output, seq_view, aug_view):
        def mapping(x):
            return torch.arctan(x)/torch.pi + 0.5

        with torch.no_grad():
            test_item_emb = self.item_embedding.weight
            x_prob = torch.matmul(aug_output, test_item_emb.transpose(0, 1))
            rec_loss = self.celoss_no_reduction(x_prob, pos_items)
            performance = (rec_loss.mean()-rec_loss).sigmoid() #[B]
        
    
        cl_loss = self.cl_loss_fct(*self.info_nce(seq_output, aug_output))
    
        kl_loss = self.kl_loss(seq_view.log_softmax(dim=1), aug_view.log_softmax(dim=1))
        

        return {
            "losses": (
                self.cl_lambda*0.5*cl_loss, 
                -self.view_difficulty * (performance*kl_loss.mean(dim=1)).mean(),
            ),
            "performance": 0.5*(performance-0.5)
        }
    
    
    def performance_reward2(self, seq_output, pos_items, aug_output, seq_view, aug_view):

        with torch.no_grad():
            test_item_emb = self.item_embedding.weight
            x_prob = torch.matmul(seq_output, test_item_emb.transpose(0, 1)).softmax(dim=1)
            aug_x_prob = torch.matmul(aug_output, test_item_emb.transpose(0, 1)).softmax(dim=1)
            #  print(F.cosine_similarity(x_prob, aug_x_prob))
            performace = F.cosine_similarity(x_prob, aug_x_prob) - 0.3
    

        cl_loss = self.cl_loss_fct(*self.info_nce(seq_output, aug_output))
        # cl_loss = self.celoss_no_reduction(*self.info_nce(seq_output, aug_output))
        
        kl_loss = self.kl_loss(seq_view.log_softmax(dim=1), aug_view.log_softmax(dim=1))
        
        return self.cl_lambda*(cl_loss), -self.view_difficulty* (performace * kl_loss.mean(dim=1)).mean()
    
    def Test(self, item_seq, item_seq_len, aug_chosen, aug_chosen_len):
        seq_views = self.forward(self.item_embedding(item_seq), item_seq_len, has_converted_to_emb=True, test_mode=True)
        aug_views = self.forward( aug_chosen, aug_chosen_len, has_converted_to_emb=True, test_mode=True)
        with torch.no_grad():     
            for i, (seq_view, aug_view) in enumerate(zip(seq_views, aug_views)):
                kl_loss = self.kl_loss(seq_view.log_softmax(dim=1), aug_view.log_softmax(dim=1)) + \
                        self.kl_loss(aug_view.log_softmax(dim=1), seq_view.log_softmax(dim=1)) 
                info = kl_loss.mean().cpu().numpy()
                print(f"LAYER {i} KL_Div: ")
                print(info)
        
    
    def embs_similarity(self, x1, x2, positive_sample=True):
        out = self.W_sim(torch.hstack([x1, x2, torch.abs(x1-x2)])) + 1e-8
        if positive_sample:
            return self.celoss_no_reduction(out, torch.ones(x1.shape[0], dtype=torch.long, device=x1.device)).mean()
        else:
            return self.celoss_no_reduction(out, torch.zeros(x1.shape[0], dtype=torch.long, device=x1.device)).mean()

    def generate_negative_embs(self, seq_output):
        assert seq_output.shape[0] > 1
        negative_embs = seq_output.sum(dim=0).unsqueeze(0).repeat((seq_output.shape[0],1))
        negative_embs = negative_embs - seq_output
        return negative_embs / (seq_output.shape[0]-1)


    def forward(self, item_seq, item_seq_len, has_converted_to_emb=False, test_mode=False):
        if not has_converted_to_emb:
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

            seq_summarize = trm_output[0]
            seq_summarize = seq_summarize.sum(dim=1)

            if test_mode:
                return [trm_output[i].sum(dim=1) for i in range(len(trm_output))]
            
            return output, seq_summarize  # [B H]
        

    

# class PolicyChooser(nn.Module):
#     def __init__(self, aug_num, config):
#         super(PolicyChooser, self).__init__()

#         self.n_heads = config['n_heads']
#         self.hidden_size = config['hidden_size']  # same as embedding_size
#         self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
#         self.hidden_dropout_prob = config['hidden_dropout_prob']
#         self.policy_trm_num_layers = config['policy_trm_num_layers']
#         self.MAX_ITEM_LIST_LENGTH = config['MAX_ITEM_LIST_LENGTH']
#         self.loss_fn = nn.CrossEntropyLoss()

#         self.policy_trm_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=self.MAX_ITEM_LIST_LENGTH, 
#                 nhead=2 ,
#                 dim_feedforward=64,
#                 dropout=0.1
#             ),
#             num_layers=self.policy_trm_num_layers
#         )
#         self.hidden_trm_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=self.hidden_size, 
#                 nhead=2 ,
#                 dim_feedforward=64,
#                 dropout=0.1
#             ),
#             num_layers=2
#         )
#         self.fc_before_trm = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size//4),
#             nn.Dropout(self.hidden_dropout_prob),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size//4, 1)
#         )

#         self.fc_after_trm = nn.Sequential(
#             nn.Linear(self.MAX_ITEM_LIST_LENGTH, aug_num* 2),
#             nn.Dropout(self.hidden_dropout_prob),
#             nn.Linear(aug_num* 2, aug_num)
#         )
#         self.cls_token = torch.rand(self.hidden_size).view(1,1,1,-1)
#         self.cls_token.requires_grad = True

#         self.apply(self._init_weights)

#     def forward(self, data, item_embeddings):
#         self.fc_before_trm.to(data.device)
#         self.policy_trm_encoder.to(data.device)
#         self.fc_after_trm.to(data.device)
        
#         # data: [B, aug_num, MAXLEN]
#         self.cls_token = self.cls_token.to(data.device)
#         # rp = self.cls_token.repeat(data.shape[0], data.shape[1], 1, 1)
#         emb = item_embeddings(data)
#         # emb: [B, aug_num, MAXLEN, hidden_size]
#         print(emb.shape)
#         out = self.fc_before_trm(self.hidden_trm_encoder(emb))
#         out = out.squeeze(-1)
#         # out: [B, aug_num, MAXLEN+1]

#         out = self.policy_trm_encoder(out)
#         # out = out.reshape(len(data), -1)
#         out = self.fc_after_trm(out)
#         out = out.squeeze(-1)
#         # out: [B, aug_num]

#         return out
    
#     def calculate_loss(self, output, y):
#         return self.loss_fn(output, y)

#     def forward_with_token(self, data, item_embeddings):
#         self.fc_before_trm.to(data.device)
#         self.policy_trm_encoder.to(data.device)
#         self.fc_after_trm.to(data.device)
        
#         # data: [B, aug_num, MAXLEN]
#         self.cls_token = self.cls_token.to(data.device)
#         rp = self.cls_token.repeat(data.shape[0], data.shape[1], 1, 1)
#         emb = item_embeddings(data)
#         # emb: [B, aug_num, MAXLEN, hidden_size]
#         emb = torch.cat([rp, emb], dim=2)
#         # emb: [B, aug_num, MAXLEN+1, hidden_size]

#         out = self.fc_before_trm(emb)
#         out = out.squeeze(-1)
#         # out: [B, aug_num, MAXLEN+1]

#         out = self.policy_trm_encoder(out)
#         out = out[:,:,0]
#         # out = out.reshape(len(data), -1)
#         out = self.fc_after_trm(out)
#         # out: [B, aug_num]

#         return out
    
#     def _init_weights(self, module):
#         """ Initialize the weights """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=1.0)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()
    
#     def _init_weights_biasver(self, module):
        
#         mean_bias = random.random()
#         std_bias = random.random()

#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0+mean_bias, std=1.0+std_bias)
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.fill_(0+mean_bias)
#             module.weight.data.fill_(1.0+mean_bias)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.fill_(0+mean_bias)


class PolicyChooser(nn.Module):
    def __init__(self, aug_num, config):
        super(PolicyChooser, self).__init__()

        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.policy_trm_num_layers = config['policy_trm_num_layers']
        self.MAX_ITEM_LIST_LENGTH = config['MAX_ITEM_LIST_LENGTH']
        self.loss_fn = nn.CrossEntropyLoss()
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
    
    def calculate_loss(self, output, y):
        return self.loss_fn(output, y)
    
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

