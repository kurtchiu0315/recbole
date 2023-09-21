# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch import nn
import faiss
from tqdm import tqdm

from recbole.model.sequential_recommender.sasrec import SASRec


class ICLRec(SASRec):
    
    def __init__(self, config, dataset):
        super(ICLRec, self).__init__(config, dataset)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.tau = config['tau']
        self.cl_lambda = config['cl_lambda']
        self.cl_loss_type = config['cl_loss_type']
        self.similarity_type = config['similarity_type']
        
        # for ICLRec
        self.contrast_type = config['contrast_type']
        self.num_intent_clusters = config['num_intent_clusters']
        self.instance_weight = config['instance_weight']
        self.intent_weight = config['intent_weight']
        
        # for Kmeans clustering
        self.seed = config['seed']
        self.gpu_id = config['gpu_id']
        self.device = config['device']
        self.clusters = KMeans(
                            num_cluster=self.num_intent_clusters,
                            seed=self.seed,
                            hidden_size=self.hidden_size,
                            gpu_id=self.gpu_id,
                            device=self.device,
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
    

    def mask_correlated_samples(self, batch_size):
        N = batch_size
        mask = torch.ones((2 * N, 2 * N)).bool()
        mask = mask.fill_diagonal_(0)
        mask *= ~ torch.diagflat(torch.ones(N), offset=N).bool()
        mask *= ~ torch.diagflat(torch.ones(N), offset=-N).bool()
        return mask
    
    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output_seq = trm_output[-1]
        output = self.gather_indexes(output_seq, item_seq_len - 1)
        
        return output, output_seq  # [B H]

    def calculate_loss(self, interaction):
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, output_seq = self.forward(item_seq, item_seq_len)
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

        seq_output1, _ = self.forward(aug_item_seq1, aug_len1)
        seq_output2, _ = self.forward(aug_item_seq2, aug_len2)
            
        if self.contrast_type in ['InstanceCL', 'Hybrid']:
            
            logits, labels = self.info_nce(seq_output1, seq_output2)
            cl_loss = self.cl_loss_fct(logits, labels)
            losses.append(cl_loss * self.instance_weight)
            
        if self.contrast_type in ['IntentCL', 'Hybrid']:
            
            output_seq = torch.mean(output_seq, dim = 1, keepdim=False)
            output_seq = output_seq.cpu().detach().numpy()
            self.clusters.train(output_seq)
            
            mean_pcl_loss = 0
        
            intent_id, seq2intent = self.clusters.query(output_seq)
            logits1, labels1 = self.info_nce(seq_output1, seq2intent)
            logits2, labels2 = self.info_nce(seq_output2, seq2intent)
            mean_pcl_loss += self.cl_loss_fct(logits1, labels1)
            mean_pcl_loss += self.cl_loss_fct(logits2, labels2)

            mean_pcl_loss /= 2
            
            losses.append(mean_pcl_loss * self.intent_weight)
        
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
        seq_output, _ = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size,       # dimension of the vectors
                                self.num_cluster)  # number of centroids
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] >= self.num_cluster:
            self.clus.train(x, self.index)
        
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]    
