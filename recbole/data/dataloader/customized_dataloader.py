# @Time   : 2021/12/11
# @Author : Ray Wu
# @Email  : ray7102ray7102@gmail.com

"""
recbole.data.dataloader.customized_dataloader
################################################
"""

import numpy as np
import torch
import random
import copy

from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize
from scipy.special import softmax
from itertools import combinations
from recbole.data.dataloader.general_dataloader import TrainDataLoader
from recbole.data.interaction import Interaction



class CLTrainDataLoader(TrainDataLoader):
    """:class:`CLTrainDataLoader` is a dataloader for Contrastive Learning training.
    It can generate negative interaction when :attr:`training_neg_sample_num` is not zero.
    For the result of every batch, we permit that every positive interaction and its negative interaction
    must be in the same batch.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        self.iid_field = dataset.iid_field
        self.uid_field = dataset.uid_field
        self.iid_list_field = getattr(dataset, f'{self.iid_field}_list_field')
        self.item_list_length_field = dataset.item_list_length_field

    def _next_batch_data(self):
        cur_data = self._contrastive_learning_augmentation(self.dataset[self.pr:self.pr + self.step])
        self.pr += self.step
        return cur_data

    def _contrastive_learning_augmentation(self):
        raise NotImplementedError('Method _contrastive_learning_augmentation should be implemented.')

class CL4RecTrainDataLoader(CLTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        if config['model'] == 'ContraRec':
            augmentation_type = ['mask', 'reorder', 'random']
        else:
            augmentation_type = ['crop', 'mask', 'reorder', 'random']
            
        self.augmentation_table = {aug_type: idx for idx, aug_type in enumerate(augmentation_type)}
        
        self.aug_type1 = config['aug_type1']
        self.aug_type2 = config['aug_type2'] if config['aug_type2'] else config['aug_type1']
        
        self.eta = config['eta'] if config['model'] != 'ContraRec' else None # for crop
        self.gamma = config['gamma']  # for mask
        self.beta = config['beta']  # for reorder
        self.model_name = config['model']
        
    
    def _contrastive_learning_augmentation(self, cur_data):
        
        sequences = cur_data[self.iid_list_field]
        lengths = cur_data[self.item_list_length_field]
        aug_seq1, aug_len1 = self._augmentation(sequences, lengths, aug_type=self.aug_type1)
        aug_seq2, aug_len2 = self._augmentation(sequences, lengths, aug_type=self.aug_type2)
        cur_data.update(Interaction({'aug1': aug_seq1, 'aug_len1': aug_len1,
                                     'aug2': aug_seq2, 'aug_len2': aug_len2}))
        return cur_data

    def _augmentation(self, sequences, lengths, targets=None, aug_type='random'):
        
        aug_idx = self.augmentation_table[aug_type]
        aug_sequences = torch.zeros_like(sequences)
        aug_lengths = torch.zeros_like(lengths)
        def crop(seq, length):
            new_seq = torch.zeros_like(seq)
            new_seq_length = max(1, int(length * self.eta))
            crop_start = random.randint(0, length - new_seq_length)
            new_seq[:new_seq_length] = seq[crop_start:crop_start + new_seq_length]
            return new_seq, new_seq_length
        
        def mask(seq, length):
            num_mask = int(length * self.gamma)
            mask_index = random.sample(range(length), k=num_mask)
            seq[mask_index] = self.dataset.item_num  # token 0 has been used for semantic masking
            return seq, length
        
        def reorder(seq, length):
            num_reorder = int(length * self.beta)
            reorder_start = random.randint(0, length - num_reorder)
            shuffle_index = torch.randperm(num_reorder) + reorder_start
            seq[reorder_start:reorder_start + num_reorder] = seq[shuffle_index]
            return seq, length

        if self.model_name == 'ContraRec':
            aug_func = [mask, reorder]
        else:
            aug_func = [crop, mask, reorder]
        
        for i, (seq, length) in enumerate(zip(sequences, lengths)):
            
            if aug_type == 'random':
                aug_idx = random.randrange(len(aug_func))
            
            aug_sequences[i][:], aug_lengths[i] = aug_func[aug_idx](seq.clone(), length)

        return aug_sequences, aug_lengths
    
class SRMATrainDataLoader(CLTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        augmentation_type = ['crop', 'mask', 'reorder', 'identity', 'random']
        self.augmentation_table = {aug_type: idx for idx, aug_type in enumerate(augmentation_type)}
        
        self.aug_type1 = config['aug_type1']
        self.aug_type2 = config['aug_type2'] if config['aug_type2'] else config['aug_type1']
        self.eta = config['eta']  # for crop
        self.gamma = config['gamma']  # for mask
        self.beta = config['beta']  # for reorder
    
    def _contrastive_learning_augmentation(self, cur_data):
        sequences = cur_data[self.iid_list_field]
        lengths = cur_data[self.item_list_length_field]
        aug_seq1, aug_len1 = self._augmentation(sequences, lengths, self.aug_type1)
        aug_seq2, aug_len2 = self._augmentation(sequences, lengths, self.aug_type2)
        cur_data.update(Interaction({'aug1': aug_seq1, 'aug_len1': aug_len1,
                                     'aug2': aug_seq2, 'aug_len2': aug_len2}))
        return cur_data

    def _augmentation(self, sequences, lengths, targets=None, aug_type='random'):
        aug_idx = self.augmentation_table[aug_type]
        aug_sequences = torch.zeros_like(sequences)
        aug_lengths = torch.zeros_like(lengths)

        def crop(seq, length):
            new_seq = torch.zeros_like(seq)
            new_seq_length = max(1, int(length * self.eta))
            crop_start = random.randint(0, length - new_seq_length)
            new_seq[:new_seq_length] = seq[crop_start:crop_start + new_seq_length]
            return new_seq, new_seq_length
        
        def mask(seq, length):
            num_mask = int(length * self.gamma)
            mask_index = random.sample(range(length), k=num_mask)
            seq[mask_index] = self.dataset.item_num  # token 0 has been used for semantic masking
            return seq, length
        
        def reorder(seq, length):
            num_reorder = int(length * self.beta)
            reorder_start = random.randint(0, length - num_reorder)
            shuffle_index = torch.randperm(num_reorder) + reorder_start
            seq[reorder_start:reorder_start + num_reorder] = seq[shuffle_index]
            return seq, length
        
        def identity(seq, length):
            copied_seq = copy.deepcopy(seq)
            return copied_seq, length

        aug_func = [crop, mask, reorder, identity]
        for i, (seq, length) in enumerate(zip(sequences, lengths)):
            if aug_type == 'random':
                aug_idx = random.randrange(len(aug_func))
            
            aug_sequences[i][:], aug_lengths[i] = aug_func[aug_idx](seq.clone(), length)

        return aug_sequences, aug_lengths


class CoSeRecTrainDataLoader(CL4RecTrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        self.inter_num = len(dataset)
        self.item_num = self.dataset.item_num
        self.max_item_list_len = self.dataset.max_item_list_len

        augmentation_type = ['insert', 'substitute', 'crop', 'mask', 'reorder', 'random', 'random_all']
        self.augmentation_table = {aug_type: idx for idx, aug_type in enumerate(augmentation_type)}
        self.online_item_cor_mat = None
        self.offline_item_cor_mat = None

        self.offline_similarity_type = config['cl_aug_offline_similarity_type']
        self.topk = config['cl_aug_similarity_topk']
        self.insert_rate = config['insert_rate']
        self.substitute_rate = config['substitute_rate']
        self._init_augmentation()

    def _init_augmentation(self):
        uids = self.dataset.inter_feat[self.uid_field].numpy()
        iids = self.dataset.inter_feat[self.iid_field].numpy()
        item_list = self.dataset[self.iid_list_field].numpy()
        item_length = self.dataset[self.item_list_length_field].numpy()
        last_item_index = item_length - 1
        prev_iids = item_list[np.arange(len(self.dataset)), last_item_index]
        
        last_uid = None
        iids_inter_list = []
        for uid, iid, prev_iid in zip(uids, iids, prev_iids):
            if last_uid != uid:
                last_uid = uid
                iids_inter_list.append([prev_iid])
            iids_inter_list[-1].append(iid)

        shape = (self.item_num, self.item_num)
        item_inter_matrix = coo_matrix(shape)
        for iids_inter in iids_inter_list:
            row, col = zip(*combinations(iids_inter, 2))
            if self.offline_similarity_type == 'itemcf_iuf':
                data = np.ones_like(row) / np.log(1 + len(iids_inter))
            else:  # itemcf
                data = np.ones_like(row)
            item_inter_matrix += coo_matrix((data, (row, col)), shape=shape)
        item_inter_matrix += item_inter_matrix.T
        # norm_scalar = np.sqrt(np.outer(item_inter_matrix.getnnz(axis=1),
        #                                item_inter_matrix.getnnz(axis=0)))
        # item_inter_matrix2 = item_inter_matrix.multiply(np.reciprocal(norm_scalar))
        norm_scalar1 = np.reciprocal(np.sqrt(item_inter_matrix.getnnz(axis=1)) + 1e-8).reshape(-1, 1)
        norm_scalar2 = np.reciprocal(np.sqrt(item_inter_matrix.getnnz(axis=0)) + 1e-8)
        self.offline_item_cor_mat = item_inter_matrix.multiply(norm_scalar1).multiply(norm_scalar2)
        self._update_most_similar_items()

    def _update_most_similar_items(self):
        if self.online_item_cor_mat != None:
            cor_mat = self.offline_item_cor_mat.maximum(self.online_item_cor_mat)
        else:
            cor_mat = self.offline_item_cor_mat.copy()
        cor_mat = cor_mat.tolil()

        most_similar_items = np.zeros((self.item_num, self.topk), dtype=int)
        for i, (data, row) in enumerate(zip(cor_mat.data, cor_mat.rows)):
            data, row = np.array(data), np.array(row)
            topk_indices = np.argsort(-data)[:self.topk]
            most_similar_items[i, :len(data)] = row[topk_indices]  # using len(data) because sometimes # of data is less than topk
        self.most_similar_items = most_similar_items
        
    def update_embedding_matrix(self, item_embedding):
        cor_mat = np.matmul(item_embedding, item_embedding.T)
        _min, _max = np.min(cor_mat), np.max(cor_mat)
        cor_mat = (cor_mat - _min) / (_max - _min)
        idx = (-cor_mat).argpartition(self.topk, axis=1)[:, :self.topk]
        row = np.repeat(np.arange(self.item_num), self.topk)
        col = idx.flatten()
        data = np.take_along_axis(cor_mat, idx, axis=1).flatten()
        self.online_item_cor_mat = coo_matrix((data, (row, col)),
                                              shape=(self.item_num, self.item_num))
        self._update_most_similar_items()

    def _augmentation(self, sequences, lengths, targets=None, aug_type='random'):
        aug_idx = self.augmentation_table[aug_type]
        aug_sequences = torch.zeros_like(sequences)
        aug_lengths = torch.zeros_like(lengths)

        def insert(seq, length):  # it has duplication in seq
            new_seq = torch.zeros_like(seq)
            num_insert = max(int(length * self.insert_rate), 1)
            insert_index = random.sample(range(length), k=num_insert)
            ref_items = [seq[i] for i in insert_index]
            insert_items = self.most_similar_items[ref_items, 0]
            seq = seq[:length].unsqueeze(1).tolist()
            for i, item in zip(insert_index, insert_items):
                seq[i].insert(0, item)
            seq = np.concatenate(seq, axis=0)[-self.max_item_list_len:]
            new_seq_length = len(seq)
            new_seq[:new_seq_length] = torch.from_numpy(seq)
            return new_seq, new_seq_length
        
        def substitute(seq, length):
            num_substitute = max(int(length * self.substitute_rate), 1)
            substitute_index = random.sample(range(length), k=num_substitute)
            ref_items = [seq[i] for i in substitute_index]
            substitute_items = self.most_similar_items[ref_items, 0]
            seq[substitute_index] = torch.from_numpy(substitute_items)
            return seq, length
        
        def crop(seq, length):  # copy from cl4rec
            new_seq = torch.zeros_like(seq)
            new_seq_length = max(1, int(length * self.eta))
            crop_start = random.randint(0, length - new_seq_length)
            new_seq[:new_seq_length] = seq[crop_start:crop_start + new_seq_length]
            return new_seq, new_seq_length
        
        def mask(seq, length):  # copy from cl4rec
            num_mask = int(length * self.gamma)
            mask_index = random.sample(range(length), k=num_mask)
            seq[mask_index] = self.dataset.item_num  # token 0 has been used for semantic masking
            return seq, length
        
        def reorder(seq, length):  # copy from cl4rec
            num_reorder = int(length * self.beta)
            reorder_start = random.randint(0, length - num_reorder)
            shuffle_index = torch.randperm(num_reorder) + reorder_start
            seq[reorder_start:reorder_start + num_reorder] = seq[shuffle_index]
            return seq, length
        
        aug_func = [insert, substitute, crop, mask, reorder]
        if 'random' in aug_type:
            if aug_type == 'random_all':  # [insert, substitute, crop, mask, reorder]
                aug_idxs = random.choices(range(len(aug_func)), k=self.inter_num)
            elif aug_type == 'random':  # [insert, substitute]
                aug_idxs = random.choices(range(2), k=self.inter_num)
        else:
            aug_idx = self.augmentation_table[aug_type]
            aug_idxs = [aug_idx] * self.inter_num
        
        for i, (aug_idx, seq, length) in enumerate(zip(aug_idxs, sequences, lengths)):
            aug_sequences[i][:], aug_lengths[i] = aug_func[aug_idx](seq.clone(), length)

        return aug_sequences, aug_lengths


class DuoRecTrainDataLoader(CLTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        self._init_augmentation()


    def _contrastive_learning_augmentation(self, cur_data):
        aug_seq, aug_len = self._augmentation(cur_data)
        cur_data.update(Interaction({'aug': aug_seq, 'aug_len': aug_len}))
        return cur_data

    def _init_augmentation(self):
        target_item_list = self.dataset.inter_feat[self.iid_field].numpy()
        index = {}
        for i, key in enumerate(target_item_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        self.same_target_index = index


    def _shuffle(self):
        super()._shuffle()
        self._init_augmentation()

    def _augmentation(self, cur_data):
        targets = cur_data[self.iid_field].numpy()
        select_index = [np.random.choice(self.same_target_index[target]) for target in targets]
        aug_sequences = self.dataset[self.iid_list_field][select_index]
        aug_lengths = self.dataset[self.item_list_length_field][select_index]
        assert (targets == self.dataset[self.iid_field][select_index].numpy()).all()
        return aug_sequences, aug_lengths



class MyRec3TrainDataLoader(DuoRecTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        self.ucl_type = config['ucl_type']
        ucl_dataloader_table = {
            'cl4rec': CL4RecTrainDataLoader,
            'coserec': CoSeRecTrainDataLoader,
            'myrec': MyRec2TrainDataLoader,
        }
        ucl_aug_type_table = {
            'cl4rec': 'random',
            'coserec': 'random_all',
            'myrec': 'random_all',
        }
        self.ucl_dataloader = ucl_dataloader_table[self.ucl_type](config, dataset, sampler, shuffle)
        self._ucl_aug_type = ucl_aug_type_table[self.ucl_type]

    def _contrastive_learning_augmentation(self, cur_data):
        sequences = cur_data[self.iid_list_field]
        lengths = cur_data[self.item_list_length_field]
        targets = cur_data[self.iid_field]
        aug_seq0, aug_len0 = self._supervised_augmentation(cur_data)
        aug_seq1, aug_len1 = self._unsupervised_augmentation(aug_seq0, aug_len0, targets)
        aug_seq2, aug_len2 = self._unsupervised_augmentation(sequences, lengths, targets)
        cur_data.update(Interaction({'aug0': aug_seq0, 'aug_len0': aug_len0, # supervised seq.
                                     'aug1': aug_seq1, 'aug_len1': aug_len1, # supervised seq. w/ augmentation
                                     'aug2': aug_seq2, 'aug_len2': aug_len2})) # original seq. w/ augmentation
        return cur_data

    def _supervised_augmentation(self, cur_data):
        return super()._augmentation(cur_data)
    
    def _unsupervised_augmentation(self, sequences, lengths, targets=None):
        return self.ucl_dataloader._augmentation(sequences,
                                                 lengths,
                                                 targets=targets,
                                                 aug_type=self._ucl_aug_type)   


class MyRecTrainDataLoader(CLTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        self.inter_num = len(dataset)

        augmentation_type = ['crop', 'drop', 'substitute', 'insert', 'reorder', 'random']
        self.augmentation_table = {aug_type: idx for idx, aug_type in enumerate(augmentation_type)}
        
        self.start_id = self.dataset.item_num
        self.end_id = self.dataset.item_num + 1
        self.entity_num = self.dataset.item_num + 2

        self.aug_type1 = config['aug_type1']
        self.aug_type2 = config['aug_type2'] if config['aug_type2'] else config['aug_type1']

        self.gamma = config['gamma']

        self._init_augmentation()

    def _init_augmentation(self):
        uids = self.dataset.inter_feat[self.uid_field].numpy()
        iids = self.dataset.inter_feat[self.iid_field].numpy()

        item_list = self.dataset[self.iid_list_field].numpy()
        item_length = self.dataset[self.item_list_length_field].numpy()
        last_item_index = item_length - 1
        prev_iids = item_list[np.arange(len(self.dataset)), last_item_index]
        start_iids = []
        end_iids = []

        last_uid = None
        for i, (uid, iid, prev_iid) in enumerate(zip(uids, iids, prev_iids)):
            if last_uid != uid:
                last_uid = uid
                start_iids.append(prev_iid)
                end_iids.append(iid)
            end_iids[-1] = iid

        assert self.dataset.user_num >= len(start_iids)
        assert self.dataset.user_num >= len(end_iids)

        total_num = len(iids) + len(start_iids) + len(end_iids)
        row = np.zeros(total_num)
        col = np.zeros(total_num)
        data = np.ones(total_num)

        row[:len(iids)] = prev_iids
        col[:len(iids)] = iids

        row[len(iids):-len(end_iids)] = np.full(len(start_iids), self.start_id)
        col[len(iids):-len(end_iids)] = start_iids

        row[-len(end_iids):] = end_iids
        col[-len(end_iids):] = np.full(len(end_iids), self.end_id)

        self.adj_matrix = coo_matrix((data, (row, col)),
                                     shape=(self.entity_num, self.entity_num),
                                     dtype=np.float16).tocsr()
        self.adj_matrix = normalize(self.adj_matrix, norm='l1', axis=1)
        self.adj_matrix2 = self.adj_matrix.dot(self.adj_matrix) + self.adj_matrix

    def _contrastive_learning_augmentation(self, cur_data):
        sequences = cur_data[self.iid_list_field]
        lengths = cur_data[self.item_list_length_field]
        targets = cur_data[self.iid_field]
        aug_seq1, aug_len1 = self._augmentation(sequences, lengths, targets, self.aug_type1)
        aug_seq2, aug_len2 = self._augmentation(sequences, lengths, targets, self.aug_type2)
        cur_data.update(Interaction({'aug1': aug_seq1, 'aug_len1': aug_len1,
                                     'aug2': aug_seq2, 'aug_len2': aug_len2}))
        return cur_data
    
    def _augmentation(self, sequences, lengths, targets, aug_type='random', eps=1e-8):
        aug_idx = self.augmentation_table[aug_type]
        aug_sequences = torch.zeros_like(sequences)
        aug_lengths = torch.zeros_like(lengths)
        
        def crop(seq, length, target):
            if length <= 2:
                return seq, length

            new_seq = torch.zeros_like(seq)
            seq = seq[:length]
            start_p = self.adj_matrix[self.start_id, seq].toarray().squeeze(0) + eps
            crop_start = random.choices(range(length), weights=start_p)[0]

            new_seq_length = length - crop_start
            new_seq[:new_seq_length] = seq[crop_start:]
            return new_seq, new_seq_length
        
        def drop(seq, length, target):
            if length <= 2:
                return seq, length
            new_seq = seq.clone()
            seq  = torch.cat((torch.tensor([self.start_id]), seq[:length], target.unsqueeze(0)))
            
            seq_head = seq[:-2].tolist()
            seq_tail = seq[2:].tolist()
            prob_ac_hop2 = self.adj_matrix2[seq_head, seq_tail].A1 + eps
            prob_ac = self.adj_matrix[seq_head, seq_tail].A1 + eps
            drop_p = prob_ac / prob_ac_hop2
            drop_idx = random.choices(range(length), weights=drop_p)[0]

            new_seq = torch.cat((new_seq[:drop_idx], new_seq[drop_idx + 1:], torch.zeros(1)))
            new_length = length - 1
            return new_seq, new_length
        
        def substitute(seq, length, target):
            if length <= 2:
                return seq, length
            new_seq = seq.clone()
            seq  = torch.cat((torch.tensor([self.start_id]), seq[:length], target.unsqueeze(0)))
            
            seq_head = seq[:-2].tolist()
            seq_tail = seq[2:].tolist()
            prob_ac_hop2 = self.adj_matrix2[seq_head, seq_tail].A1 + eps
            prob_ac = self.adj_matrix[seq_head, seq_tail].A1 + eps
            substitute_p = 1 - prob_ac / prob_ac_hop2
            substitute_idx = random.choices(range(length), weights=substitute_p)[0]
            a, c = seq_head[substitute_idx], seq_tail[substitute_idx]
            prob_abc = self.adj_matrix[a].multiply(self.adj_matrix.transpose()[c])
            if prob_abc.count_nonzero() == 0:
                return new_seq, length

            select_item = random.choices(prob_abc.indices, weights=prob_abc.data)[0]

            if select_item in [self.start_id, self.end_id]:
                return new_seq, length

            new_seq[substitute_idx] = select_item
            return new_seq, length
        
        def insert(seq, length, target):
            if length == seq.size(0):
                seq, length = drop(seq, length, target)

            new_seq = seq.clone()
            seq  = torch.cat((torch.tensor([self.start_id]), seq[:length], target.unsqueeze(0)))

            seq_head = seq[:-1].tolist()
            seq_tail = seq[1:].tolist()
            prob_ac_hop2 = self.adj_matrix2[seq_head, seq_tail].A1 + eps
            prob_ac = self.adj_matrix[seq_head, seq_tail].A1 + eps
            insert_p = 1 - prob_ac / prob_ac_hop2
            insert_idx = random.choices(range(length + 1), weights=insert_p)[0]
            a, c = seq_head[insert_idx], seq_tail[insert_idx]
            prob_abc = self.adj_matrix[a].multiply(self.adj_matrix.transpose()[c])
            if prob_abc.count_nonzero() == 0:
                return new_seq, length
                
            select_item = random.choices(prob_abc.indices, weights=prob_abc.data)[0]

            if select_item in [self.start_id, self.end_id]:
                return new_seq, length

            new_seq = torch.cat((new_seq[:insert_idx], torch.tensor([select_item]), new_seq[insert_idx:-1]))
            new_length = length + 1
            return new_seq, new_length
            
        def reorder(seq, length, target):
            # a -> b -> c -> d
            # a -> c -> b -> d
            # p(c|b) = p(b|c)
            if length <= 2:
                return seq, length

            new_seq = seq.clone()
            seq = seq[:length]
            
            seq_head = seq[:-1].tolist()
            seq_tail = seq[1:].tolist()
            prob_ab = self.adj_matrix[seq_head, seq_tail].A1
            prob_ba = self.adj_matrix[seq_tail, seq_head].A1

            swap_p = -np.absolute(prob_ab - prob_ba)
            swap_idx = random.choices(range(length - 1), weights=swap_p)[0]
            
            new_seq[[swap_idx, swap_idx + 1]] = new_seq[[swap_idx + 1, swap_idx]]
            return new_seq, length
        
        aug_func = [crop, drop, substitute, insert, reorder]
        if aug_type == 'random':
            aug_idxs = random.choices(range(len(aug_func)), k=self.inter_num)
        else:
            aug_idxs = [aug_idx] * self.inter_num

        for i, (aug_idx, target, seq, length) in enumerate(zip(aug_idxs, targets, sequences, lengths)):
            aug_sequences[i][:], aug_lengths[i] = aug_func[aug_idx](seq.clone(), length, target)

        return aug_sequences, aug_lengths


class MyRec2TrainDataLoader(MyRecTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        augmentation_type = ['crop', 'insert', 'reorder', 'drop', 'substitute', 'random']
        self.augmentation_table = {aug_type: idx for idx, aug_type in enumerate(augmentation_type)}

        self.tau = config['aug_tau'] if config['aug_tau'] else 1
    
    def _augmentation(self, sequences, lengths, targets, aug_type='random', eps=1e-8):
        aug_sequences = torch.zeros_like(sequences)
        aug_lengths = torch.zeros_like(lengths)
        
        def crop(seq, length, target):
            if length <= 2:
                return seq, length

            new_seq = torch.zeros_like(seq)
            seq = seq[:length]
            start_p = self.adj_matrix[self.start_id, seq].toarray().squeeze(0)
            start_p = softmax(start_p / self.tau)
            crop_start = random.choices(range(length), weights=start_p)[0]

            new_seq_length = length - crop_start
            new_seq[:new_seq_length] = seq[crop_start:]
            return new_seq, new_seq_length
        
        def drop(seq, length, target):
            if length <= 2:
                return seq, length
            new_seq = seq.clone()
            seq  = torch.cat((torch.tensor([self.start_id]), seq[:length], target.unsqueeze(0)))
            
            seq_head = seq[:-2].tolist()
            seq_tail = seq[2:].tolist()
            prob_ac_hop2 = self.adj_matrix2[seq_head, seq_tail].A1
            prob_ac = self.adj_matrix[seq_head, seq_tail].A1
            
            prob_ac_hop2 = np.log(prob_ac_hop2 + eps)
            prob_ac = np.log(prob_ac + eps)
            drop_p = softmax((prob_ac - prob_ac_hop2) / self.tau)
            drop_idx = random.choices(range(length), weights=drop_p)[0]

            new_seq = torch.cat((new_seq[:drop_idx], new_seq[drop_idx + 1:], torch.zeros(1)))
            new_length = length - 1
            return new_seq, new_length
        
        def substitute(seq, length, target, eps=1e-8):
            if length <= 2:
                return seq, length
            new_seq = seq.clone()
            seq  = torch.cat((torch.tensor([self.start_id]), seq[:length], target.unsqueeze(0)))
            
            seq_head = seq[:-2].tolist()
            seq_tail = seq[2:].tolist()
            prob_ac_hop2 = self.adj_matrix2[seq_head, seq_tail].A1
            prob_ac = self.adj_matrix[seq_head, seq_tail].A1
            prob_ac_hop2 = np.log(prob_ac_hop2 + eps)
            prob_ac = np.log(prob_ac + eps)

            substitute_p = softmax((prob_ac_hop2 - prob_ac) / self.tau)
            substitute_idx = random.choices(range(length), weights=substitute_p)[0]
            a, c = seq_head[substitute_idx], seq_tail[substitute_idx]
            prob_abc = self.adj_matrix[a].multiply(self.adj_matrix.transpose()[c])
            if prob_abc.count_nonzero() == 0:
                return new_seq, length

            select_p = softmax(prob_abc.data / self.tau)
            select_item = random.choices(prob_abc.indices, weights=select_p)[0]

            if select_item in [self.start_id, self.end_id]:
                return new_seq, length

            new_seq[substitute_idx] = select_item
            return new_seq, length
        
        def insert(seq, length, target):
            if length == seq.size(0):
                seq, length = drop(seq, length, target)

            new_seq = seq.clone()
            seq  = torch.cat((torch.tensor([self.start_id]), seq[:length], target.unsqueeze(0)))

            seq_head = seq[:-1].tolist()
            seq_tail = seq[1:].tolist()
            prob_ac_hop2 = self.adj_matrix2[seq_head, seq_tail].A1
            prob_ac = self.adj_matrix[seq_head, seq_tail].A1

            prob_ac_hop2 = np.log(prob_ac_hop2 + eps)
            prob_ac = np.log(prob_ac + eps)
            
            insert_p = softmax((prob_ac_hop2 - prob_ac) / self.tau)
            insert_idx = random.choices(range(length + 1), weights=insert_p)[0]
            a, c = seq_head[insert_idx], seq_tail[insert_idx]
            prob_abc = self.adj_matrix[a].multiply(self.adj_matrix.transpose()[c])
            if prob_abc.count_nonzero() == 0:
                return new_seq, length
            
            select_p = softmax(prob_abc.data / self.tau)
            select_item = random.choices(prob_abc.indices, weights=select_p)[0]

            if select_item in [self.start_id, self.end_id]:
                return new_seq, length

            new_seq = torch.cat((new_seq[:insert_idx], torch.tensor([select_item]), new_seq[insert_idx:-1]))
            new_length = length + 1
            return new_seq, new_length
            
        def reorder(seq, length, target):
            # a -> b -> c -> d
            # a -> c -> b -> d
            # p(c|b) = p(b|c)
            if length <= 2:
                return seq, length

            new_seq = seq.clone()
            seq  = torch.cat((torch.tensor([self.start_id]), seq[:length], target.unsqueeze(0)))
            
            seq_a = seq[:-3].tolist()
            seq_b = seq[1:-2].tolist()
            seq_c = seq[2:-1].tolist()
            seq_d = seq[3:].tolist()

            prob_ab = self.adj_matrix[seq_a, seq_b].A1
            prob_ac = self.adj_matrix[seq_a, seq_c].A1
            prob_bc = self.adj_matrix[seq_b, seq_c].A1
            prob_cb = self.adj_matrix[seq_c, seq_b].A1
            prob_bd = self.adj_matrix[seq_b, seq_d].A1
            prob_cd = self.adj_matrix[seq_c, seq_d].A1

            prob_ab = np.log(prob_ab + eps)
            prob_ac = np.log(prob_ac + eps)
            prob_bc = np.log(prob_bc + eps)
            prob_cb = np.log(prob_cb + eps)
            prob_bd = np.log(prob_bd + eps)
            prob_cd = np.log(prob_cd + eps)

            swap_p = (prob_ac + prob_cb + prob_bd) - (prob_ab + prob_bc + prob_cd)
            swap_p = softmax(swap_p / self.tau)
            swap_idx = random.choices(range(length - 1), weights=swap_p)[0]
            
            new_seq[[swap_idx, swap_idx + 1]] = new_seq[[swap_idx + 1, swap_idx]]
            return new_seq, length
        
        aug_func = [crop, insert, reorder, drop, substitute]
        # aug_func = [crop, insert, reorder]
        if 'random' in aug_type:
            if aug_type == 'random_all':  # [crop, insert, reorder, drop, substitute]
                aug_idxs = random.choices(range(len(aug_func)), k=self.inter_num)
            elif aug_type == 'random':  # [crop, insert, reorder]
                aug_idxs = random.choices(range(3), k=self.inter_num)
        else:
            aug_idx = self.augmentation_table[aug_type]
            aug_idxs = [aug_idx] * self.inter_num

        for i, (aug_idx, target, seq, length) in enumerate(zip(aug_idxs, targets, sequences, lengths)):
            aug_sequences[i][:], aug_lengths[i] = aug_func[aug_idx](seq.clone(), length, target)

        return aug_sequences, aug_lengths


class DSSRecTrainDataLoader(TrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        self.iid_field = dataset.iid_field
        self.uid_field = dataset.uid_field
        self.iid_list_field = getattr(dataset, f'{self.iid_field}_list_field')
        self.item_list_length_field = dataset.item_list_length_field

    def _next_batch_data(self):
        cur_data = self.dataset[self.pr:self.pr + self.step]
        cur_data = self._get_input_label_subsequences(cur_data)
        self.pr += self.step
        return cur_data
    
    def _shuffle(self):
        super()._shuffle()
   

    def _get_input_label_subsequences(self, cur_data):
        sequences = cur_data[self.iid_list_field].numpy()
        padding_len = sequences.shape[1]
        lengths = cur_data[self.item_list_length_field].numpy()

        inp_subseqs = []
        label_subseqs = []
        next_items = []
        for i in range(sequences.shape[0]):
            if lengths[i] == 2:
                t = 1
            else:
                if lengths[i] != 1:
                    t = torch.randint(1, lengths[i] - 1, (1,))
            
            if lengths[i] == 1:
                inp_subseq = sequences[i].tolist()
            else:
                inp_subseq = sequences[i][0:t].tolist()

            # label sub-sequence
            if lengths[i] == 1:
                label_subseq = sequences[i].tolist()
            else:
                label_subseq = sequences[i][t:lengths[i]].tolist()
                
            # next item
            if lengths[i] == 1:
                next_item = sequences[i][0].tolist()
            else:
                next_item = sequences[i][t].tolist()

            # padding
         
            inp_subseq = ([0] * max(0, padding_len - len(inp_subseq)) + inp_subseq)[-padding_len:]
            label_subseq.reverse()
            label_subseq = ([0] * max(0, padding_len - len(label_subseq)) + label_subseq)[-padding_len:]

            inp_subseqs.append(inp_subseq)
            label_subseqs.append(label_subseq)
            next_items.append([next_item])
        
        cur_data.update(
            Interaction({
                'inp_subseqs': torch.tensor(inp_subseqs, dtype=torch.int64), 
                'label_subseqs': torch.tensor(label_subseqs, dtype=torch.int64),
                'next_items': torch.tensor(next_items, dtype=torch.int64)
            })
        )
        return cur_data


class EC4SRecTrainDataLoader(CLTrainDataLoader):
    
    def __init__(self, config, dataset, sampler, shuffle=False):
        
        super().__init__(config, dataset, sampler, shuffle=shuffle)

        augmentation_type = ['crop', 'mask', 'reorder', 'retrieval', 'random']
            
        self.augmentation_table = {aug_type: idx for idx, aug_type in enumerate(augmentation_type)}
        
        self.aug_type1 = config['aug_type1']
        self.aug_type2 = config['aug_type2']
        
        self.eta = config['eta'] # for crop
        self.gamma = config['gamma']  # for mask
        self.beta = config['beta']  # for reorder
        self.model_name = config['model']
        self.hard_sample = config['hard_sample']
        self._prepare_retrieval_data()
        
    def _contrastive_learning_augmentation(self, cur_data):
        pass

    def _shuffle(self):
        super()._shuffle()
        self._prepare_retrieval_data()  

    def explanation_contrastive_learning_augmentation(self, cur_data, item_importance_scores):
        # importance_scores:
        ## shape == sequences.shape
        sequences = cur_data[self.iid_list_field]
        lengths = cur_data[self.item_list_length_field]
        targets = cur_data[self.iid_field].cpu().numpy()
        aug_seq1, aug_len1 = self._augmentation(sequences, lengths, item_importance_scores, targets, aug_type=self.aug_type1)
        aug_seq2, aug_len2 = self._augmentation(sequences, lengths, item_importance_scores, targets, aug_type=self.aug_type2)
   

        cur_data.update(Interaction({'aug1': aug_seq1, 'aug_len1': aug_len1,
                                     'aug2': aug_seq2, 'aug_len2': aug_len2}))
        return cur_data

    def _prepare_retrieval_data(self):
        target_item_list = self.dataset.inter_feat[self.iid_field].numpy()
        index = {}
        for i, key in enumerate(target_item_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        self.same_target_index = index

    def _next_batch_data(self):
        cur_data = self.dataset[self.pr:self.pr + self.step]
        self.pr += self.step
        return cur_data
    
    def _retrieval_data_selection(self, cur_sequence, cur_length, cur_target, item_importance_scores):

        retrieval_data_indices = self.same_target_index[cur_target]
        aug_sequences = self.dataset[self.iid_list_field][retrieval_data_indices]
        aug_lengths = self.dataset[self.item_list_length_field][retrieval_data_indices]

        
        intersection = torch.full_like(aug_sequences, False, dtype=bool)
        for item in (cur_sequence[:cur_length].cpu()):
            intersection = torch.logical_or(intersection ,(aug_sequences == item))
        # If aug_sequences[a,b] in cur_sequence, intersection[a,b] == True

        intersection_amount = intersection.sum(axis=1)
        union_amount = cur_length.cpu() + aug_lengths - intersection_amount # |union(A,B)| = |A| + |B| - |intersect(A,B)|
        ratio = intersection_amount/union_amount
        
        P = (item_importance_scores.cpu()[aug_sequences] * intersection).sum(axis=1) * ratio
        P = P.numpy()

        if np.nan in P or np.inf in P or sum(P) == 0:
            return cur_sequence, cur_length
        else:
            if self.hard_sample:
                prob = 1.1-P/P.sum()
                prob = prob/prob.sum()
            else:
                prob = P/P.sum()
            
            selected_ind = np.random.choice(range(len(retrieval_data_indices)), p = prob)
            assert (cur_target == (self.dataset[self.iid_field][retrieval_data_indices])[selected_ind].numpy())
            return aug_sequences[selected_ind].to(cur_sequence.device), aug_lengths[selected_ind].to(cur_sequence.device)


    def _get_seq_importance_scores(self, sequences, item_importance_scores):
        # item_importance_scores: [num_item]
        seq_importance_scores = item_importance_scores[sequences]
        return seq_importance_scores


    def _augmentation(self, sequences, lengths, item_importance_scores, targets, aug_type=None):
        if aug_type not in self.augmentation_table: # ValueError or NoneValue
            return sequences, lengths

        seq_importance_scores = self._get_seq_importance_scores(sequences, item_importance_scores)
        aug_idx = self.augmentation_table[aug_type]
        aug_sequences = torch.zeros_like(sequences)
        aug_lengths = torch.zeros_like(lengths)
        def crop(seq, length, seq_importance_score):
            crop_num = max(0, int(length * self.eta))
            new_seq_length = length - crop_num
            useful_score = seq_importance_score[0:length]

            if self.hard_sample:
                keep_index = useful_score.topk(new_seq_length, largest=False, sorted=False).indices.sort().values
            else:
                keep_index = useful_score.topk(new_seq_length, largest=True, sorted=False).indices.sort().values

            new_seq = seq[keep_index.cpu()]

            new_seq = torch.cat([new_seq, torch.zeros_like(seq)]) [ :seq.shape[0]]
            return new_seq, new_seq_length
        
        def mask(seq, length, seq_importance_score):
            useful_score = seq_importance_score[0:length] 
            num_mask = int(length * self.gamma)

            if self.hard_sample:
                mask_index = useful_score.topk(num_mask, largest=True, sorted=False).indices
            else:
                mask_index = useful_score.topk(num_mask, largest=False, sorted=False).indices
        
            seq[mask_index.cpu()] = self.dataset.item_num  # token 0 has been used for semantic masking
            return seq, length
        
        def reorder(seq, length, seq_importance_score):
            useful_score = seq_importance_score[0:length] 
            num_reorder = int(length * self.beta)
            if self.hard_sample:
                less_importance_index = useful_score.topk(num_reorder, largest=True, sorted=False).indices
            else:
                less_importance_index = useful_score.topk(num_reorder, largest=False, sorted=False).indices
            
            reorder_index = less_importance_index[torch.randperm(num_reorder)]

            seq[reorder_index.cpu()] = seq[less_importance_index.cpu()]
            return seq, length
        
        retrieval = self._retrieval_data_selection

        aug_func = [crop, mask, reorder, retrieval]

        for i, (seq, length, seq_importance_score, target) in enumerate(zip(sequences, lengths, seq_importance_scores, targets)):
            if aug_type == 'random':
                aug_idx = random.randrange(len(aug_func)-1)
                # random choice [crop, mask, reorder]
            
            if aug_idx in [0,1,2]: # not retrieval
                aug_sequences[i][:], aug_lengths[i] = aug_func[aug_idx](seq.clone(), length, seq_importance_score)
            else:
                aug_sequences[i][:], aug_lengths[i] = retrieval(seq.clone(), length, target, item_importance_scores)

        return aug_sequences, aug_lengths


class DADACLRecTrainDataLoader(CL4RecTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):
        shuffle=False
        self._aug_domain_config(config)
        super().__init__(config, dataset, sampler, shuffle=shuffle)
    
        self._construct_reversed_data()
        self._prepare_retrieval_data()
        self._prepare_reverse_retrieval_data()
        


    def _contrastive_learning_augmentation(self, cur_data):
        
        sequences = cur_data[self.iid_list_field]
        lengths = cur_data[self.item_list_length_field]
        # retrieval_aug, retrieval_aug_len = self._retrieval_augmentation(cur_data)
        # re_retrieval_aug, re_retrieval_aug_len = self._reverse_retrieval_augmentation(cur_data)

        # self.aug_names = ["crop_aug", "mask_aug", "reorder_aug", "su_aug", "re_su_aug"]
        aug_dicts = {}
        if "crop_aug" in self.aug_names:
            aug_seq1, aug_len1 = self._augmentation(sequences, lengths, aug_type='crop')
            cur_data.update(Interaction({'crop_aug': aug_seq1, 'crop_aug_len': aug_len1}))
        if "mask_aug" in self.aug_names:
            aug_seq2, aug_len2 = self._augmentation(sequences, lengths, aug_type='mask')
            cur_data.update(Interaction({'mask_aug': aug_seq2, 'mask_aug_len': aug_len2}))
        if "reorder_aug" in self.aug_names:
            aug_seq3, aug_len3 = self._augmentation(sequences, lengths, aug_type='reorder')
            cur_data.update(Interaction({'reorder_aug': aug_seq3, 'reorder_aug_len': aug_len3}))
        if "su_aug" in self.aug_names:
            retrieval_aug, retrieval_aug_len = self._retrieval_augmentation(cur_data)
            cur_data.update(Interaction({'su_aug':retrieval_aug, 'su_aug_len':retrieval_aug_len}))
        if "re_su_aug" in self.aug_names:
            re_retrieval_aug, re_retrieval_aug_len = self._reverse_retrieval_augmentation(cur_data)
            cur_data.update(Interaction({'re_su_aug': re_retrieval_aug, 're_su_aug_len':re_retrieval_aug_len}))  
        
        return cur_data
    
    def _augmentation(self, sequences, lengths, scales=None, aug_type='random'):
        if scales == None:
            scales = [None]*len(sequences)

        aug_idx = self.augmentation_table[aug_type]
        aug_sequences = torch.zeros_like(sequences)
        aug_lengths = torch.zeros_like(lengths)
        def crop(seq, length, eta=None):
            if eta == None:
                eta = self.eta
            new_seq = torch.zeros_like(seq)
            new_seq_length = max(1, int(length * eta))
            crop_start = random.randint(0, length - new_seq_length)
            new_seq[:new_seq_length] = seq[crop_start:crop_start + new_seq_length]
            return new_seq, new_seq_length
        
        def mask(seq, length, gamma=None):
            if gamma == None:
                gamma = self.gamma
            num_mask = int(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            seq[mask_index] = self.dataset.item_num  # token 0 has been used for semantic masking
            return seq, length
        
        def reorder(seq, length, beta=None):
            if beta == None:
                beta = self.beta
            num_reorder = int(length * beta)
            reorder_start = random.randint(0, length - num_reorder)
            shuffle_index = torch.randperm(num_reorder) + reorder_start
            seq[reorder_start:reorder_start + num_reorder] = seq[shuffle_index]
            return seq, length

        aug_func = [crop, mask, reorder]
        
        for i, (seq, length, scale) in enumerate(zip(sequences, lengths, scales)):
            
            if aug_type == 'random':
                aug_idx = random.randrange(len(aug_func))
            
            aug_sequences[i][:], aug_lengths[i] = aug_func[aug_idx](seq.clone(), length, scale)

        return aug_sequences, aug_lengths

    def _prepare_retrieval_data(self):
        target_item_list = self.dataset.inter_feat[self.iid_field].numpy()
        index = {}
        for i, key in enumerate(target_item_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        self.same_target_index = index

    def _prepare_reverse_retrieval_data(self):
        target_item_list = self.dataset.inter_feat[self.iid_field].numpy()
        rev_target_item_list = self.dataset.inter_feat["reverse_target"].numpy()
        index = {}
        for i, key in enumerate(target_item_list):
            if key not in index:
                if np.any(rev_target_item_list == key):
                    index[key] = (rev_target_item_list == key).nonzero()[0]
                else:
                    # print("!")
                    index[key] = [-1]
        self.same_target_rev_index = index
    
    def _retrieval_augmentation(self, cur_data):
        targets = cur_data[self.iid_field].numpy()
        select_index = [np.random.choice(self.same_target_index[target]) for target in targets]
        aug_sequences = self.dataset[self.iid_list_field][select_index]
        aug_lengths = self.dataset[self.item_list_length_field][select_index]
        assert (targets == self.dataset[self.iid_field][select_index].numpy()).all()
        return aug_sequences, aug_lengths

    def _reverse_retrieval_augmentation(self, cur_data):
        targets = cur_data[self.iid_field].numpy()
        apply_reverse = []
        apply_normal = []
        select_index = []
        notfoundcnt = 0
        for i, target in enumerate(targets):
            choice = np.random.choice(self.same_target_rev_index[target])
            if choice == -1:
                notfoundcnt += 1
                choice = np.random.choice(self.same_target_index[target])
                apply_normal += [i]
            else:
                apply_reverse += [i]
            select_index.append(choice)
        select_index = np.array(select_index)

        aug_sequences = self.dataset["reverse_seqs"][select_index]
        aug_sequences[apply_normal] = self.dataset[self.iid_list_field][ select_index[apply_normal] ]
        aug_lengths = self.dataset[self.item_list_length_field][select_index]
     

        check_targets = self.dataset["reverse_target"][select_index].numpy()
        check_targets[apply_normal] = self.dataset[self.iid_field][ select_index[apply_normal] ].numpy()
        assert (targets == check_targets).all()
     
        # print("NOT_FOUND_RATE:",  notfoundcnt, "/", len(targets))
        return aug_sequences, aug_lengths

    def _aug_domain_config(self, config):
        self.aug_domain = config['aug_domain'] 
        assert (self.aug_domain in ['CL4Rec', 'CL4Rec+SU', 'SU+RSU' ,'ALL', 'RSU', 'SU'])

        if self.aug_domain == 'CL4Rec':
            self.aug_names = ["crop_aug", "mask_aug", "reorder_aug"]
        elif self.aug_domain == 'CL4Rec+SU':
            self.aug_names = ["crop_aug", "mask_aug", "reorder_aug", "su_aug"]
        elif self.aug_domain == 'SU+RSU':
            self.aug_names = ["su_aug", "re_su_aug"]
        elif self.aug_domain == 'ALL':
            self.aug_names = ["crop_aug", "mask_aug", "reorder_aug", "su_aug", "re_su_aug"]
        elif self.aug_domain == 'RSU':
            self.aug_names = ["re_su_aug"]
        elif self.aug_domain == 'SU':
            self.aug_names = ["su_aug"]

        self.augnum = len(self.aug_names)

    def _shuffle(self):
        super()._shuffle()
        self._construct_reversed_data()
        self._prepare_retrieval_data()
        self._prepare_reverse_retrieval_data()

    def _construct_reversed_data(self):
        target_item_list = self.dataset.inter_feat[self.iid_field] # [10, 20, 30, 40, 50]
        seqs = self.dataset[self.iid_list_field] # [[1,2,3,4,0,0,0], 
                                                 #  [5,4,9,6,4,0,0],
                                                 #  [9,6,0,0,0,0,0],
                                                 #  [7,8,9,4,5,6,3],
                                                 #  [7,0,0,0,0,0,0]]
        
        rev_whole_seqs = torch.flip(torch.hstack([seqs, target_item_list.reshape(-1,1)]), [1])
        #          hstack        flip
        # [[1,2,3,4,0,0,0, 10],  --> [10, 0, 0, 0, 4, 3, 2, 1]
        #  [5,4,9,6,4,0,0, 20],  --> [20, 0, 0, 4, 6, 9, 4, 5]
        #  [9,6,0,0,0,0,0, 30],  --> [30, 0, 0, 0, 0, 0, 6  9]
        #  [7,8,9,4,5,6,3, 40],  --> [40, 3, 6, 5, 4, 9, 8, 7]
        #  [7,0,0,0,0,0,0, 50]]  --> [50, 0, 0, 0, 0, 0, 0, 7]

        # original_targets = rev_whole_seqs[:,0].clone()
        # reverse_targets = rev_whole_seqs[:,-1].clone()
        # for index, original_target in enumerate(original_targets):
        #     if original_target not in reverse_targets:
        #         rev_whole_seqs[index] = torch.flip(rev_whole_seqs[index], [0])
        #         assert np.all(rev_whole_seqs[index].numpy()[0:-1] == self.dataset[self.iid_list_field][index].numpy())


        self.dataset.inter_feat.interaction["reverse_target"] = rev_whole_seqs[:,-1] # [1,5,9,7,7]
        
        new_reverse_seqs = torch.zeros_like(self.dataset[self.iid_list_field])
        for i, rev_whole_seq in enumerate(rev_whole_seqs):
            new_rev_seq = rev_whole_seq[rev_whole_seq>0] [0:-1]
            new_reverse_seqs[i][0:len(new_rev_seq)] = new_rev_seq
        
        self.dataset.inter_feat.interaction["reverse_seqs"] =  new_reverse_seqs

class MCLRecTrainDataLoader(CL4RecTrainDataLoader):

    def __init__(self, config, dataset, sampler, shuffle=False):

        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _contrastive_learning_augmentation(self, cur_data):

        # cur_data = self._neg_sampling(cur_data)
        sequences = cur_data[self.iid_list_field]
        lengths = cur_data[self.item_list_length_field]
        aug_seq1, aug_len1 = self._augmentation(sequences, lengths, aug_type=self.aug_type1)
        aug_seq2, aug_len2 = self._augmentation(sequences, lengths, aug_type=self.aug_type2)
        cur_data.update(Interaction({'aug1': aug_seq1, 'aug_len1': aug_len1,
                                     'aug2': aug_seq2, 'aug_len2': aug_len2}))
        return cur_data
