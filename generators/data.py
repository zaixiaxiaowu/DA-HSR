import copy
import numpy as np
from torch.utils.data import Dataset
from utils.utils import random_neq
from generators.data_utils import truncate_padding


class SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation'''

    def __init__(self, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.var_name = ["seq", "pos", "neg", "positions"]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        non_neg = list(non_neg)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions


class CDSRSeq2SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation with seq-to-seq loss'''

    def __init__(self, args, data, domain_data, item_num_dict, max_len, neg_num=1):
        super().__init__()
        self.data = data
        self.domain_data = domain_data
        self.item_numA = item_num_dict["0"]
        self.item_numB = item_num_dict["1"]
        self.item_num = self.item_numA + self.item_numB
        self.max_len = max_len
        self.neg_num = neg_num
        self.aug_seq = args.aug_seq
        self.aug_seq_len = args.aug_seq_len
        
        self.var_name = ["seq", "pos", "neg", "positions",
                         "seqA", "posA", "negA", "positionsA",
                         "seqB", "posB", "negB", "positionsB",
                         "target_domain", "domain_mask", 
                         "gather_idx_A", "gather_idx_B"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inter = self.data[index]
        inter = np.array(inter)
        domain_mask = self.domain_data[index]
        domain_mask = np.array(domain_mask)

        all_inter = copy.deepcopy(inter)
        all_inter[np.where(domain_mask==1)] = all_inter[np.where(domain_mask==1)] + self.item_numA
        seq, pos, neg, positions, mask = truncate_padding(all_inter, domain_mask, self.max_len, self.item_numA, self.item_numB)

        inter_len = len(all_inter)
        
        if inter_len > self.max_len:
            seq_domain_mask = domain_mask[-(self.max_len+1) : -1]
        else:
            valid_mask_segment = domain_mask[:-1]
            seq_domain_mask = np.concatenate([np.zeros(self.max_len - len(valid_mask_segment)), valid_mask_segment])

    
        indices_A = np.where(seq_domain_mask == 0)[0]
        real_indices_A = [i for i in indices_A if seq[i] != 0] 
        
        if len(real_indices_A) > 0:
            gather_idx_A = real_indices_A[-1] 
        else:
            gather_idx_A = 0 

        indices_B = np.where(seq_domain_mask == 1)[0]
        real_indices_B = [i for i in indices_B if seq[i] != 0]
        
        if len(real_indices_B) > 0:
            gather_idx_B = real_indices_B[-1]
        else:
            gather_idx_B = 0

        target_domain = domain_mask[-1]

        indices_A_all = np.where(domain_mask[:-1] == 0)[0]
        if len(indices_A_all) > 0:
            interA = inter[indices_A_all]
            domain_maskA = np.zeros_like(interA)
            seqA, posA, negA, positionsA, _ = truncate_padding(interA, domain_maskA, self.max_len, self.item_numA, self.item_numB)
        else:
            seqA = np.zeros(self.max_len, dtype=np.int32)
            posA = np.zeros(self.max_len, dtype=np.int32)
            if self.neg_num == 1:
                negA = np.zeros(self.max_len, dtype=np.int32)
            else:
                negA = np.zeros((self.max_len, self.neg_num), dtype=np.int32)
            positionsA = np.zeros(self.max_len, dtype=np.int32)

        indices_B_all = np.where(domain_mask[:-1] == 1)[0]
        if len(indices_B_all) > 0:
            interB = inter[indices_B_all]
            domain_maskB = np.ones_like(interB) 
            seqB, posB, negB, positionsB, _ = truncate_padding(interB, domain_maskB, self.max_len, self.item_numB, self.item_numB)
        else:
            seqB = np.zeros(self.max_len, dtype=np.int32)
            posB = np.zeros(self.max_len, dtype=np.int32)
            if self.neg_num == 1:
                negB = np.zeros(self.max_len, dtype=np.int32)
            else:
                negB = np.zeros((self.max_len, self.neg_num), dtype=np.int32)
            positionsB = np.zeros(self.max_len, dtype=np.int32)

        if domain_mask[-self.max_len-1:][0] == 0 and (mask==1).sum() != 0: 
            if len(np.where(mask==1)[0]) > 0:
                index_tmp = np.where(mask==1)[0][0]
                mask[index_tmp] = -1
        elif domain_mask[-self.max_len-1:][0] == 1 and (mask==0).sum() != 0: 
            if len(np.where(mask==0)[0]) > 0:
                index_tmp = np.where(mask==0)[0][0]
                mask[index_tmp] = -1

        return seq, pos, neg, positions, \
               seqA, posA, negA, positionsA, \
               seqB, posB, negB, positionsB, \
               target_domain, mask, \
               gather_idx_A, gather_idx_B


class CDSREvalSeq2SeqDataset(Dataset):
    '''The eval dataset for Sequential recommendation'''

    def __init__(self, args, data, domain_data, item_num_dict, max_len, neg_num=1):
        super().__init__()
        self.data = data
        self.domain_data = domain_data
        self.item_numA = item_num_dict["0"]
        self.item_numB = item_num_dict["1"]
        self.item_num = self.item_numA + self.item_numB
        self.max_len = max_len
        self.neg_num = neg_num
        self.aug_seq = args.aug_seq
        self.aug_seq_len = args.aug_seq_len
        
        self.var_name = ["seq", "pos", "neg", "positions",
                         "seqA", "posA", "negA", "positionsA",
                         "seqB", "posB", "negB", "positionsB",
                         "target_domain", "domain_mask", "user_id"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inter = self.data[index]
        domain_mask = self.domain_data[index]
        domain_mask = np.array(domain_mask)
        target_domain = domain_mask[-1]

        inter = np.array(inter)
        all_inter = copy.deepcopy(inter)
        all_inter[np.where(domain_mask==1)] = all_inter[np.where(domain_mask==1)] + self.item_numA
        all_inter = list(all_inter)

        pos = all_inter[-1]
        neg, negA, negB = [], [], []
        non_neg = copy.deepcopy(all_inter)
        non_neg_A = list(inter[np.where(domain_mask==0)])
        non_neg_B = list(inter[np.where(domain_mask==1)])
        
        for _ in range(self.neg_num):
            if target_domain == 0:
                per_neg = random_neq(1, self.item_numA, non_neg)
            elif target_domain == 1:
                per_neg = random_neq(self.item_numA+1, self.item_numA+self.item_numB, non_neg)
            else:
                raise ValueError
            neg.append(per_neg)
            non_neg.append(per_neg)

            per_neg_A = random_neq(1, self.item_numA, non_neg_A)
            negA.append(per_neg_A)
            non_neg_A.append(per_neg_A)

            per_neg_B = random_neq(1, self.item_numB, non_neg_B)
            negB.append(per_neg_B)
            non_neg_B.append(per_neg_B)

        neg, negA, negB = np.array(neg), np.array(negA), np.array(negB)

        seq, _, _, positions, mask = truncate_padding(all_inter, domain_mask, self.max_len, self.item_numA, self.item_numB)

        if len(np.where(domain_mask==0)[0]) == 0:   
            posA = 0
        else:
            posA = inter[np.where(domain_mask==0)][-1]

        if len(np.where(domain_mask==1)[0]) == 0:
            posB = 0
        else:
            posB = inter[np.where(domain_mask==1)][-1]

        inter = np.array(inter)
        interA = inter[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==0)]
        domain_maskA = domain_mask[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==0)]
        seqA, _, _, positionsA, _ = truncate_padding(interA, domain_maskA, self.max_len, self.item_numA, self.item_numB)
        interB = inter[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==1)]
        domain_maskB = np.zeros_like(domain_mask[-self.max_len-1:][np.where(domain_mask[-self.max_len-1:]==1)])
        seqB, _, _, positionsB, _ = truncate_padding(interB, domain_maskB, self.max_len, self.item_numB, self.item_numB)

        if domain_mask[-self.max_len-1:][0] == 0 and (mask==1).sum() != 0: 
            index_tmp = np.where(mask==1)[0][0]
            mask[index_tmp] = -1
        elif domain_mask[-self.max_len-1:][0] == 1 and (mask==0).sum() != 0: 
            index_tmp = np.where(mask==0)[0][0]
            mask[index_tmp] = -1
        
        return seq, pos, neg, positions, \
               seqA, posA, negA, positionsA, \
               seqB, posB, negB, positionsB, \
               target_domain, mask, index


class CDSRRegSeq2SeqDatasetUser(CDSRSeq2SeqDataset):

    def __init__(self, args, data, domain_data, item_num_dict, max_len, neg_num=1):
        super().__init__(args, data, domain_data, item_num_dict, max_len, neg_num)
        
        self.var_name = ["seq", "pos", "neg", "positions",
                         "seqA", "posA", "negA", "positionsA",
                         "seqB", "posB", "negB", "positionsB",
                         "target_domain", "domain_mask",
                         "gather_idx_A", "gather_idx_B", 
                         "reg_A", "reg_B", "user_id"]

    def __getitem__(self, index):
        base_returns = super().__getitem__(index)
        seq, pos, neg, positions, \
        seqA, posA, negA, positionsA, \
        seqB, posB, negB, positionsB, \
        target_domain, mask, \
        gather_idx_A, gather_idx_B = base_returns

        inter = self.data[index][:-1]
        domain_mask = copy.deepcopy(self.domain_data[index][:-1])

        inter = np.array(inter)
        domain_mask = np.array(domain_mask)
        inter[np.where(domain_mask==1)] = inter[np.where(domain_mask==1)] + self.item_numA

        index_A = np.where(domain_mask==0)[0]
        index_B = np.where(domain_mask==1)[0]

        if len(index_A) == 0 or len(index_B) == 0:
            reg_A, reg_B = np.array([0]), np.array([0])
        else:
            reg_index_A = np.random.choice(index_A, 1)
            reg_index_B = np.random.choice(index_B, 1)
            reg_A = inter[reg_index_A]
            reg_B = inter[reg_index_B]

        return seq, pos, neg, positions, \
               seqA, posA, negA, positionsA, \
               seqB, posB, negB, positionsB, \
               target_domain, mask, \
               gather_idx_A, gather_idx_B, \
               reg_A, reg_B, index