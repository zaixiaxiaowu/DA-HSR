import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.utils import Contrastive_Loss2, cal_bpr_loss

# ------------------------------------------------------------------
# 门控残差网络 (Gated Residual Network)
# 用于融合 LLM 语义向量 和 ID/RQ-VAE 协同向量
# ------------------------------------------------------------------
class GatedResidualFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb_semantic, emb_collaborative):
        base_feat = emb_semantic + emb_collaborative
        concat = torch.cat([emb_semantic, emb_collaborative], dim=-1)
        gate_weight = self.sigmoid(self.gate_linear(concat))
        final_feat = base_feat + (gate_weight * emb_semantic)
        
        return self.layer_norm(final_feat)


class LLM4CDSR_base(BaseSeqModel):

    def __init__(self, user_num, item_num_dict, device, args) -> None:
        
        self.item_numA, self.item_numB = item_num_dict["0"], item_num_dict["1"]
        item_num =  self.item_numA + self.item_numB

        super().__init__(user_num, item_num, device, args)

        self.global_emb = args.global_emb
        self.args = args 
        self.n_neg = getattr(args, 'n_neg', 20) 
        self.tau = getattr(args, 'tau', 0.1)

        # ------------------------------------------------------------------
        # 1. 加载数据
        # ------------------------------------------------------------------
        llm_emb_all = pickle.load(open("./data/{}/handled/{}_all.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        llm_emb_q = pickle.load(open("./data/{}/handled/{}.pkl".format(args.dataset, args.sid_emb_file), "rb"))

        llm_emb_A = llm_emb_all[:self.item_numA]
        llm_emb_B = llm_emb_all[self.item_numA:]

        llm_emv_A_q = llm_emb_q[:self.item_numA]
        llm_emv_B_q = llm_emb_q[self.item_numA:]
        llm_item_emb = np.concatenate([np.zeros((1, llm_emb_all.shape[1])), llm_emb_all])
        
        if args.global_emb:
            self.item_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb), padding_idx=0)
            self.item_emb_llm_q = nn.Embedding.from_pretrained(torch.Tensor(np.concatenate([np.zeros((1, llm_emb_q.shape[1])), llm_emb_q])), padding_idx=0)
        else:
            self.item_emb_llm = nn.Embedding(self.item_numA+self.item_numB+1, args.hidden_size, padding_idx=0)
        
        if args.freeze_emb:
            self.item_emb_llm.weight.requires_grad = False
            if hasattr(self, 'item_emb_llm_q'):
                self.item_emb_llm_q.weight.requires_grad = False
        else:
            self.item_emb_llm.weight.requires_grad = True

        # ------------------------------------------------------------------
        # 2. Adapters & Residual Gates
        # ------------------------------------------------------------------
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )
        self.adapter_A = nn.Sequential(
            nn.Linear(llm_emb_A.shape[1], int(llm_emb_A.shape[1] / 2)),
            nn.Linear(int(llm_emb_A.shape[1] / 2), args.hidden_size)
        )
        self.adapter_B = nn.Sequential(
            nn.Linear(llm_emb_B.shape[1], int(llm_emb_B.shape[1] / 2)),
            nn.Linear(int(llm_emb_B.shape[1] / 2), args.hidden_size)
        )
        
        # Embedding 融合门控
        self.gate_A = GatedResidualFusion(args.hidden_size)
        self.gate_B = GatedResidualFusion(args.hidden_size)
        self.gate_Global = GatedResidualFusion(args.hidden_size)

        # ------------------------------------------------------------------
        # 3. Backbone (SASRec)
        # ------------------------------------------------------------------
        self.pos_emb = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.backbone = SASRecBackbone(device, args)

        # Domain A
        if args.local_emb:
            llm_embA = np.concatenate([np.zeros((1, llm_emb_A.shape[1])), llm_emb_A])
            self.item_embA = nn.Embedding.from_pretrained(torch.Tensor(llm_embA), padding_idx=0)
            self.item_emb_llmA_q = nn.Embedding.from_pretrained(torch.Tensor(np.concatenate([np.zeros((1, llm_emv_A_q.shape[1])), llm_emv_A_q])), padding_idx=0)
        else:
            self.item_embA = nn.Embedding(self.item_numA+1, args.hidden_size, padding_idx=0)
        self.pos_embA = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropoutA = nn.Dropout(p=args.dropout_rate)
        self.backboneA = SASRecBackbone(device, args)

        # Domain B
        if args.local_emb:
            llm_embB = np.concatenate([np.zeros((1, llm_emb_B.shape[1])), llm_emb_B])
            self.item_embB = nn.Embedding.from_pretrained(torch.Tensor(llm_embB), padding_idx=0)
            self.item_emb_llmB_q = nn.Embedding.from_pretrained(torch.Tensor(np.concatenate([np.zeros((1, llm_emv_B_q.shape[1])), llm_emv_B_q])), padding_idx=0)
        else:
            self.item_embB = nn.Embedding(self.item_numB+1, args.hidden_size, padding_idx=0)
        self.pos_embB = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropoutB = nn.Dropout(p=args.dropout_rate)
        self.backboneB = SASRecBackbone(device, args)


        # ------------------------------------------------------------------
        # 4. Alignment Projectors (用于对齐损失)
        # ------------------------------------------------------------------
        # 使用线性层将 Global 特征投影到 A/B 域空间，进行 Soft Alignment
        self.proj_global_to_A = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_global_to_B = nn.Linear(args.hidden_size, args.hidden_size)
        
        # 初始化为 Identity (近似 Hard Alignment 作为起点)
        nn.init.eye_(self.proj_global_to_A.weight)
        nn.init.zeros_(self.proj_global_to_A.bias)
        nn.init.eye_(self.proj_global_to_B.weight)
        nn.init.zeros_(self.proj_global_to_B.bias)

        self.seq_cl_loss_func = Contrastive_Loss2(tau=args.tau)

        if args.global_emb: 
            self.filter_init_modules.append("item_emb_llm")
        if args.local_emb:
            self.filter_init_modules.append("item_embA")
            self.filter_init_modules.append("item_embB")
        self._init_weights()


    def _get_embedding(self, log_seqs, domain="A"):
        if domain == "A":
            feat_semantic = self.adapter_A(self.item_embA(log_seqs))
            feat_collab = self.item_emb_llmA_q(log_seqs)
            item_seq_emb = self.gate_A(feat_semantic, feat_collab)
            
        elif domain == "B":
            feat_semantic = self.adapter_B(self.item_embB(log_seqs))
            feat_collab = self.item_emb_llmB_q(log_seqs)
            item_seq_emb = self.gate_B(feat_semantic, feat_collab)
            
        elif domain == "AB":
            if self.global_emb:
                feat_semantic = self.adapter(self.item_emb_llm(log_seqs))
                feat_collab = self.item_emb_llm_q(log_seqs)
                item_seq_emb = self.gate_Global(feat_semantic, feat_collab)
            else:
                item_seq_emb = self.item_emb_llm(log_seqs)
        else:
            raise ValueError
        return item_seq_emb
    

    def log2feats(self, log_seqs, positions, domain="A"):
        if domain == "AB":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_emb_llm.embedding_dim ** 0.5
            seqs += self.pos_emb(positions.long())
            seqs = self.emb_dropout(seqs)
            log_feats = self.backbone(seqs, log_seqs)
        elif domain == "A":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_embA.embedding_dim ** 0.5
            seqs += self.pos_embA(positions.long())
            seqs = self.emb_dropoutA(seqs)
            log_feats = self.backboneA(seqs, log_seqs)
        elif domain == "B":
            seqs = self._get_embedding(log_seqs, domain=domain)
            seqs *= self.item_embB.embedding_dim ** 0.5
            seqs += self.pos_embB(positions.long())
            seqs = self.emb_dropoutB(seqs)
            log_feats = self.backboneB(seqs, log_seqs)
        return log_feats
    
    def strict_negative_sampling(self, seq, pos, domain="AB", n_neg=20):
        batch_size, seq_len = seq.shape
        if domain == "A": high = self.item_numA
        elif domain == "B": high = self.item_numB
        else: high = self.item_numA + self.item_numB

        neg_ids = torch.zeros((batch_size, seq_len, n_neg), dtype=torch.long, device=self.dev)
        valid_mask = (pos != 0)
        
        seq_cpu = seq.cpu().numpy()
        pos_cpu = pos.cpu().numpy()
        valid_mask_cpu = valid_mask.cpu().numpy()

        for i in range(batch_size):
            seen_items = set(seq_cpu[i][seq_cpu[i] > 0]) | set(pos_cpu[i][pos_cpu[i] > 0])
            n_valid = np.sum(valid_mask_cpu[i])
            if n_valid == 0: continue

            candidates = np.random.randint(1, high + 1, size=(n_valid * n_neg * 2))
            valid_negs = []
            for cand in candidates:
                if cand not in seen_items:
                    valid_negs.append(cand)
                    if len(valid_negs) == n_valid * n_neg: break
            while len(valid_negs) < n_valid * n_neg:
                cand = np.random.randint(1, high + 1)
                if cand not in seen_items: valid_negs.append(cand)
            
            valid_negs = torch.tensor(valid_negs, dtype=torch.long, device=self.dev).view(n_valid, n_neg)
            neg_ids[i, valid_mask[i], :] = valid_negs
        return neg_ids


    def cal_infonce_loss(self, log_feats, seq, pos, domain="AB"):
        tau = self.tau
        mask = (pos != 0)
        valid_feats = log_feats[mask] 
        valid_pos = pos[mask]         
        
        if valid_feats.shape[0] == 0:
            return torch.tensor(0.0, device=self.dev, requires_grad=True)

        pos_embs = self._get_embedding(valid_pos, domain=domain)
        neg_ids_full = self.strict_negative_sampling(seq, pos, domain=domain, n_neg=self.n_neg)
        valid_neg_ids = neg_ids_full[mask] 
        neg_embs = self._get_embedding(valid_neg_ids.view(-1), domain=domain)
        neg_embs = neg_embs.view(valid_feats.shape[0], self.n_neg, -1)
        
        pos_logits = (valid_feats * pos_embs).sum(dim=-1, keepdim=True) / tau
        neg_logits = torch.bmm(neg_embs, valid_feats.unsqueeze(-1)).squeeze(-1) / tau
        
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.dev)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, 
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                gather_idx_A=None, gather_idx_B=None,
                **kwargs):
        
        # --- 1. Global (Mixed) Sequence ---
        log_feats_global = self.log2feats(seq, positions, domain="AB")
        lossAB = self.cal_infonce_loss(log_feats_global, seq, pos, domain="AB")

        # --- 2. Domain A ---
        log_featsA = self.log2feats(seqA, positionsA, domain="A")
        lossA = self.cal_infonce_loss(log_featsA, seqA, posA, domain="A")
        final_A = log_featsA[:, -1, :] 

        # --- 3. Domain B ---
        log_featsB = self.log2feats(seqB, positionsB, domain="B")
        lossB = self.cal_infonce_loss(log_featsB, seqB, posB, domain="B")
        final_B = log_featsB[:, -1, :] 

        # --- 4. Alignment Loss --- 
        # 提取 A 时刻的 Global 特征
        if gather_idx_A is not None:
            idx_A_expanded = gather_idx_A.view(-1, 1, 1).expand(-1, -1, log_feats_global.size(-1))
            global_feat_at_A = torch.gather(log_feats_global, 1, idx_A_expanded).squeeze(1)
        else:
            global_feat_at_A = log_feats_global[:, -1, :] 

        # 提取 B 时刻的 Global 特征
        if gather_idx_B is not None:
            idx_B_expanded = gather_idx_B.view(-1, 1, 1).expand(-1, -1, log_feats_global.size(-1))
            global_feat_at_B = torch.gather(log_feats_global, 1, idx_B_expanded).squeeze(1)
        else:
            global_feat_at_B = log_feats_global[:, -1, :]

        target_A = global_feat_at_A.detach()
        target_B = global_feat_at_B.detach()

        # 投影并计算损失
        global_view_A = self.proj_global_to_A(target_A)
        align_loss_A = self.seq_cl_loss_func(global_view_A, final_A)

        global_view_B = self.proj_global_to_B(target_B)
        align_loss_B = self.seq_cl_loss_func(global_view_B, final_B)
        
        loss = lossAB + lossA + lossB + self.alpha * (align_loss_A + align_loss_B)

        return loss
    

    def predict(self,
                seq, item_indices, positions,
                seqA, item_indicesA, positionsA,
                seqB, item_indicesB, positionsB,
                target_domain,
                **kwargs): 
        
        # --- Global Prediction ---
        log_feats_global = self.log2feats(seq, positions, domain="AB")
        final_feat = log_feats_global[:, -1, :] 
        item_embs = self._get_embedding(item_indices, domain="AB") 
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # --- Domain A Prediction ---
        log_featsA = self.log2feats(seqA, positionsA, domain="A") 
        final_featA = log_featsA[:, -1, :] 
        item_embsA = self._get_embedding(item_indicesA, domain="A") 
        logitsA = item_embsA.matmul(final_featA.unsqueeze(-1)).squeeze(-1)

        # --- Domain B Prediction ---
        log_featsB = self.log2feats(seqB, positionsB, domain="B") 
        final_featB = log_featsB[:, -1, :] 
        item_embsB = self._get_embedding(item_indicesB, domain="B") 
        logitsB = item_embsB.matmul(final_featB.unsqueeze(-1)).squeeze(-1)

        # --- Fusion (Late Fusion) ---
        logits[target_domain==0] += logitsA[target_domain==0]
        logits[target_domain==1] += logitsB[target_domain==1]

        return logits


class LLM4CDSR(LLM4CDSR_base):
    def __init__(self, user_num, item_num_dict, device, args):
        super().__init__(user_num, item_num_dict, device, args)
        self.alpha = args.alpha
        

    def forward(self, 
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                reg_A, reg_B,
                user_id,
                gather_idx_A=None, gather_idx_B=None, 
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions,
                    seqA, posA, negA, positionsA,
                    seqB, posB, negB, positionsB,
                    target_domain, domain_mask,
                    gather_idx_A=gather_idx_A, 
                    gather_idx_B=gather_idx_B,
                    **kwargs)

        return loss