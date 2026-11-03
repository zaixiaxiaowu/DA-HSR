import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.modules import (
    SequenceAwareRouter,
    CrossDomainGuidedAttention,
    HyperbolicHCL,
    cosine_infonce,
    augment_sequence,
)


class DAHSR(BaseSeqModel):
    """Domain-Aware Hierarchical Semantic Representation."""

    def __init__(self, user_num, item_num_dict, device, args):
        self.item_numA = item_num_dict["0"]
        self.item_numB = item_num_dict["1"]
        item_num = self.item_numA + self.item_numB
        super().__init__(user_num, item_num, device, args)

        self.args = args
        self.hidden_size = args.hidden_size
        self.n_neg = getattr(args, "n_neg", 20)
        self.tau = getattr(args, "tau", 1.0)
        self.tau1 = getattr(args, "tau1", 0.1)
        self.tau2 = getattr(args, "tau2", 0.1)
        self.alpha = getattr(args, "alpha", 0.5)
        self.beta = getattr(args, "beta", 0.1)
        self.gamma = getattr(args, "gamma", 0.05)
        self.mask_ratio = getattr(args, "mask_ratio", 0.2)
        self.crop_ratio = getattr(args, "crop_ratio", 0.2)

        llm_emb_all = self._load_array("./data/{}/handled/{}_all.pkl".format(args.dataset, args.llm_emb_file))
        layer_emb = self._load_array("./data/{}/handled/{}.pkl".format(args.dataset, args.sid_emb_file))
        if layer_emb.ndim == 2:
            layer_emb = layer_emb[:, None, :]

        assert llm_emb_all.shape[0] == self.item_numA + self.item_numB, (
            "LLM embedding count must equal the number of items in both domains."
        )
        assert layer_emb.shape[0] == llm_emb_all.shape[0], (
            "DRQ layer embeddings must be aligned with LLM embeddings."
        )

        self.n_code_layers = layer_emb.shape[1]
        self.code_dim = layer_emb.shape[2]
        self.semantic_dim = llm_emb_all.shape[1]

        self.llm_emb = self._frozen_embedding(llm_emb_all)
        self.llm_emb_A = self._frozen_embedding(llm_emb_all[:self.item_numA])
        self.llm_emb_B = self._frozen_embedding(llm_emb_all[self.item_numA:])
        self.codes = self._frozen_codebook(layer_emb)
        self.codes_A = self._frozen_codebook(layer_emb[:self.item_numA])
        self.codes_B = self._frozen_codebook(layer_emb[self.item_numA:])

        self.adapter = self._make_adapter(self.semantic_dim, args.hidden_size)
        self.adapter_A = self._make_adapter(self.semantic_dim, args.hidden_size)
        self.adapter_B = self._make_adapter(self.semantic_dim, args.hidden_size)

        self.id_emb = nn.Embedding(self.item_numA + self.item_numB + 1, args.hidden_size, padding_idx=0)
        self.id_A = nn.Embedding(self.item_numA + 1, args.hidden_size, padding_idx=0)
        self.id_B = nn.Embedding(self.item_numB + 1, args.hidden_size, padding_idx=0)

        self.router = SequenceAwareRouter(
            hidden_size=args.hidden_size,
            n_code_layers=self.n_code_layers,
            semantic_dim=args.hidden_size,
            code_dim=self.code_dim,
            n_domains=3,
        )
        self.cga = CrossDomainGuidedAttention(args.hidden_size)
        self.hcl = HyperbolicHCL(
            hidden_size=args.hidden_size,
            n_levels=self.n_code_layers + 2,
            curvature=getattr(args, "lorentz_c", 1.0),
            cone_k=getattr(args, "cone_k", 0.1),
        )

        self.pos_emb = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.pos_emb_A = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.pos_emb_B = nn.Embedding(args.max_len + 1, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.emb_dropout_A = nn.Dropout(p=args.dropout_rate)
        self.emb_dropout_B = nn.Dropout(p=args.dropout_rate)
        self.backbone = SASRecBackbone(device, args)
        self.backbone_A = SASRecBackbone(device, args)
        self.backbone_B = SASRecBackbone(device, args)

        self.proj_G2A = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
        )
        self.proj_G2B = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
        )

        self.filter_init_modules.extend(["llm_emb", "llm_emb_A", "llm_emb_B", "codes", "codes_A", "codes_B"])
        self._init_weights()

    @staticmethod
    def _load_array(path):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _make_adapter(in_dim, out_dim):
        hidden = max(out_dim, in_dim // 2)
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def _frozen_embedding(self, matrix):
        padded = np.concatenate([np.zeros((1, matrix.shape[1]), dtype=np.float32), matrix], axis=0)
        emb = nn.Embedding.from_pretrained(torch.tensor(padded), padding_idx=0, freeze=True)
        return emb

    def _frozen_codebook(self, matrix):
        padded = np.concatenate(
            [np.zeros((1, matrix.shape[1], matrix.shape[2]), dtype=np.float32), matrix],
            axis=0,
        )
        return nn.Parameter(torch.tensor(padded), requires_grad=False)

    def _domain_high(self, domain):
        if domain == "A":
            return self.item_numA
        if domain == "B":
            return self.item_numB
        return self.item_numA + self.item_numB

    def _domain_index(self, domain, batch_size):
        mapping = {"A": 0, "B": 1, "AB": 2}
        return torch.full((batch_size,), mapping[domain], dtype=torch.long, device=self.dev)

    def _masked_mean(self, features, valid_mask):
        weights = valid_mask.float().unsqueeze(-1)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _get_components(self, item_ids, domain):
        if domain == "A":
            semantic = self.adapter_A(self.llm_emb_A(item_ids))
            codes = self.codes_A[item_ids]
            collab = self.id_A(item_ids)
        elif domain == "B":
            semantic = self.adapter_B(self.llm_emb_B(item_ids))
            codes = self.codes_B[item_ids]
            collab = self.id_B(item_ids)
        else:
            semantic = self.adapter(self.llm_emb(item_ids))
            codes = self.codes[item_ids]
            collab = self.id_emb(item_ids)
        return semantic, codes, collab

    def _fuse(self, item_ids, domain, seq_context, domain_id):
        squeeze = False
        if item_ids.dim() == 1:
            item_ids = item_ids.unsqueeze(1)
            squeeze = True
        semantic, codes, collab = self._get_components(item_ids, domain)
        fused, _, _ = self.router(semantic, codes, collab, seq_context, domain_id)
        if squeeze:
            fused = fused.squeeze(1)
        return fused

    def _log2feats(self, item_emb, log_seqs, positions, domain):
        seqs = item_emb * (self.hidden_size ** 0.5)
        if domain == "A":
            seqs = self.emb_dropout_A(seqs + self.pos_emb_A(positions.long()))
            return self.backbone_A(seqs, log_seqs)
        if domain == "B":
            seqs = self.emb_dropout_B(seqs + self.pos_emb_B(positions.long()))
            return self.backbone_B(seqs, log_seqs)
        seqs = self.emb_dropout(seqs + self.pos_emb(positions.long()))
        return self.backbone(seqs, log_seqs)

    def encode(self, seq, positions, domain, global_hidden=None, global_pad=None):
        domain_id = self._domain_index(domain, seq.size(0))
        semantic, codes, collab = self._get_components(seq, domain)
        seq_context = self._masked_mean(collab, seq > 0)
        fused, _, components = self.router(semantic, codes, collab, seq_context, domain_id)
        hidden = self._log2feats(fused, seq, positions, domain)
        if global_hidden is not None:
            hidden = self.cga(hidden, global_hidden, seq == 0, global_pad)
        user_repr = hidden[:, -1, :]
        empty = ~(seq > 0).any(dim=1)
        if empty.any() and global_hidden is not None:
            user_repr = torch.where(empty.unsqueeze(-1), global_hidden[:, -1, :], user_repr)
        return {
            "H": hidden,
            "z": user_repr,
            "seq_ctx": seq_context,
            "domain_id": domain_id,
            "components": components,
        }

    def cal_rec_loss(self, log_feats, seq, pos, domain, seq_context, domain_id):
        mask = pos != 0
        if mask.sum() == 0:
            return log_feats.new_zeros(())

        valid_feats = log_feats[mask]
        valid_pos = pos[mask]
        batch_size, seq_len, hidden = log_feats.shape
        context = seq_context.unsqueeze(1).expand(batch_size, seq_len, hidden)[mask]
        domain_ids = domain_id.unsqueeze(1).expand(batch_size, seq_len)[mask]
        pos_emb = self._fuse(valid_pos, domain, context, domain_ids)

        n_valid = valid_pos.size(0)
        neg_ids = torch.randint(
            1, self._domain_high(domain) + 1, (n_valid, self.n_neg), device=self.dev
        )
        neg_context = context.repeat_interleave(self.n_neg, dim=0)
        neg_domain = domain_ids.repeat_interleave(self.n_neg, dim=0)
        neg_emb = self._fuse(neg_ids.reshape(-1), domain, neg_context, neg_domain)
        neg_emb = neg_emb.view(n_valid, self.n_neg, hidden)

        pos_logits = (valid_feats * pos_emb).sum(dim=-1, keepdim=True) / self.tau
        neg_logits = torch.bmm(neg_emb, valid_feats.unsqueeze(-1)).squeeze(-1) / self.tau
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(n_valid, dtype=torch.long, device=self.dev)
        return F.cross_entropy(logits, labels)

    def _ccip_loss(self, global_z, local_z, projector, valid):
        if valid.sum() < 2:
            return global_z.new_zeros(())
        projected = projector(global_z[valid])
        return cosine_infonce(projected, local_z[valid], self.tau1)

    def _iic_loss(self, seq, positions, domain, valid):
        if valid.sum() < 2:
            return seq.new_zeros((), dtype=torch.float32)
        view1, pos1 = augment_sequence(seq, positions, self.mask_ratio, self.crop_ratio)
        view2, pos2 = augment_sequence(seq, positions, self.mask_ratio, self.crop_ratio)
        z1 = self.encode(view1, pos1, domain)["z"]
        z2 = self.encode(view2, pos2, domain)["z"]
        return cosine_infonce(z1[valid], z2[valid], self.tau2)

    def forward(
        self,
        seq, pos, neg, positions,
        seqA, posA, negA, positionsA,
        seqB, posB, negB, positionsB,
        target_domain, domain_mask,
        gather_idx_A=None, gather_idx_B=None,
        **kwargs
    ):
        global_out = self.encode(seq, positions, "AB")
        local_A = self.encode(seqA, positionsA, "A", global_out["H"], seq == 0)
        local_B = self.encode(seqB, positionsB, "B", global_out["H"], seq == 0)

        rec_loss = (
            self.cal_rec_loss(global_out["H"], seq, pos, "AB", global_out["seq_ctx"], global_out["domain_id"])
            + self.cal_rec_loss(local_A["H"], seqA, posA, "A", local_A["seq_ctx"], local_A["domain_id"])
            + self.cal_rec_loss(local_B["H"], seqB, posB, "B", local_B["seq_ctx"], local_B["domain_id"])
        )

        valid_A = (seqA > 0).any(dim=1)
        valid_B = (seqB > 0).any(dim=1)
        ccip_loss = (
            self._ccip_loss(global_out["z"], local_A["z"], self.proj_G2A, valid_A)
            + self._ccip_loss(global_out["z"], local_B["z"], self.proj_G2B, valid_B)
        )
        iic_loss = self._iic_loss(seqA, positionsA, "A", valid_A) + self._iic_loss(seqB, positionsB, "B", valid_B)

        last_components = [comp[:, -1, :] for comp in global_out["components"]]
        hcl_loss = self.hcl(last_components)

        return rec_loss + self.alpha * ccip_loss + self.beta * iic_loss + self.gamma * hcl_loss

    def _get_embedding(self, log_seqs, domain="AB"):
        if domain in ["0", "A"]:
            domain = "A"
        elif domain in ["1", "B"]:
            domain = "B"
        else:
            domain = "AB"
        if log_seqs.dim() == 1:
            context = torch.zeros(log_seqs.size(0), self.hidden_size, device=self.dev)
            domain_id = self._domain_index(domain, log_seqs.size(0))
            return self._fuse(log_seqs, domain, context, domain_id)
        context = torch.zeros(log_seqs.size(0), self.hidden_size, device=self.dev)
        domain_id = self._domain_index(domain, log_seqs.size(0))
        return self._fuse(log_seqs, domain, context, domain_id)

    def predict(
        self,
        seq, item_indices, positions,
        seqA, item_indicesA, positionsA,
        seqB, item_indicesB, positionsB,
        target_domain,
        **kwargs
    ):
        global_out = self.encode(seq, positions, "AB")
        local_A = self.encode(seqA, positionsA, "A", global_out["H"], seq == 0)
        local_B = self.encode(seqB, positionsB, "B", global_out["H"], seq == 0)

        def score(item_ids, domain, user_state, seq_ctx, domain_id):
            batch, n_cand = item_ids.shape
            ctx = seq_ctx.unsqueeze(1).expand(batch, n_cand, -1).reshape(batch * n_cand, -1)
            did = domain_id.unsqueeze(1).expand(batch, n_cand).reshape(batch * n_cand)
            item_emb = self._fuse(item_ids.reshape(-1), domain, ctx, did).view(batch, n_cand, -1)
            return (item_emb * user_state.unsqueeze(1)).sum(dim=-1)

        logits_A = score(item_indicesA, "A", local_A["z"], local_A["seq_ctx"], local_A["domain_id"])
        logits_B = score(item_indicesB, "B", local_B["z"], local_B["seq_ctx"], local_B["domain_id"])
        target = target_domain.unsqueeze(1)
        return torch.where(target == 0, logits_A, logits_B)
