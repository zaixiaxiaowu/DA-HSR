import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceAwareRouter(nn.Module):
    """Sequence-conditioned cross-attention over L+2 representation components.

    Components are stacked as [static semantic, code_1, ..., code_L, collaborative ID].
    The query is formed from a sequence context vector and a domain identifier.
    """

    def __init__(self, hidden_size, n_code_layers, semantic_dim, code_dim, n_domains=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_code_layers = n_code_layers

        self.proj_s = nn.Linear(semantic_dim, hidden_size)
        self.proj_codes = nn.ModuleList(
            [nn.Linear(code_dim, hidden_size) for _ in range(n_code_layers)]
        )
        self.proj_c = nn.Linear(hidden_size, hidden_size)

        self.domain_emb = nn.Embedding(n_domains, hidden_size)
        self.W_Q = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** 0.5

    def forward(self, e_s, codes, h_c, seq_context, domain_id):
        f_s = self.proj_s(e_s)
        f_codes = [self.proj_codes[layer](codes[:, :, layer, :]) for layer in range(self.n_code_layers)]
        f_c = self.proj_c(h_c)
        components = [f_s] + f_codes + [f_c]
        stacked = torch.stack(components, dim=2)

        query = self.W_Q(torch.cat([seq_context, self.domain_emb(domain_id)], dim=-1))
        key = self.W_K(stacked)
        value = self.W_V(stacked)

        alpha = torch.softmax(torch.einsum("bd,bscd->bsc", query, key) / self.scale, dim=-1)
        fused = torch.einsum("bsc,bscd->bsd", alpha, value)
        return fused, alpha, components


class CrossDomainGuidedAttention(nn.Module):
    """Retrieve global history with a local-domain query (Eq. 22-24)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.ln = nn.LayerNorm(hidden_size)
        self.scale = hidden_size ** 0.5

    def forward(self, H_local, H_global, local_pad, global_pad):
        query = self.W_Q(H_local)
        key = self.W_K(H_global)
        value = self.W_V(H_global)
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        scores = scores.masked_fill(global_pad.unsqueeze(1), -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(local_pad.unsqueeze(-1), 0.0)
        enhanced = torch.matmul(attn, value)
        return self.ln(H_local + self.eta * enhanced)


class HyperbolicHCL(nn.Module):
    """Entailment-cone consistency on the Lorentz manifold (Eq. 16-21)."""

    def __init__(self, hidden_size, n_levels, curvature=1.0, cone_k=0.1):
        super().__init__()
        self.c = curvature
        self.k = cone_k
        self.projs = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_levels)])

    def _lorentz_inner(self, x, y):
        return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)

    def _expmap0(self, tangent):
        norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sqrt_c = math.sqrt(self.c)
        time = torch.cosh(sqrt_c * norm) / sqrt_c
        spatial = torch.sinh(sqrt_c * norm) / (sqrt_c * norm) * tangent
        return torch.cat([time, spatial], dim=-1)

    def _aperture(self, x):
        spatial_norm = x[..., 1:].norm(dim=-1).clamp_min(1e-6)
        arg = (2.0 * self.k) / (math.sqrt(self.c) * spatial_norm)
        return torch.asin(arg.clamp(0.0, 1.0 - 1e-6))

    def _exterior_angle(self, parent, child):
        inner = self._lorentz_inner(parent, child)
        numerator = child[..., 0] + parent[..., 0] * self.c * inner
        spatial_norm = parent[..., 1:].norm(dim=-1).clamp_min(1e-6)
        denominator = spatial_norm * ((self.c * inner).pow(2) - 1.0).clamp_min(1e-6).sqrt()
        arg = (numerator / denominator).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.acos(arg)

    def forward(self, components):
        hyperbolic = [self._expmap0(proj(comp)) for proj, comp in zip(self.projs, components)]
        loss = components[0].new_zeros(())
        for parent, child in zip(hyperbolic[:-1], hyperbolic[1:]):
            loss = loss + F.relu(self._exterior_angle(parent, child) - self._aperture(parent)).mean()
        return loss / max(len(hyperbolic) - 1, 1)


def cosine_infonce(anchor, positive, tau):
    if anchor.size(0) < 2:
        return anchor.new_zeros(())
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.t() / tau
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)


def augment_sequence(seq, positions, mask_ratio=0.2, crop_ratio=0.2):
    """Build an intra-domain view via random masking and prefix cropping."""
    seq = seq.clone()
    valid = seq > 0
    drop = valid & (torch.rand_like(seq, dtype=torch.float32) < mask_ratio)
    seq = seq.masked_fill(drop, 0)

    if crop_ratio > 0:
        batch_size = seq.size(0)
        for i in range(batch_size):
            idx = (seq[i] > 0).nonzero(as_tuple=False).view(-1)
            n_valid = idx.numel()
            if n_valid < 2:
                continue
            cut = int(n_valid * crop_ratio * torch.rand(1, device=seq.device).item())
            if cut > 0:
                seq[i, idx[:cut]] = 0

    positions = torch.cumsum((seq > 0).long(), dim=1) * (seq > 0).long()
    return seq, positions
