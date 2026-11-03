import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdaptiveVectorQuantizer(nn.Module):
    """Shared codebook with a low-rank domain-specific residual (Eq. 3-5)."""

    def __init__(self, n_e, e_dim, rank=16, n_domains=2, kmeans_init=False, kmeans_iters=10):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.rank = rank
        self.n_domains = n_domains
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.A = nn.Parameter(torch.empty(self.n_e, rank))
        self.B = nn.Parameter(torch.empty(rank, self.e_dim))
        self.g = nn.Parameter(torch.empty(n_domains, rank))

        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.B)
        nn.init.xavier_normal_(self.g)

    def get_codebook(self):
        return self.embedding.weight

    def get_adapted_codebook(self, domain_id):
        scale = self.g[domain_id]
        delta = (self.A * scale.unsqueeze(0)) @ self.B
        return self.embedding.weight + delta

    def adaptation_reg(self):
        loss = self.A.new_zeros(())
        for domain_id in range(self.n_domains):
            delta = (self.A * self.g[domain_id].unsqueeze(0)) @ self.B
            loss = loss + delta.pow(2).sum()
        return loss

    def init_emb(self, data):
        try:
            from k_means_constrained import KMeansConstrained

            x = data.detach().cpu().numpy()
            n_clusters = self.n_e
            size_min = min(max(len(data) // (n_clusters * 2), 1), 50)
            size_max = int(len(data) / n_clusters) + 1
            clf = KMeansConstrained(
                n_clusters=n_clusters,
                size_min=size_min,
                size_max=size_max,
                max_iter=self.kmeans_iters,
                n_init=10,
                n_jobs=10,
                verbose=False,
            )
            clf.fit(x)
            centers = torch.from_numpy(clf.cluster_centers_).to(self.embedding.weight.device)
        except Exception:
            from sklearn.cluster import KMeans

            x = data.detach().cpu().numpy()
            clf = KMeans(n_clusters=self.n_e, n_init=10, max_iter=self.kmeans_iters)
            clf.fit(x)
            centers = torch.from_numpy(clf.cluster_centers_).to(self.embedding.weight.device)
        self.embedding.weight.data.copy_(centers.to(self.embedding.weight.dtype))
        self.initted = True

    def vq_init(self, x):
        latent = x.reshape(-1, self.e_dim)
        if not self.initted:
            self.init_emb(latent)
        codebook = self.embedding.weight
        dist = (
            latent.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1)
            - 2 * latent @ codebook.t()
        )
        indices = torch.argmin(dist, dim=-1)
        return codebook[indices].view_as(x)

    def forward(self, residual, domains):
        latent = residual.reshape(-1, self.e_dim)
        if latent.size(0) != domains.numel():
            domains = domains.reshape(-1)
        if not self.initted and self.training:
            self.init_emb(latent)

        quantized = torch.zeros_like(latent)
        indices = torch.zeros(latent.size(0), dtype=torch.long, device=latent.device)
        commit = latent.new_zeros(())

        for domain_id in domains.unique():
            domain_id = int(domain_id.item())
            mask = domains == domain_id
            z = latent[mask]
            codebook = self.get_adapted_codebook(domain_id)
            dist = (
                z.pow(2).sum(dim=1, keepdim=True)
                + codebook.pow(2).sum(dim=1)
                - 2 * z @ codebook.t()
            )
            idx = torch.argmin(dist, dim=-1)
            code = codebook[idx]
            quantized[mask] = code
            indices[mask] = idx
            commit = commit + F.mse_loss(z, code.detach(), reduction="sum")

        commit = commit / max(latent.size(0), 1)
        quantized = quantized.view_as(residual)
        return quantized, indices.view(residual.shape[:-1]), commit
