import torch
import torch.nn as nn

from vq import DomainAdaptiveVectorQuantizer


class DomainAdaptiveResidualQuantizer(nn.Module):
    """Hierarchical DRQ with L shared codebooks and domain-adaptive codewords."""

    def __init__(
        self,
        n_e_list,
        e_dim,
        rank=16,
        n_domains=2,
        kmeans_init=False,
        kmeans_iters=100,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.vq_layers = nn.ModuleList(
            [
                DomainAdaptiveVectorQuantizer(
                    n_e=n_e,
                    e_dim=e_dim,
                    rank=rank,
                    n_domains=n_domains,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                )
                for n_e in n_e_list
            ]
        )

    def get_code(self):
        return [quantizer.get_codebook() for quantizer in self.vq_layers]

    def adaptation_reg(self):
        loss = 0.0
        for quantizer in self.vq_layers:
            loss = loss + quantizer.adaptation_reg()
        return loss

    def vq_ini(self, x):
        residual = x
        for quantizer in self.vq_layers:
            quantized = quantizer.vq_init(residual)
            residual = residual - quantized

    def forward(self, x, domains):
        residual = x
        reconstructed = torch.zeros_like(x)
        layer_codes = []
        all_indices = []
        commit_loss = 0.0

        for quantizer in self.vq_layers:
            quantized, indices, commit = quantizer(residual, domains)
            residual = residual - quantized.detach()
            reconstructed = reconstructed + quantized
            layer_codes.append(quantized)
            all_indices.append(indices)
            commit_loss = commit_loss + commit

        commit_loss = commit_loss / max(len(self.vq_layers), 1)
        layer_codes = torch.stack(layer_codes, dim=1)
        all_indices = torch.stack(all_indices, dim=-1)
        return reconstructed, layer_codes, commit_loss, self.adaptation_reg(), all_indices
