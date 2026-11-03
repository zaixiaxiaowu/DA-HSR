import torch
from torch import nn
from layers import MLPLayers
from rq import DomainAdaptiveResidualQuantizer


def multi_kernel_mmd(source, target, sigmas=(0.5, 1.0, 2.0, 4.0)):
    def gaussian_kernel(x, y, sigma):
        dist = torch.cdist(x, y).pow(2)
        return torch.exp(-dist / (2 * sigma ** 2))

    loss = source.new_zeros(())
    for sigma in sigmas:
        k_xx = gaussian_kernel(source, source, sigma)
        k_yy = gaussian_kernel(target, target, sigma)
        k_xy = gaussian_kernel(source, target, sigma)
        loss = loss + k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return loss


class DRQVAE(nn.Module):
    """Domain-adaptive Residual Quantization autoencoder (Stage 1)."""

    def __init__(
        self,
        in_dim=768,
        num_emb_list=None,
        e_dim=64,
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mmd",
        quant_loss_weight=0.1,
        kmeans_init=False,
        kmeans_iters=100,
        rank=16,
        n_domains=2,
        adapt_loss_weight=1e-4,
        recon=1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.adapt_loss_weight = adapt_loss_weight
        self.recon = recon

        encode_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.text_encoder = MLPLayers(layers=encode_dims, dropout=dropout_prob, bn=bn)
        self.text_rq = DomainAdaptiveResidualQuantizer(
            num_emb_list,
            e_dim,
            rank=rank,
            n_domains=n_domains,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
        )
        self.text_decoder = MLPLayers(layers=encode_dims[::-1], dropout=dropout_prob, bn=bn)

    def forward(self, x, domains):
        latent = self.text_encoder(x)
        reconstructed, layer_codes, commit_loss, adapt_loss, indices = self.text_rq(latent, domains)
        decoded = self.text_decoder(reconstructed)
        return decoded, commit_loss, adapt_loss, indices, layer_codes

    def vq_initialization(self, x, use_sk=True):
        self.text_rq.vq_ini(self.text_encoder(x))

    @torch.no_grad()
    def get_indices(self, xs, domains):
        latent = self.text_encoder(xs)
        _, layer_codes, _, _, indices = self.text_rq(latent, domains)
        return indices, layer_codes

    def compute_loss(self, decoded, commit_loss, adapt_loss, xs):
        if self.loss_type == "mse":
            loss_recon = nn.functional.mse_loss(decoded, xs)
        elif self.loss_type == "mmd":
            loss_recon = multi_kernel_mmd(xs.detach(), decoded)
        else:
            raise ValueError("incompatible loss type")

        total_loss = (
            self.recon * loss_recon
            + self.quant_loss_weight * commit_loss
            + self.adapt_loss_weight * adapt_loss
        )
        return total_loss, loss_recon, commit_loss, adapt_loss
