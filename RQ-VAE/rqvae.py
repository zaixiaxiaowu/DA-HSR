import torch
from torch import nn
from torch.nn import functional as F
from layers import MLPLayers
from rq import ResidualVectorQuantizer
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
from tllib.modules.kernels import GaussianKernel

class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 align=0.01,
                 recon=1  
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.n_clusters = n_clusters
        self.sample_fstrategy = sample_strategy
        self.align=align
        self.recon=recon
        self.text_encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.text_encoder = MLPLayers(layers=self.text_encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.text_rq = ResidualVectorQuantizer(num_emb_list, e_dim, align=self.align,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)    
        self.text_decode_layer_dims = self.text_encode_layer_dims[::-1]
        self.text_decoder = MLPLayers(layers=self.text_decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn) 
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(kernels=[GaussianKernel(alpha=2 ** -1)])
    def forward(self, x, labels, use_sk=True):
        x = self.text_encoder(x)
        x_q,  rq_loss, indices = self.text_rq(x, labels, use_sk=use_sk)
        text_out = self.text_decoder(x_q)
        return text_out, rq_loss, indices, x_q
    
    def vq_initialization(self,x, use_sk=True):
        self.text_rq.vq_ini(self.text_encoder(x))

    @torch.no_grad()
    def get_indices(self, xs, labels, use_sk=False):
        x_e = self.text_encoder(xs)
        _, _, indices = self.text_rq(x_e, labels, use_sk=use_sk)
        return indices

    def compute_loss(self, text_out, quant_loss, xs):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(text_out, xs, reduction='mean')
        elif self.loss_type == 'mmd':
            loss_recon = self.mkmmd_loss(text_out, xs)
        else:
            raise ValueError('incompatible loss type')
        total_loss = self.recon * loss_recon + self.quant_loss_weight * quant_loss

        return total_loss, None, loss_recon, quant_loss
