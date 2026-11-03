import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import EmbDataset
from rqvae import DRQVAE
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Domain-adaptive Residual Quantization")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=2000)
    parser.add_argument("--learner", type=str, default="AdamW")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="mmd")
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--e_dim", type=int, default=128)
    parser.add_argument("--quant_loss_weight", type=float, default=0.1, help="lambda_com")
    parser.add_argument("--adapt_loss_weight", type=float, default=1e-4, help="lambda_adapt")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--n_domains", type=int, default=2)
    parser.add_argument("--layers", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--maxe", type=int, default=100)
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoint")
    parser.add_argument("--recon", type=float, default=1.0)
    parser.add_argument("--id_map", type=str, default="")
    parser.add_argument("--item_num_a", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print(args)
    logging.basicConfig(level=logging.DEBUG)

    data = EmbDataset(args.data_path, item_num_a=args.item_num_a, id_map=args.id_map or None)
    model = DRQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        rank=args.rank,
        n_domains=args.n_domains,
        adapt_loss_weight=args.adapt_loss_weight,
        recon=args.recon,
    )
    print(model)
    data_loader = DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    trainer = Trainer(args, model)
    best_loss, best_collision_rate = trainer.fit(data_loader)
    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)
