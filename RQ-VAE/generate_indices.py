import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from datasets import EmbDataset
from rqvae import DRQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Export DRQ layer-wise codes")
    parser.add_argument("--dataset", type=str, default="cloth")
    parser.add_argument("--root_path", type=str, default="../checkpoint/")
    parser.add_argument("--checkpoint", type=str, default="best_loss_model.pth")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--e_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--id_map", type=str, default="")
    parser.add_argument("--item_num_a", type=int, default=None)
    return parser.parse_args()


args_setting = parse_args()
root_dir = args_setting.root_path
if not os.path.exists(root_dir):
    raise FileNotFoundError("Root path does not exist: {}".format(root_dir))

subdirs = [
    os.path.join(root_dir, d)
    for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
]
if not subdirs:
    raise FileNotFoundError("No timestamp folders found in {}".format(root_dir))

experiment_dir = max(subdirs, key=os.path.getmtime)
ckpt_path = os.path.join(experiment_dir, args_setting.checkpoint)
print("Selected experiment folder:", experiment_dir)
print("Checkpoint path:", ckpt_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ckpt_path, map_location="cpu")
args = ckpt["args"]
state_dict = ckpt["state_dict"]

data = EmbDataset(
    args_setting.data_path,
    item_num_a=args_setting.item_num_a,
    id_map=args_setting.id_map or None,
)
model = DRQVAE(
    in_dim=data.dim,
    num_emb_list=getattr(args, "num_emb_list", args_setting.num_emb_list),
    e_dim=getattr(args, "e_dim", args_setting.e_dim),
    layers=getattr(args, "layers", args_setting.layers),
    dropout_prob=getattr(args, "dropout_prob", 0.0),
    bn=getattr(args, "bn", False),
    loss_type=getattr(args, "loss_type", "mmd"),
    quant_loss_weight=getattr(args, "quant_loss_weight", 0.1),
    kmeans_init=getattr(args, "kmeans_init", True),
    kmeans_iters=getattr(args, "kmeans_iters", 100),
    rank=getattr(args, "rank", 16),
    n_domains=getattr(args, "n_domains", 2),
    adapt_loss_weight=getattr(args, "adapt_loss_weight", 1e-4),
    recon=getattr(args, "recon", 1.0),
)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()

loader = DataLoader(data, num_workers=4, batch_size=64, shuffle=False, pin_memory=True)
all_indices = []
all_layer_codes = []

for batch in tqdm(loader):
    text, _, domains = batch
    text = text.to(device)
    domains = domains.to(device)
    indices, layer_codes = model.get_indices(text, domains)
    all_indices.append(indices.detach().cpu())
    all_layer_codes.append(layer_codes.detach().cpu())

all_indices = torch.cat(all_indices, dim=0).numpy()
all_layer_codes = torch.cat(all_layer_codes, dim=0).numpy()
reconstructed = all_layer_codes.sum(axis=1)

handled_dir = os.path.dirname(args_setting.data_path)
sid_save_path = os.path.join(experiment_dir, "SID.pt")
layer_save_path = os.path.join(experiment_dir, "drq_layer_embs.pkl")
recon_save_path = os.path.join(experiment_dir, "reconstructed_emb.pkl")
handled_layer_path = os.path.join(handled_dir, "drq_layer_embs.pkl")
handled_recon_path = os.path.join(handled_dir, "reconstructed_emb.pkl")

torch.save(all_indices, sid_save_path)
with open(layer_save_path, "wb") as handle:
    pickle.dump(all_layer_codes, handle)
with open(recon_save_path, "wb") as handle:
    pickle.dump(reconstructed, handle)
with open(handled_layer_path, "wb") as handle:
    pickle.dump(all_layer_codes, handle)
with open(handled_recon_path, "wb") as handle:
    pickle.dump(reconstructed, handle)

print("SID saved to:", sid_save_path)
print("Layer-wise DRQ embeddings:", all_layer_codes.shape, "->", handled_layer_path)
print("Reconstructed embeddings:", reconstructed.shape, "->", handled_recon_path)
