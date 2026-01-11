import collections
import os  # 新增引用
import numpy as np
import torch
from tqdm import tqdm
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import EmbDataset
from rqvae import RQVAE
import argparse


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")
    parser.add_argument("--dataset", type=str, default="Beauty", help="dataset")
    parser.add_argument(
        "--root_path", type=str, default="../checkpoint/", help="root path"
    )
    parser.add_argument("--alpha", type=str, default="1e-1", help="cf loss weight")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="epoch_9999_collision_0.0012_model.pth",
        help="checkpoint name",
    )
    parser.add_argument("--beta", type=str, default="1e-4", help="div loss weight")
    parser.add_argument("--data_path", type=str, default="../data", help="Text emb.")
    parser.add_argument(
        "--num_emb_list",
        type=int,
        nargs="+",
        default=[256, 256, 256, 256],
        help="emb num of every vq",
    )
    parser.add_argument(
        "--e_dim", type=int, default=32, help="rq codebook embedding size"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[2048, 1024, 512, 256, 128, 64],
        help="hidden sizes of every layer",
    )
    return parser.parse_args()


args_setting = parse_args()

# --- 修改开始：自动寻找最新时间戳文件夹 ---
root_dir = args_setting.root_path

if not os.path.exists(root_dir):
    raise FileNotFoundError(f"Root path does not exist: {root_dir}")

# 获取 root_path 下的所有子文件夹
subdirs = [
    os.path.join(root_dir, d) 
    for d in os.listdir(root_dir) 
    if os.path.isdir(os.path.join(root_dir, d))
]

if not subdirs:
    raise FileNotFoundError(f"No subdirectories (timestamp folders) found in {root_dir}")

# 按修改时间排序，取最新的文件夹 (例如 Dec-26-2025_20-45-40)
experiment_dir = max(subdirs, key=os.path.getmtime)
print(f"Selected latest experiment folder: {experiment_dir}")

dataset = args_setting.dataset

# 使用 os.path.join 拼接完整路径，兼容性更好
ckpt_path = os.path.join(experiment_dir, args_setting.checkpoint)
print(f"Checkpoint path: {ckpt_path}")

device = torch.device("cuda:0")

# 加载 Checkpoint
try:
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
except FileNotFoundError:
    raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")

args = ckpt["args"]
state_dict = ckpt["state_dict"]


data = EmbDataset(args.data_path)

model = RQVAE(
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
    sk_epsilons=args.sk_epsilons,
    sk_iters=args.sk_iters,
)

model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(
    data, num_workers=args.num_workers, batch_size=64, shuffle=False, pin_memory=True
)

all_indices = []
all_indices_str = []


def constrained_km(data, n_clusters=10):
    from k_means_constrained import KMeansConstrained

    x = data
    size_min = min(len(data) // (n_clusters * 2), 10)
    size_max = int(len(data) / n_clusters) + 1
    clf = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        max_iter=10,
        n_init=10,
        n_jobs=10,
        verbose=False,
    )
    clf.fit(x)
    t_centers = torch.from_numpy(clf.cluster_centers_)
    t_labels = torch.from_numpy(clf.labels_).tolist()
    return t_centers, t_labels


labels = {"0": [], "1": [], "2": [], "3": {}, "4": []}
embs = [
    layer.embedding.weight.cpu().detach().numpy() for layer in model.text_rq.vq_layers
]

for idx, emb in enumerate(embs):
    centers, label = constrained_km(emb)
    labels[str(idx)] = label

for d in tqdm(data_loader):
    text, emb_idx = d[0], d[1]
    text = text.to(device)

    indices = model.get_indices(text, labels, use_sk=False)

    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(int(ind))
        all_indices.append(code)
        all_indices_str.append(str(code))

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

# save SID
sid_save_path = os.path.join(experiment_dir, "SID.pt")
torch.save(
    all_indices,
    sid_save_path,
)
print(f"SID saved to: {sid_save_path}")

# save codebook embedding
emb_save_path = os.path.join(experiment_dir, "SID_emb.pt")
emb = model.text_rq.get_code()
torch.save(
    emb, emb_save_path
)
print(f"SID_emb saved to: {emb_save_path}")

print("正在生成并保存重构的 Embedding (Reconstructed Embeddings)...")

# 1. 准备数据
# all_indices 是 numpy array, shape (N, 4)
# 我们需要将其转回 tensor 放到 GPU 上以便进行向量化查找
indices_tensor = torch.from_numpy(all_indices).long().to(device)

# 2. 获取所有层的码本权重 (Codebook Weights)
# model.text_rq.vq_layers 是一个 ModuleList，每一层都有 .embedding.weight
codebooks = [layer.embedding.weight.data for layer in model.text_rq.vq_layers]

# 3. 逐级相加 (Residual Sum)
# 初始化全 0 向量，维度为 args.e_dim
reconstructed_emb = torch.zeros((len(all_indices), args.e_dim)).to(device)

for i, codebook_weight in enumerate(codebooks):
    # 取出当前层的索引: (N,)
    layer_indices = indices_tensor[:, i]

    # 查表: (N, e_dim)
    # F.embedding(input, weight) 比手动循环快得多
    layer_vectors = F.embedding(layer_indices, codebook_weight)

    # 累加 (Residual Sum)
    reconstructed_emb += layer_vectors

# 4. 转回 CPU numpy 格式
reconstructed_emb_np = reconstructed_emb.cpu().numpy()

print(f"生成的 Embedding 形状: {reconstructed_emb_np.shape}")

# 5. 保存为 pkl 文件
save_path = os.path.join(experiment_dir, "reconstructed_emb.pkl")

with open(save_path, "wb") as f:
    pickle.dump(reconstructed_emb_np, f)

print(f"文件已保存至: {save_path}")