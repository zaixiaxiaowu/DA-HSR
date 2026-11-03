import logging
import os
from time import time

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from datasets import EmbDataset
from torch.utils.data import DataLoader
from utils import ensure_dir, get_local_time, set_color


class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()
        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.device = torch.device(args.device)
        self.ckpt_dir = os.path.join(args.ckpt_dir, "{}".format(get_local_time()))
        ensure_dir(self.ckpt_dir)
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.maxe = args.maxe

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = self.learner.lower()
        if learner == "adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "sgd":
            return optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "adamw":
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        self.logger.warning("Unrecognized optimizer, fallback to Adam")
        return optim.Adam(params, lr=self.lr)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.args.data_path, id_map=self.args.id_map or None, item_num_a=self.args.item_num_a)
        init_loader = DataLoader(
            original_data,
            num_workers=self.args.num_workers,
            batch_size=len(original_data),
            shuffle=True,
            pin_memory=True,
        )
        for data in tqdm(init_loader, desc=set_color("Initialization of vq", "pink")):
            text = data[0].to(self.device)
            self.model.vq_initialization(text)

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0
        iter_data = tqdm(train_data, total=len(train_data), ncols=100, desc=set_color(f"Train {epoch_idx}", "pink"))

        for data in iter_data:
            text, _, domains = data
            text = text.to(self.device)
            domains = domains.to(self.device)
            self.optimizer.zero_grad()
            decoded, commit_loss, adapt_loss, _, _ = self.model(text, domains)
            loss, loss_recon, quant_loss, _ = self.model.compute_loss(decoded, commit_loss, adapt_loss, text)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_quant_loss += quant_loss.item()

        return total_loss, total_recon_loss, total_quant_loss

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file) if ckpt_file else os.path.join(
            self.ckpt_dir, "epoch_%d_collision_%.4f_model.pth" % (epoch, collision_rate)
        )
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)
        self.logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        output = (set_color("epoch %d training", "green") + " [" + set_color("time", "blue") + ": %.2fs, ") % (
            epoch_idx,
            e_time - s_time,
        )
        output += set_color("train loss", "blue") + ": %.4f, " % loss
        output += set_color("reconstruction loss", "blue") + ": %.4f]" % recon_loss
        return output

    def fit(self, data):
        cur_eval_step = 0
        self.vq_init()
        for epoch_idx in range(self.epochs):
            start = time()
            train_loss, train_recon_loss, _ = self._train_epoch(data, epoch_idx)
            end = time()
            self.logger.info(self._generate_train_loss_output(epoch_idx, start, end, train_loss, train_recon_loss))
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)
                cur_eval_step = 0
            else:
                cur_eval_step += 1
                if cur_eval_step >= self.maxe:
                    print(os.path.join(self.ckpt_dir, self.best_loss_ckpt))
                    print("Finish!")
                    break
        return self.best_loss, self.best_collision_rate
