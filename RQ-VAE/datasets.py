import json
import os
import pickle
import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):
    def __init__(self, data_path, item_num_a=None, id_map=None):
        with open(data_path, "rb") as handle:
            self.text = pickle.load(handle)
        self.text = np.asarray(self.text)
        self.dim = self.text.shape[-1]

        if id_map is None:
            candidate = os.path.join(os.path.dirname(data_path), "id_map.json")
            if os.path.exists(candidate):
                id_map = candidate
        if id_map is not None and item_num_a is None:
            mapping = json.load(open(id_map, "r", encoding="utf-8"))
            item_num_a = mapping["item_dict"]["item_count"]["0"]

        n_items = len(self.text)
        if item_num_a is None:
            self.domains = np.zeros(n_items, dtype=np.int64)
        else:
            item_num_a = int(item_num_a)
            self.domains = np.concatenate(
                [
                    np.zeros(item_num_a, dtype=np.int64),
                    np.ones(max(n_items - item_num_a, 0), dtype=np.int64),
                ]
            )[:n_items]

    def __getitem__(self, index):
        return torch.FloatTensor(self.text[index]), index, int(self.domains[index])

    def __len__(self):
        return len(self.text)
