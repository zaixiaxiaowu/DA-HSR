import numpy as np
import torch
import torch.utils.data as data
import pickle

class EmbDataset(data.Dataset):

    def __init__(self,data_path):
        with open(data_path, 'rb') as f:
            self.text = pickle.load(f)
        self.dim = self.text.shape[-1]

    def __getitem__(self, index):
        text = self.text[index]
        return torch.FloatTensor(text), index

    def __len__(self):
        return len(self.text)
