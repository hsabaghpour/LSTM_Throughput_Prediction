# utils/dataset.py
import torch
from torch.utils.data import Dataset

class ThroughputDataset(Dataset):
    def __init__(self, data, window_size, target_size):
        self.data = data
        self.window_size = window_size
        self.target_size = target_size

    def __len__(self):
        return len(self.data) - self.window_size - self.target_size

    def __getitem__(self, index):
        x = self.data[index:index + self.window_size]
        y = self.data[index + self.window_size: index + self.window_size + self.target_size]
        return torch.FloatTensor(x), torch.FloatTensor(y)
