# features_dataset.py

import torch
from torch.utils.data import Dataset

class FeaturesDataset(Dataset):
    def __init__(self, features_file):
        data = torch.load(features_file)
        self.features = data['features']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label
