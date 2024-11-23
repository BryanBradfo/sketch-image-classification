# features_dataset.py

import torch
from torch.utils.data import Dataset

class FeaturesDataset(Dataset):
    def __init__(self, features_file):
        data = torch.load(features_file)
        self.features = data['image_features']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

# #######################################################
# ### Using feature labels considering text labels
# #######################################################

# import torch
# from torch.utils.data import Dataset

# class FeaturesDataset(Dataset):
#     def __init__(self, features_file, class_text_features_file):
#         data = torch.load(features_file)
#         self.image_features = data['image_features']
#         self.labels = data['labels']
#         class_text_data = torch.load(class_text_features_file)
#         self.class_text_features = class_text_data['class_text_features']

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         image_feature = self.image_features[idx]
#         label = self.labels[idx]
#         text_feature = self.class_text_features[label]
#         return image_feature, text_feature, label
