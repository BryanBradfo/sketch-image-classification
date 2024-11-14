# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models
# import torch


# nclasses = 500


# # class Net(nn.Module):
# #     def __init__(self):
# #         super(Net, self).__init__()
# #         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
# #         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
# #         self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
# #         self.fc1 = nn.Linear(320, 50)
# #         self.fc2 = nn.Linear(50, nclasses)

# #     def forward(self, x):
# #         x = F.relu(F.max_pool2d(self.conv1(x), 2))
# #         x = F.relu(F.max_pool2d(self.conv2(x), 2))
# #         x = F.relu(F.max_pool2d(self.conv3(x), 2))
# #         x = x.view(-1, 320)
# #         x = F.relu(self.fc1(x))
# #         return self.fc2(x)

# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         self.model = models.efficientnet_v2_s(weights='DEFAULT')

#         n_freeze = 0
#         for i, block in enumerate(self.model.features):
#             if i < n_freeze:
#                 for param in block.parameters():
#                     param.requires_grad = False
                
#         self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, nclasses)

#     def forward(self, x):
#         x = self.model(x)
#         return x


import torch.nn as nn
import open_clip

nclasses = 500

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        # self.model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='openai')
        num_features = self.model.visual.output_dim
        self.classifier = nn.Linear(num_features, nclasses)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model.encode_image(x)
        x = self.classifier(x)
        return x
