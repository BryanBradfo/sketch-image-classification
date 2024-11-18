import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
from transformers import CLIPModel, CLIPProcessor, SwinModel, ViTModel, CLIPModel


nclasses = 500

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, nclasses)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

#######################################################
### EfficientNet
#######################################################

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.model = models.efficientnet_v2_l(weights='DEFAULT')

#         n_freeze = 0
#         for i, block in enumerate(self.model.features):
#             if i < n_freeze:
#                 for param in block.parameters():
#                     param.requires_grad = False
                
#         self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, nclasses)

#     def forward(self, x):
#         x = self.model(x)
#         return x

#######################################################
### CLIP OpenAI
#######################################################

# class Net(nn.Module):
#     def __init__(self, num_classes=500):
#         super(Net, self).__init__()
#         self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
#         for param in self.clip_model.parameters():
#             param.requires_grad = False
#         for param in self.clip_model.vision_model.encoder.layers[-4:].parameters():
#             param.requires_grad = True
#         image_embed_dim = self.clip_model.visual_projection.in_features
#         self.classifier = nn.Sequential(
#             nn.Linear(image_embed_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(1024, num_classes)
#         )

#         # self.classifier = nn.Sequential(
#         #     nn.Linear(image_embed_dim, 2048),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.5),
#         #     nn.Linear(2048, 1024),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.5),
#         #     nn.Linear(1024, num_classes)
#         # )

#     def forward(self, x):
#         image_features = self.clip_model.vision_model(pixel_values=x).pooler_output
#         logits = self.classifier(image_features)
#         return logits


#######################################################
class Net(nn.Module):
    def __init__(self, num_classes=500):
        super(Net, self).__init__()
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.vision_model.encoder.layers[-3:].parameters():
            param.requires_grad = True
        
        image_embed_dim = self.model.visual_projection.in_features
        
        self.classifier = nn.Sequential(
            nn.Linear(image_embed_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        image_features = self.model.get_image_features(pixel_values=x)
        # image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        logits = self.classifier(image_features)
        return logits


#######################################################
### OpenCLIP LAIONB2B ViT-G (too big)
#######################################################

# import open_clip

# class Net(nn.Module):
#     def __init__(self, num_classes=500):
#         super(Net, self).__init__()
#         self.model, _, _ = open_clip.create_model_and_transforms(
#             'ViT-g-14', pretrained='laion2b_s12b_b42k'
#         )
#         for param in self.model.parameters():
#             param.requires_grad = False
#         image_embed_dim = self.model.visual.output_dim
#         self.classifier = nn.Sequential(
#             nn.Linear(image_embed_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(1024, num_classes)
#         )

#     def forward(self, x):
#         image_features = self.model.encode_image(x)
#         logits = self.classifier(image_features)
#         return logits


#######################################################
### ViT Huge
#######################################################

# class Net(nn.Module):
#     def __init__(self, num_classes=500):
#         super(Net, self).__init__()
#         self.model = ViTModel.from_pretrained('google/vit-huge-patch14-224-in21k')

#         for param in self.model.parameters():
#             param.requires_grad = False
#         image_embed_dim = self.model.config.hidden_size
#         self.classifier = nn.Sequential(
#             nn.Linear(image_embed_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(1024, num_classes)
#         )

#     def forward(self, x):
#         outputs = self.model(pixel_values=x)
#         pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
#         logits = self.classifier(pooled_output)
#         return logits

#######################################################
### Swin Transformer Large
#######################################################

# class Net(nn.Module):
#     def __init__(self, num_classes=500):
#         super(Net, self).__init__()
#         self.model = SwinModel.from_pretrained('microsoft/swin-large-patch4-window7-224-in22k')
#         for param in self.model.parameters():
#             param.requires_grad = False
#         image_embed_dim = self.model.config.hidden_size
#         self.classifier = nn.Sequential(
#             nn.Linear(image_embed_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(1024, num_classes)
#         )

#     def forward(self, x):
#         outputs = self.model(pixel_values=x)
#         pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
#         logits = self.classifier(pooled_output)
#         return logits

