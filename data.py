import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet
# data_transforms = transforms.Compose(
#     [
#         transforms.Resize((64, 64)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )

from torchvision import transforms

# Transformations pour l'entraînement
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

# Transformations pour la validation et le test
data_transforms_val = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


# data_train_transforms = transforms.Compose(
#     [
#         transforms.Resize((64, 64)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )





# import open_clip

# # _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

# _, _, preprocess = open_clip.create_model_and_transforms('ViT-H/14-quickgelu', pretrained='openai')
# # data_transforms = preprocess

# data_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=preprocess.transforms[-1].mean, std=preprocess.transforms[-1].std),
# ])