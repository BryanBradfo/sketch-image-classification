# import argparse
# import os
# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from transformers import CLIPModel, CLIPProcessor
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# def opts():
#     parser = argparse.ArgumentParser(description="Extraction of features")
#     parser.add_argument(
#         "--data",
#         type=str,
#         default="data_sketches",
#         metavar="D",
#         help="Folder where data are stored. The folder train_images/ and val_images/ should be there.",
#     )
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=64,
#         metavar="B",
#         help="Size of batch for the extraction of features (default : 64)",
#     )
#     parser.add_argument(
#         "--num_workers",
#         type=int,
#         default=4,
#         metavar="NW",
#         help="Numbers of workers for data laoding",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="features",
#         metavar="O",
#         help="Path where to save the existinig features",
#     )
#     args = parser.parse_args()
#     return args

# def main():
#     args = opts()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if not os.path.isdir(args.output_dir):
#         os.makedirs(args.output_dir)

#     model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
#     processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

#     model.eval()
#     model.to(device)

#     for param in model.parameters():
#         param.requires_grad = False

#     data_transforms = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466, 0.4578275, 0.40821073],
#             std=[0.26862954, 0.26130258, 0.27577711]
#         ),
#     ])

#     data_transforms_val = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.48145466, 0.4578275, 0.40821073],
#             std=[0.26862954, 0.26130258, 0.27577711]
#         ),
#     ])

#     train_dataset = datasets.ImageFolder(os.path.join(args.data, "train_images"), transform=data_transforms)
#     val_dataset = datasets.ImageFolder(os.path.join(args.data, "val_images"), transform=data_transforms_val)

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     print("Extraction of features for the training dataset...")
#     train_features = []
#     train_labels = []

#     with torch.no_grad():
#         for images, labels in tqdm(train_loader):
#             images = images.to(device)
#             images = (images - images.min()) / (images.max() - images.min())
#             inputs = processor(images=images, return_tensors="pt", padding=True)
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             features = model.get_image_features(**inputs)
#             # features = model.get_image_features(pixel_values=images)
#             features = features.cpu()
#             train_features.append(features)
#             train_labels.append(labels)

#     train_features = torch.cat(train_features, dim=0)
#     train_labels = torch.cat(train_labels, dim=0)

#     torch.save({'features': train_features, 'labels': train_labels}, os.path.join(args.output_dir, 'train_features.pth'))

#     print("Extraction of features for the validation dataset...")
#     val_features = []
#     val_labels = []

#     with torch.no_grad():
#         for images, labels in tqdm(val_loader):
#             images = images.to(device)
#             images = (images - images.min()) / (images.max() - images.min())
#             inputs = processor(images=images, return_tensors="pt", padding=True)
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             features = model.get_image_features(**inputs)
#             # features = model.get_image_features(pixel_values=images)
#             features = features.cpu()
#             val_features.append(features)
#             val_labels.append(labels)

#     val_features = torch.cat(val_features, dim=0)
#     val_labels = torch.cat(val_labels, dim=0)

#     torch.save({'features': val_features, 'labels': val_labels}, os.path.join(args.output_dir, 'val_features.pth'))

#     print("Extraction of features is done.")

# if __name__ == '__main__':
#     main()


import argparse
import os
import torch
from torchvision import datasets
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
from huggingface_hub import login

def connect_to_huggingface(token: str):
    """
    Connect to Hugging Face using the provided token.
    """
    try:
        login(token=token)
        print("Successfully connected to Hugging Face!")
    except Exception as e:
        print(f"Error during login: {e}")

def opts():
    parser = argparse.ArgumentParser(description="Extraction of features")
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="Hugging Face token. Generate it from https://huggingface.co/settings/tokens"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="Folder where data are stored. The folder train_images/ and val_images/ should be there.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Size of batch for the extraction of features (default : 64)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="NW",
        help="Numbers of workers for data loading",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="features",
        metavar="O",
        help="Path where to save the existing features",
    )
    args = parser.parse_args()
    connect_to_huggingface(args.hf_token)
    return args

def custom_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

# def main():
#     args = opts()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if not os.path.isdir(args.output_dir):
#         os.makedirs(args.output_dir)

#     # Charger le modèle CLIP et le processor
#     # model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
#     # processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

#     model = CLIPModel.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
#     processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-g-14-laion2B-s12B-b42K")

#     model.eval()
#     model.to(device)

#     # Geler les paramètres du modèle
#     for param in model.parameters():
#         param.requires_grad = False

#     # Ne pas appliquer de transformations lors du chargement des données
#     train_dataset = datasets.ImageFolder(os.path.join(args.data, "train_images"), transform=None)
#     val_dataset = datasets.ImageFolder(os.path.join(args.data, "val_images"), transform=None)

#     # Utiliser la fonction de collation personnalisée
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         collate_fn=custom_collate_fn
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         collate_fn=custom_collate_fn
#     )

#     # Obtenir les noms des classes
#     class_names = train_dataset.classes

#     # Pré-calculer les features textuelles pour chaque classe
#     with torch.no_grad():
#         text_inputs = processor(text=class_names, return_tensors="pt", padding=True)
#         text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
#         class_text_features = model.get_text_features(**text_inputs)
#         class_text_features = class_text_features.cpu()

#     print("Extraction of features for the training dataset...")
#     train_image_features = []
#     train_labels = []

#     with torch.no_grad():
#         for images, labels in tqdm(train_loader):
#             # Appliquer le processor aux images
#             inputs = processor(images=images, return_tensors="pt", padding=True)
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             image_features = model.get_image_features(**inputs)
#             image_features = image_features.cpu()
#             train_image_features.append(image_features)
#             train_labels.append(labels)

#     train_image_features = torch.cat(train_image_features, dim=0)
#     train_labels = torch.cat(train_labels, dim=0)

#     # Sauvegarder les features et labels
#     torch.save({'image_features': train_image_features, 'labels': train_labels}, os.path.join(args.output_dir, 'train_features.pth'))

#     # Sauvegarder les features textuelles des classes
#     torch.save({'class_text_features': class_text_features}, os.path.join(args.output_dir, 'class_text_features.pth'))

#     print("Extraction of features for the validation dataset...")
#     val_image_features = []
#     val_labels = []

#     with torch.no_grad():
#         for images, labels in tqdm(val_loader):
#             # Appliquer le processor aux images
#             inputs = processor(images=images, return_tensors="pt", padding=True)
#             inputs = {k: v.to(device) for k, v in inputs.items()}
#             image_features = model.get_image_features(**inputs)
#             image_features = image_features.cpu()
#             val_image_features.append(image_features)
#             val_labels.append(labels)

#     val_image_features = torch.cat(val_image_features, dim=0)
#     val_labels = torch.cat(val_labels, dim=0)

#     # Sauvegarder les features et labels
#     torch.save({'image_features': val_image_features, 'labels': val_labels}, os.path.join(args.output_dir, 'val_features.pth'))

#     print("Extraction of features is done.")

# if __name__ == '__main__':
#     main()

############################################################################################################
### EXTRACT FEATURES WITH CoCA
############################################################################################################

# import open_clip

from open_clip import create_model_from_pretrained, get_tokenizer

def main():
    args = opts()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Charger le modèle CoCa et les transformations
    model = CLIPModel.from_pretrained("QuanSun/EVA-CLIP", model_name="EVA02_CLIP_L_psz14_224to336")
    preprocess = CLIPProcessor.from_pretrained("QuanSun/EVA-CLIP", model_name="EVA02_CLIP_L_psz14_224to336")

    # model = CLIPModel.from_pretrained("UCSC-VLAA/ViT-H-14-CLIPA-336-laion2B")
    # preprocess = CLIPProcessor.from_pretrained("UCSC-VLAA/ViT-H-14-CLIPA-336-laion2B")

    # model, preprocess = create_model_from_pretrained('hf-hub:ViT-H-14-CLIPA-336')
    # tokenizer = get_tokenizer('hf-hub:ViT-H-14-CLIPA-336')

    # model, _, preprocess = open_clip.create_model_and_transforms(
    #     'hf-hub:laion/CoCa-ViT-L-14-laion2B-s13B-b90k',
    #     pretrained='laion2B-s13B-b90k'
    # )
    # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CoCa-ViT-L-14-laion2B-s13B-b90k')
    # model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CoCa-ViT-L-14-laion2B-s13B-b90k', pretrained='laion2B-s13B-b90k')
    # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CoCa-ViT-L-14-laion2B-s13B-b90k')

    model.eval()
    model.to(device)

    # Geler les paramètres du modèle
    for param in model.parameters():
        param.requires_grad = False

    # Appliquer les transformations du modèle CoCa lors du chargement des données
    train_dataset = datasets.ImageFolder(os.path.join(args.data, "train_images"), transform=preprocess)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, "val_images"), transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Obtenir les noms des classes
    class_names = train_dataset.classes

    # Pré-calculer les features textuelles pour chaque classe
    # with torch.no_grad():
    #     text_inputs = tokenizer(class_names)
    #     text_inputs = text_inputs.to(device)
    #     class_text_features = model.encode_text(text_inputs)
    #     class_text_features = class_text_features.cpu()

    # Extraction des features pour le jeu d'entraînement
    print("Extraction des features pour le jeu d'entraînement...")
    train_image_features = []
    train_labels = []

    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = image_features.cpu()
            train_image_features.append(image_features)
            train_labels.append(labels)

    train_image_features = torch.cat(train_image_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Sauvegarder les features et labels
    torch.save({'image_features': train_image_features, 'labels': train_labels}, os.path.join(args.output_dir, 'train_features.pth'))
    # torch.save({'class_text_features': class_text_features}, os.path.join(args.output_dir, 'class_text_features.pth'))

    print("Extraction of features for the validation dataset...")
    val_image_features = []
    val_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            # Appliquer le processor aux images
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = image_features.cpu()
            val_image_features.append(image_features)
            val_labels.append(labels)

    val_image_features = torch.cat(val_image_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    # Sauvegarder les features et labels
    torch.save({'image_features': val_image_features, 'labels': val_labels}, os.path.join(args.output_dir, 'val_features.pth'))

    print("Extraction of features is done.")

if __name__ == '__main__':
    main()