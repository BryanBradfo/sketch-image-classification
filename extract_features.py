import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import CLIPModel
from torch.utils.data import DataLoader
from tqdm import tqdm

def opts():
    parser = argparse.ArgumentParser(description="Script d'extraction de features")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="Dossier où les données sont situées. Les dossiers train_images/ et val_images/ doivent s'y trouver.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Taille de batch pour l'extraction des features (par défaut : 64)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="NW",
        help="Nombre de workers pour le chargement des données",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="features",
        metavar="O",
        help="Répertoire où sauvegarder les features extraites",
    )
    args = parser.parse_args()
    return args

def main():
    args = opts()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Charger le modèle CLIP pré-entraîné
    model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model.eval()
    model.to(device)

    # Geler tous les paramètres
    for param in model.parameters():
        param.requires_grad = False

    # Définit les transformations de données (sans augmentations)
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    # Charger les datasets
    train_dataset = datasets.ImageFolder(os.path.join(args.data, "train_images"), transform=data_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, "val_images"), transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Extraire les features et labels pour le jeu d'entraînement
    print("Extraction des features pour le jeu d'entraînement...")
    train_features = []
    train_labels = []

    with torch.no_grad():
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            features = model.get_image_features(pixel_values=images)
            features = features.cpu()
            train_features.append(features)
            train_labels.append(labels)

    # Concaténer toutes les features et labels
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Sauvegarder les features et labels
    torch.save({'features': train_features, 'labels': train_labels}, os.path.join(args.output_dir, 'train_features.pth'))

    # Extraire les features et labels pour le jeu de validation
    print("Extraction des features pour le jeu de validation...")
    val_features = []
    val_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            features = model.get_image_features(pixel_values=images)
            features = features.cpu()
            val_features.append(features)
            val_labels.append(labels)

    # Concaténer toutes les features et labels
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    # Sauvegarder les features et labels
    torch.save({'features': val_features, 'labels': val_labels}, os.path.join(args.output_dir, 'val_features.pth'))

    print("Extraction des features terminée.")

if __name__ == '__main__':
    main()
