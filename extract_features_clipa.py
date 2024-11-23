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
        default="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",
        required=True,
        help="Hugging Face token. Generate it from https://huggingface.co/settings/tokens"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/kaggle/input/mva-recvis-2024/sketch_recvis2024/sketch_recvis2024",
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

def main():
    args = opts()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Charger le modèle CoCa et les transformations
    model = load_eva_clip_model("/kaggle/working/EVA02_CLIP_L_336_psz14_s6B.pt", device) 
    model.eval()

    # Geler les paramètres du modèle
    for param in model.parameters():
        param.requires_grad = False

    # Appliquer les transformations du modèle CoCa lors du chargement des données
    train_dataset = datasets.ImageFolder(os.path.join(args.data, "train_images"), transform=None)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, "val_images"), transform=None)

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