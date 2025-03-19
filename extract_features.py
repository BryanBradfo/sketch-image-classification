import argparse
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import timm

def opts():
    parser = argparse.ArgumentParser(description='Extract features using EVA-CLIP')
    parser.add_argument('--data', type=str, default='data_sketches', metavar='D',
                        help='Dossier où les données sont situées. train_images/ et val_images/ doivent s\'y trouver')
    parser.add_argument('--output_dir', type=str, default='.', metavar='OUT',
                        help='Dossier de sortie pour sauvegarder les caractéristiques')
    parser.add_argument('--batch_size', type=int, default=64, metavar='B',
                        help='Taille de lot pour le traitement des images')
    parser.add_argument('--num_workers', type=int, default=4, metavar='NW',
                        help='Nombre de workers pour le chargement des données')
    args = parser.parse_args()
    return args

def main():
    args = opts()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Utilisation du dispositif : {device}')

    # model = timm.create_model('eva_giant_patch14_336', pretrained=True)
    model = timm.create_model('eva_giant_patch14_336.clip_ft_in1k', pretrained=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)

    # data_config = timm.data.resolve_model_data_config(model)
    # timm_data_transforms = timm.data.create_transform(**data_config, is_training=False)

    # default_cfg = model.default_cfg

    # data_transforms = transforms.Compose([
    #     transforms.Resize(default_cfg['input_size'][-2:]),
    #     transforms.CenterCrop(default_cfg['input_size'][-2:]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=default_cfg['mean'],
    #         std=default_cfg['std']
    #     ),
    # ])

    data_transforms = transforms.Compose([
        transforms.Resize(336),
        transforms.RandomResizedCrop(336),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    data_transforms_val = transforms.Compose([
        transforms.Resize(336),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    print(os.path.join(args.data, 'train_images'))
    train_dataset = datasets.ImageFolder(os.path.join(args.data, "train_images"), transform=data_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, "val_images"), transform=data_transforms_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Extraction des caractéristiques des données d\'entraînement...')
    train_features, train_labels = extract_features(model, train_loader, device)
    print('Extraction des caractéristiques des données de validation...')
    val_features, val_labels = extract_features(model, val_loader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    train_output_file = os.path.join(args.output_dir, 'train_features.pt')
    val_output_file = os.path.join(args.output_dir, 'val_features.pt')

    torch.save({'image_features': train_features, 'labels': train_labels}, train_output_file)
    torch.save({'image_features': val_features, 'labels': val_labels}, val_output_file)

    print(f'Caractéristiques d\'entraînement sauvegardées dans {train_output_file}')
    print(f'Caractéristiques de validation sauvegardées dans {val_output_file}')

def extract_features(model, data_loader, device):
    features_list = []
    labels_list = []

    with torch.no_grad():
        for data, labels in tqdm(data_loader):
            data = data.to(device)
            labels = labels.to(device)

            features = model.forward_features(data)
            features = model.forward_head(features, pre_logits=True)

            features_list.append(features.cpu())
            labels_list.append(labels.cpu())

    features = torch.cat(features_list)
    labels = torch.cat(labels_list)

    print(f'Caractéristiques extraites de forme : {features.shape}')
    return features, labels

if __name__ == '__main__':
    main()
