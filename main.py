import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW, Adam
import torchvision.transforms as transforms
from torchvision import datasets

from model_factory import ModelFactory
from tqdm import tqdm
import wandb

from torch.optim.lr_scheduler import CosineAnnealingLR

from features_dataset import FeaturesDataset  
from model import Net 

import numpy as np

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

import torch.nn.functional as F

def pixel_contrastive_loss(features1, features2, temperature=0.1):
    # features1, features2 : [batch_size, num_patches, feature_dim]
    batch_size, num_patches, feature_dim = features1.shape

    # Normaliser les features
    features1 = F.normalize(features1, dim=-1)
    features2 = F.normalize(features2, dim=-1)

    # Applatir les batchs et les patches
    features1 = features1.view(-1, feature_dim)  # [batch_size * num_patches, feature_dim]
    features2 = features2.view(-1, feature_dim)

    # Calculer les similarités
    logits = torch.mm(features1, features2.t()) / temperature  # [batch_size * num_patches, batch_size * num_patches]

    # Créer les labels
    labels = torch.arange(features1.size(0)).to(features1.device)

    # Calculer la perte
    loss = F.cross_entropy(logits, labels)

    return loss


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 training script")
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        metavar="WAPI",
        help="API key for wandb",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. train_images/ and val_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="experiment",
        metavar="E",
        help="folder where experiment outputs are located.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        metavar="NW",
        help="number of workers for data loading",
    )
    args = parser.parse_args()
    return args


# def train(
#     model: nn.Module,
#     optimizer: torch.optim.Optimizer,
#     scheduler: torch.optim.lr_scheduler._LRScheduler,
#     train_loader: torch.utils.data.DataLoader,
#     use_cuda: bool,
#     epoch: int,
#     args: argparse.ArgumentParser,
# ) -> None:
#     """Default Training Loop.

#     Args:
#         model (nn.Module): Model to train
#         optimizer (torch.optimizer): Optimizer to use
#         train_loader (torch.utils.data.DataLoader): Training data loader
#         use_cuda (bool): Whether to use cuda or not
#         epoch (int): Current epoch
#         args (argparse.ArgumentParser): Arguments parsed from command line
#     """
#     model.train()
#     correct = 0
#     total_loss = 0
#     total_samples = 0
#     # criterion = torch.nn.CrossEntropyLoss()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         if use_cuda:
#             data, target = data.cuda(), target.cuda()
        
#         # data, targets_a, targets_b, lam = mixup_data(data, target, alpha=1.0)

#         optimizer.zero_grad()
#         output = model(data)
#         criterion = torch.nn.CrossEntropyLoss(reduction="mean")
#         loss = criterion(output, target)
#         # loss = lam * criterion(output, targets_a) + (1 - lam) * criterion(output, targets_b)
#         loss.backward()
#         optimizer.step()
#         pred = output.data.max(1, keepdim=True)[1]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#         total_loss += loss.item() * data.size(0)
#         total_samples += data.size(0)
#         if batch_idx % args.log_interval == 0:
#             print(
#                 "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                     epoch,
#                     batch_idx * len(data),
#                     len(train_loader.dataset),
#                     100.0 * batch_idx / len(train_loader),
#                     loss.data.item(),
#                 )
#             )
#             # wandb.log({
#             #     "Train Loss": loss.item(),
#             #     "Train Accuracy": 100.0 * correct / total_samples,
#             #     "Epoch": epoch,
#             #     "Batch": batch_idx
#             # })
    
#     scheduler.step()
#     epoch_loss = total_loss / len(train_loader.dataset)
#     epoch_accuracy = 100.0 * correct / len(train_loader.dataset)
#     print(
#         "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
#             correct,
#             len(train_loader.dataset),
#             100.0 * correct / len(train_loader.dataset),
#         )
#     )
#     wandb.log({
#         "Train Epoch Loss": epoch_loss,
#         "Train Epoch Accuracy": epoch_accuracy,
#         "Learning Rate": scheduler.get_last_lr()[0],
#         "Epoch": epoch
#     })

def data_augmentation(images):
    # Définir les transformations
    augmentation_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        # Vous pouvez ajouter d'autres augmentations si nécessaire
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        # transforms.RandomRotation(15),
        # transforms.RandomErasing(p=0.1)

    ])
    # Appliquer les transformations
    augmented_images = []
    for img in images:
        img = transforms.ToPILImage()(img.cpu())
        img = augmentation_transforms(img)
        img = transforms.ToTensor()(img)
        augmented_images.append(img)
    augmented_images = torch.stack(augmented_images).to(images.device)
    return augmented_images


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    model.train()
    correct = 0
    total_loss = 0
    total_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Générer une vue augmentée des données
        data_aug = data_augmentation(data)  # Définir cette fonction pour appliquer des augmentations

        optimizer.zero_grad()

        # Passer les deux vues à travers le modèle
        logits, features = model(data)
        _, features_aug = model(data_aug)

        # Calculer la perte de classification
        criterion = torch.nn.CrossEntropyLoss()
        loss_cls = criterion(logits, target)

        # Calculer la perte contrastive
        loss_contrastive = pixel_contrastive_loss(features, features_aug)

        # Combiner les pertes
        loss = loss_cls + loss_contrastive

        loss.backward()
        optimizer.step()

        # Calculer la précision
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)

        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    scheduler.step()
    epoch_loss = total_loss / total_samples
    epoch_accuracy = 100.0 * correct / total_samples
    print(
        f"\nTrain set: Average loss: {epoch_loss:.4f}, Accuracy: {correct}/{total_samples} "
        f"({epoch_accuracy:.2f}%)\n"
    )
    wandb.log({
        "Train Epoch Loss": epoch_loss,
        "Train Epoch Accuracy": epoch_accuracy,
        "Learning Rate": scheduler.get_last_lr()[0],
        "Epoch": epoch
    })



# def validation(
#     model: nn.Module,
#     val_loader: torch.utils.data.DataLoader,
#     use_cuda: bool,
# ) -> float:
#     """Default Validation Loop.

#     Args:
#         model (nn.Module): Model to train
#         val_loader (torch.utils.data.DataLoader): Validation data loader
#         use_cuda (bool): Whether to use cuda or not

#     Returns:
#         float: Validation loss
#     """
#     model.eval()
#     validation_loss = 0
#     correct = 0
#     for data, target in val_loader:
#         if use_cuda:
#             data, target = data.cuda(), target.cuda()
#         output = model(data)
#         # sum up batch loss
#         criterion = torch.nn.CrossEntropyLoss(reduction="mean")
#         validation_loss += criterion(output, target).data.item()
#         # get the index of the max log-probability
#         pred = output.data.max(1, keepdim=True)[1]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     validation_loss /= len(val_loader.dataset)
#     accuracy = 100.0 * correct / len(val_loader.dataset)

#     print(
#         "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
#             validation_loss,
#             correct,
#             len(val_loader.dataset),
#             100.0 * correct / len(val_loader.dataset),
#         )
#     )

#     wandb.log({
#         "Validation Loss": validation_loss,
#         "Validation Accuracy": accuracy,
#     })
#     return validation_loss

def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    validation_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            logits, _ = model(data)
            loss = criterion(logits, target)
            validation_loss += loss.item() * data.size(0)
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(val_loader.dataset)
    accuracy = 100.0 * correct / len(val_loader.dataset)

    print(
        f"\nValidation set: Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} "
        f"({accuracy:.2f}%)\n"
    )

    wandb.log({
        "Validation Loss": validation_loss,
        "Validation Accuracy": accuracy,
    })
    return validation_loss

# def main():
#     args = opts()

#     if args.wandb_api_key:
#         wandb.login(key=args.wandb_api_key)
#     else:
#         wandb.login()

#     wandb.init(project='recvis2024', name="classifier_training", config=vars(args))

#     use_cuda = torch.cuda.is_available()
#     device = torch.device('cuda' if use_cuda else 'cpu')

#     torch.manual_seed(args.seed)

#     if not os.path.isdir(args.experiment):
#         os.makedirs(args.experiment)

#     # Charger le modèle (classifieur uniquement)
#     model = Net(input_dim=1024, num_classes=500)
#     model.to(device)

#     # Charger les datasets de features
#     train_dataset = FeaturesDataset(os.path.join(args.data, 'train_features.pth'))
#     val_dataset = FeaturesDataset(os.path.join(args.data, 'val_features.pth'))

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#     )

#     # Configurer l'optimiseur et le scheduler
#     optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
#     scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

#     # Boucle d'entraînement
#     best_val_loss = float('inf')
#     for epoch in range(1, args.epochs + 1):
#         # Entraînement
#         train(model, optimizer, scheduler, train_loader, use_cuda, epoch, args)
#         # Validation
#         val_loss = validation(model, val_loader, use_cuda)
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_file = os.path.join(args.experiment, "model_best.pth")
#             torch.save(model.state_dict(), best_model_file)
#         # Sauvegarder le modèle à chaque époque (optionnel)
#         if epoch % 5 == 0:
#             model_file = os.path.join(args.experiment, f"model_{epoch}.pth")
#             torch.save(model.state_dict(), model_file)
#             print(f"Saved model to {model_file}")

#     wandb.finish()



def main():
    args = opts()

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    else:
        wandb.login()

    wandb.init(project='recvis2024', name="contrastive_learning", config=vars(args))

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)

    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Charger le modèle
    model = Net(num_classes=500)
    model.to(device)

    # Définir les transformations de données
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(15),
        # AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        # transforms.GaussianBlur(kernel_size=1),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(p=0.1)
    ])

    data_transforms_val = transforms.Compose([
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
    val_dataset = datasets.ImageFolder(os.path.join(args.data, "val_images"), transform=data_transforms_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Configurer l'optimiseur et le scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Boucle d'entraînement
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        # Entraînement
        train(model, optimizer, scheduler, train_loader, device, epoch, args)
        # Validation
        val_loss = validation(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_file = os.path.join(args.experiment, "model_best.pth")
            torch.save(model.state_dict(), best_model_file)
        # Sauvegarder le modèle à chaque époque (optionnel)
        if epoch % 5 == 0:
            model_file = os.path.join(args.experiment, f"model_{epoch}.pth")
            torch.save(model.state_dict(), model_file)
            print(f"Saved model to {model_file}")

    wandb.finish()


# def main():
#     """Default Main Function."""
#     # options
#     args = opts()

#     if args.wandb_api_key:
#         wandb.login(key=args.wandb_api_key)
#     else:
#         wandb.login()

#     wandb.init(project='recvis2024', name="CLIP-ViT-H-14-laion2B-s32B-b79K_adamw_betas9e-1n9_8e-1_eps1e-6_weightdecay_2e-1_with_feature_label", config=vars(args))

#     # Check if cuda is available
#     use_cuda = torch.cuda.is_available()

#     # Set the seed (for reproducibility)
#     torch.manual_seed(args.seed)

#     # Create experiment folder
#     if not os.path.isdir(args.experiment):
#         os.makedirs(args.experiment)

#     # load model and transform
#     model, data_transforms, val_transforms = ModelFactory(args.model_name).get_all()
#     if use_cuda:
#         print("Using GPU")
#         model.cuda()
#     else:
#         print("Using CPU")

#     # Data initialization and loading
#     train_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#     )
#     val_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(args.data + "/val_images", transform=val_transforms),
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#     )

#     # Setup optimizer
#     # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
#     # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
#     optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9,0.98),eps=1e-6, weight_decay=0.2)

#     scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

#     # Loop over the epochs
#     best_val_loss = 1e8
#     for epoch in range(1, args.epochs + 1):
#         # training loop
#         # train(model, optimizer, train_loader, use_cuda, epoch, args)
#         train(model, optimizer, scheduler, train_loader, use_cuda, epoch, args)
#         # validation loop
#         val_loss = validation(model, val_loader, use_cuda)
#         if val_loss < best_val_loss:
#             # save the best model for validation
#             best_val_loss = val_loss
#             best_model_file = args.experiment + "/model_best.pth"
#             torch.save(model.state_dict(), best_model_file)
#         # also save the model every epoch
#         model_file = args.experiment + "/model_" + str(epoch) + ".pth"

#         if epoch % 5 == 0:
#             torch.save(model.state_dict(), model_file)
#             print(
#                 "Saved model to "
#                 + model_file
#                 + f". You can run `python evaluate.py --model_name {args.model_name} --model "
#                 + best_model_file
#                 + "` to generate the Kaggle formatted csv file\n"
#             )

#         # torch.save(model.state_dict(), model_file)
#         # print(
#         #     "Saved model to "
#         #     + model_file
#         #     + f". You can run `python evaluate.py --model_name {args.model_name} --model "
#         #     + best_model_file
#         #     + "` to generate the Kaggle formatted csv file\n"
#         # )

#     wandb.finish()


if __name__ == "__main__":
    main()
