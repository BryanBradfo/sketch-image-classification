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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch.nn.functional as F

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
    parser.add_argument(
        "--image_folder",
        type=str,
        default="data_sketches",
        metavar="IF",
        help="folder where original images are located",
    )
    args = parser.parse_args()
    return args

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
    epoch: int,
    args: argparse.ArgumentParser,
) -> None:
    """Default Training Loop.

    Args:
        model (nn.Module): Model to train
        optimizer (torch.optimizer): Optimizer to use
        train_loader (torch.utils.data.DataLoader): Training data loader
        use_cuda (bool): Whether to use cuda or not
        epoch (int): Current epoch
        args (argparse.ArgumentParser): Arguments parsed from command line
    """
    model.train()
    correct = 0
    total_loss = 0
    total_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.data.item(),
                )
            )
    
    scheduler.step()
    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_accuracy = 100.0 * correct / len(train_loader.dataset)
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset),
        )
    )
    wandb.log({
        "Train Epoch Loss": epoch_loss,
        "Train Epoch Accuracy": epoch_accuracy,
        "Learning Rate": scheduler.get_last_lr()[0],
        "Epoch": epoch
    })


def validation(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    use_cuda: bool,
) -> float:
    """Default Validation Loop.

    Args:
        model (nn.Module): Model to train
        val_loader (torch.utils.data.DataLoader): Validation data loader
        use_cuda (bool): Whether to use cuda or not

    Returns:
        float: Validation loss
    """
    model.eval()
    validation_loss = 0
    correct = 0
    misclassified_info = []

    for batch_idx, (data, target) in enumerate(val_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # misclassified = (pred != target.data.view_as(pred)).nonzero(as_tuple=True)[0]
        # for idx in misclassified:
        #     misclassified_info.append((batch_idx * val_loader.batch_size + idx.item(), target[idx].item(), pred[idx].item()))


    validation_loss /= len(val_loader.dataset)
    accuracy = 100.0 * correct / len(val_loader.dataset)

    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            validation_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )

    wandb.log({
        "Validation Loss": validation_loss,
        "Validation Accuracy": accuracy,
    })
    return validation_loss

def visualize_misclassified_images(misclassified_info, val_dataset, image_folder, num_images=5):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
    for i, (idx, true_label, pred_label) in enumerate(misclassified_info[:num_images]):
        img_path = val_dataset.samples[idx][0]
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        img = img.permute(1, 2, 0).numpy()

        true_label_path = os.path.join(image_folder, val_dataset.classes[true_label])
        true_label_img = Image.open(true_label_path).convert("RGB")
        true_label_img = transform(true_label_img)
        true_label_img = true_label_img.permute(1, 2, 0).numpy()

        pred_label_path = os.path.join(image_folder, val_dataset.classes[pred_label])
        pred_label_img = Image.open(pred_label_path).convert("RGB")
        pred_label_img = transform(pred_label_img)
        pred_label_img = pred_label_img.permute(1, 2, 0).numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'True: {true_label}')

        axes[i, 1].imshow(pred_label_img)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'Pred: {pred_label}')

    plt.tight_layout()
    plt.show()

def main():
    args = opts()

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    else:
        wandb.login()

    wandb.init(project='recvis2024', name="Eva-Clip_with_img_feature_labels_AdamW", config=vars(args))

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)

    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    model = Net(input_dim=1408, num_classes=500)
    model.to(device)

    train_dataset = FeaturesDataset(os.path.join(args.data, 'train_features.pt'))
    val_dataset = FeaturesDataset(os.path.join(args.data, 'val_features.pt'))

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

    optimizer = AdamW(model.parameters(), lr=args.lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    steps_per_epoch=len(train_loader), 
                                                    max_lr=3e-4, 
                                                    pct_start=0.1,
                                                    epochs=30,
                                                    anneal_strategy='cos')
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, scheduler, train_loader, use_cuda, epoch, args)
        val_loss = validation(model, val_loader, use_cuda)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_file = os.path.join(args.experiment, "model_best.pth")
            torch.save(model.state_dict(), best_model_file)

        model_file = os.path.join(args.experiment, f"model_{epoch}.pth")
        torch.save(model.state_dict(), model_file)
        print(f"Saved model to {model_file}")


    wandb.finish()

if __name__ == "__main__":
    main()
