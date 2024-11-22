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
import torch.nn.functional as F

class TwoCropsTransform:
    """Créer deux augmentations d'une même image."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        xi = self.base_transform(x)
        xj = self.base_transform(x)
        return xi, xj
    


def compute_dense_contrastive_loss(patches_i, patches_j, temperature=0.1):
    """
    Calculer la perte contrastive dense entre les embeddings de patches.
    patches_i, patches_j: [batch_size, num_patches, hidden_dim]
    """
    batch_size, num_patches, hidden_dim = patches_i.size()
    
    # Normaliser les embeddings
    patches_i = F.normalize(patches_i, dim=2)
    patches_j = F.normalize(patches_j, dim=2)
    
    # Reshaper pour avoir [batch_size * num_patches, hidden_dim]
    z_i = patches_i.view(-1, hidden_dim)
    z_j = patches_j.view(-1, hidden_dim)
    
    # Calculer les similitudes
    logits = torch.matmul(z_i, z_j.T) / temperature  # [B*N, B*N]
    
    # Créer les étiquettes positives (positions diagonales)
    labels = torch.arange(z_i.size(0)).to(z_i.device)
    
    # Calculer la perte
    loss = F.cross_entropy(logits, labels)
    return loss

    
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
    lambda_contrastive = 1.0
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, ((data_i, data_j), target) in enumerate(train_loader):
        if use_cuda:
            data_i, data_j, target = data_i.cuda(), data_j.cuda(), target.cuda()
        
        optimizer.zero_grad()
        
        # Passer les deux versions à travers le modèle
        logits_i, patches_i = model(data_i)
        logits_j, patches_j = model(data_j)
        
        # Calculer la perte de classification sur la première vue
        criterion_cls = torch.nn.CrossEntropyLoss()
        loss_cls = criterion_cls(logits_i, target)
        
        # Calculer la perte contrastive dense
        loss_contrastive = compute_dense_contrastive_loss(patches_i, patches_j, temperature=0.1)
        
        # Combiner les pertes
        loss = loss_cls + lambda_contrastive * loss_contrastive
        
        loss.backward()
        optimizer.step()
        
        # Calcul des métriques
        pred = logits_i.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total_loss += loss.item() * data_i.size(0)
        total_samples += data_i.size(0)
        
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data_i),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    
    scheduler.step()
    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_accuracy = 100.0 * correct / len(train_loader.dataset)
    print(
        "\nTrain set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct,
            len(train_loader.dataset),
            epoch_accuracy,
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
    lambda_contrastive = 1.0
    for batch_idx, ((data_i, data_j), target) in enumerate(val_loader):
        if use_cuda:
            data_i, data_j, target = data_i.cuda(), data_j.cuda(), target.cuda()
        logits_i, patches_i = model(data_i)
        logits_j, patches_j = model(data_j)
        # sum up batch loss
        criterion_cls = torch.nn.CrossEntropyLoss()
        loss_cls = criterion_cls(logits_i, target)

        loss_contrastive = compute_dense_contrastive_loss(patches_i, patches_j, temperature=0.1)

        validation_loss += loss_cls + lambda_contrastive * loss_contrastive

        # validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        # pred = output.data.max(1, keepdim=True)[1]
        pred = logits_i.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

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

def main():
    """Default Main Function."""
    # options
    args = opts()

    if args.wandb_api_key:
        wandb.login(key=args.wandb_api_key)
    else:
        wandb.login()

    wandb.init(project='recvis2024', name="CLIP-ViT-H-14-laion2B-s32B-b79K_adamw_betas9e-1n9_8e-1_eps1e-6_weightdecay_2e-1_contrastive_learning", config=vars(args))

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()

    # Set the seed (for reproducibility)
    torch.manual_seed(args.seed)

    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # load model and transform
    model, data_transforms, val_transforms = ModelFactory(args.model_name).get_all()
    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    # Utiliser TwoCropsTransform pour générer deux versions
    train_dataset = datasets.ImageFolder(
        args.data + "/train_images",
        transform=TwoCropsTransform(data_transforms)
    )

    validation_dataset = datasets.ImageFolder(
        args.data + "/val_images",
        transform=TwoCropsTransform(val_transforms)
    )


    # Data initialization and loading
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(args.data + "/train_images", transform=data_transforms),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    # )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Setup optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9,0.98),eps=1e-6, weight_decay=0.2)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loop over the epochs
    best_val_loss = 1e8
    for epoch in range(1, args.epochs + 1):
        # training loop
        # train(model, optimizer, train_loader, use_cuda, epoch, args)
        train(model, optimizer, scheduler, train_loader, use_cuda, epoch, args)
        # validation loop
        val_loss = validation(model, val_loader, use_cuda)
        if val_loss < best_val_loss:
            # save the best model for validation
            best_val_loss = val_loss
            best_model_file = args.experiment + "/model_best.pth"
            torch.save(model.state_dict(), best_model_file)
        # also save the model every epoch
        model_file = args.experiment + "/model_" + str(epoch) + ".pth"

        # if epoch % 5 == 0:
        torch.save(model.state_dict(), model_file)
        print(
            "Saved model to "
            + model_file
            + f". You can run `python evaluate.py --model_name {args.model_name} --model "
            + best_model_file
            + "` to generate the Kaggle formatted csv file\n"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
