# import argparse
# import os

# import PIL.Image as Image
# import torch
# from tqdm import tqdm

# from model_factory import ModelFactory


# def opts() -> argparse.ArgumentParser:
#     """Option Handling Function."""
#     parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
#     parser.add_argument(
#         "--data",
#         type=str,
#         default="data_sketches",
#         metavar="D",
#         help="folder where data is located. test_images/ need to be found in the folder",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         metavar="M",
#         help="the model file to be evaluated. Usually it is of the form model_X.pth",
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="basic_cnn",
#         metavar="MOD",
#         help="Name of the model for model and transform instantiation.",
#     )
#     parser.add_argument(
#         "--outfile",
#         type=str,
#         default="experiment/kaggle.csv",
#         metavar="D",
#         help="name of the output csv file",
#     )
#     args = parser.parse_args()
#     return args


# def pil_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, "rb") as f:
#         with Image.open(f) as img:
#             return img.convert("RGB")


# def main() -> None:
#     """Main Function."""
#     # options
#     args = opts()
#     test_dir = args.data + "/test_images/mistery_category"

#     # cuda
#     use_cuda = torch.cuda.is_available()

#     # load model and transform
#     state_dict = torch.load(args.model)
#     model, _, data_transforms = ModelFactory(args.model_name).get_all()
#     model.load_state_dict(state_dict)
#     model.eval()
#     if use_cuda:
#         print("Using GPU")
#         model.cuda()
#     else:
#         print("Using CPU")

#     output_file = open(args.outfile, "w")
#     output_file.write("Id,Category\n")
#     for f in tqdm(os.listdir(test_dir)):
#         if "jpeg" in f:
#             data = data_transforms(pil_loader(test_dir + "/" + f))
#             data = data.view(1, data.size(0), data.size(1), data.size(2))
#             if use_cuda:
#                 data = data.cuda()
#             output = model(data)
#             pred = output.data.max(1, keepdim=True)[1]
#             output_file.write("%s,%d\n" % (f[:-5], pred))

#     output_file.close()

#     print(
#         "Succesfully wrote "
#         + args.outfile
#         + ", you can upload this file to the kaggle competition website"
#     )


# if __name__ == "__main__":
#     main()

############################################################################################################
##### VERSION WITH CONTRASTIVE LEARNING
############################################################################################################

# import argparse
# import os

# import PIL.Image as Image
# import torch
# from tqdm import tqdm

# from model_factory import ModelFactory


# def opts() -> argparse.ArgumentParser:
#     """Option Handling Function."""
#     parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
#     parser.add_argument(
#         "--data",
#         type=str,
#         default="data_sketches",
#         metavar="D",
#         help="folder where data is located. test_images/ need to be found in the folder",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         metavar="M",
#         help="the model file to be evaluated. Usually it is of the form model_X.pth",
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="basic_cnn",
#         metavar="MOD",
#         help="Name of the model for model and transform instantiation.",
#     )
#     parser.add_argument(
#         "--outfile",
#         type=str,
#         default="experiment/kaggle.csv",
#         metavar="D",
#         help="name of the output csv file",
#     )
#     args = parser.parse_args()
#     return args


# def pil_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, "rb") as f:
#         with Image.open(f) as img:
#             return img.convert("RGB")


# def main() -> None:
#     """Main Function."""
#     # options
#     args = opts()
#     test_dir = args.data + "/test_images/mistery_category"

#     # cuda
#     use_cuda = torch.cuda.is_available()

#     # load model and transform
#     state_dict = torch.load(args.model)
#     model, _, data_transforms = ModelFactory(args.model_name).get_all()
#     model.load_state_dict(state_dict)
#     model.eval()
#     if use_cuda:
#         print("Using GPU")
#         model.cuda()
#     else:
#         print("Using CPU")

#     output_file = open(args.outfile, "w")
#     output_file.write("Id,Category\n")
#     for f in tqdm(os.listdir(test_dir)):
#         if "jpeg" in f:
#             data = data_transforms(pil_loader(test_dir + "/" + f))
#             data = data.view(1, data.size(0), data.size(1), data.size(2))
#             if use_cuda:
#                 data = data.cuda()
#             logits, _ = model(data)
#             pred = logits.data.max(1, keepdim=True)[1]
#             output_file.write("%s,%d\n" % (f[:-5], pred))

#     output_file.close()

#     print(
#         "Succesfully wrote "
#         + args.outfile
#         + ", you can upload this file to the kaggle competition website"
#     )


# if __name__ == "__main__":
#     main()


# import argparse
# import os

# import PIL.Image as Image
# import torch
# from tqdm import tqdm

# from model import Net  # Importer le classifieur
# from transformers import CLIPModel
# from torchvision import transforms

# def opts() -> argparse.ArgumentParser:
#     """Option Handling Function."""
#     parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
#     parser.add_argument(
#         "--data",
#         type=str,
#         default="data_sketches",
#         metavar="D",
#         help="Dossier où les données sont situées. test_images/ doit s'y trouver.",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         metavar="M",
#         help="Le fichier du modèle à évaluer (exemple : model_best.pth).",
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="basic_cnn",
#         metavar="MOD",
#         help="Name of the model for model and transform instantiation.",
#     )
#     parser.add_argument(
#         "--outfile",
#         type=str,
#         default="experiment/kaggle.csv",
#         metavar="D",
#         help="Nom du fichier de sortie CSV",
#     )
#     args = parser.parse_args()
#     return args

# def pil_loader(path):
#     with open(path, "rb") as f:
#         with Image.open(f) as img:
#             return img.convert("RGB")

# def main() -> None:
#     args = opts()
#     test_dir = os.path.join(args.data, "test_images", "mistery_category")

#     use_cuda = torch.cuda.is_available()
#     device = torch.device('cuda' if use_cuda else 'cpu')

#     # Charger le classifieur entraîné
#     state_dict = torch.load(args.model, map_location=device)
#     model = Net(input_dim=1024, num_classes=500)
#     model.load_state_dict(state_dict)
#     model.eval()
#     if use_cuda:
#         print("Using GPU")
#         model.cuda()
#     else:
#         print("Using CPU")

#     # Charger le modèle CLIP gelé
#     clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
#     clip_model.eval()
#     clip_model.to(device)
#     for param in clip_model.parameters():
#         param.requires_grad = False

#     # Définir les transformations de données
#     data_transforms = transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         # transforms.Normalize(
#         #     mean=[0.48145466, 0.4578275, 0.40821073],
#         #     std=[0.26862954, 0.26130258, 0.27577711]
#         # ),
#     ])

#     output_file = open(args.outfile, "w")
#     output_file.write("Id,Category\n")
#     for f in tqdm(os.listdir(test_dir)):
#         if "jpeg" in f:
#             data = data_transforms(pil_loader(os.path.join(test_dir, f)))
#             data = data.unsqueeze(0).to(device)
#             with torch.no_grad():
#                 image_features = clip_model.get_image_features(pixel_values=data)
#                 logits = model(image_features)
#                 pred = logits.data.max(1, keepdim=True)[1]
#             output_file.write("%s,%d\n" % (f[:-5], pred.item()))

#     output_file.close()

#     print(
#         "Successfully wrote "
#         + args.outfile
#         + ", you can upload this file to the kaggle competition website"
#     )

# if __name__ == "__main__":
#     main()


import argparse
import os
import timm

import PIL.Image as Image
import torch
from tqdm import tqdm

from model import Net  # Importer le classifieur
from transformers import CLIPModel
from torchvision import transforms

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="Dossier où les données sont situées. test_images/ doit s'y trouver.",
    )
    parser.add_argument(
        "--model",
        type=str,
        metavar="M",
        help="Le fichier du modèle à évaluer (exemple : model_best.pth).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="basic_cnn",
        metavar="MOD",
        help="Name of the model for model and transform instantiation.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="experiment/kaggle.csv",
        metavar="D",
        help="Nom du fichier de sortie CSV",
    )
    args = parser.parse_args()
    return args

def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")

def main() -> None:
    args = opts()
    test_dir = os.path.join(args.data, "test_images", "mistery_category")

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    state_dict = torch.load(args.model, map_location=device)
    model = Net(input_dim=1408, num_classes=500)
    model.load_state_dict(state_dict)
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    timm_data_transforms = timm.data.create_transform(**data_config, is_training=False)

    if use_cuda:
        print("Using GPU")
        model.cuda()
    else:
        print("Using CPU")

    eva_model = timm.create_model('eva_giant_patch14_336.clip_ft_in1k', pretrained=True)
    eva_model.eval()
    eva_model.to(device)
    for param in eva_model.parameters():
        param.requires_grad = False

    data_transforms = transforms.Compose([
        transforms.Resize(336),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])

    # data_transforms = transforms.Compose([
    #     transforms.Resize(eva_model.default_cfg['input_size'][-2:]),
    #     transforms.CenterCrop(eva_model.default_cfg['input_size'][-2:]),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=eva_model.default_cfg['mean'],
    #         std=eva_model.default_cfg['std']
    #     ),
    # ])

    # data_config = timm.data.resolve_model_data_config(model)
    # timm_data_transforms = timm.data.create_transform(**data_config, is_training=False)

    output_file = open(args.outfile, "w")
    output_file.write("Id,Category\n")
    for f in tqdm(os.listdir(test_dir)):
        if "jpeg" in f:
            data = data_transforms(pil_loader(os.path.join(test_dir, f)))
            data = data.unsqueeze(0).to(device)
            with torch.no_grad():
                features = eva_model.forward_features(data)
                image_features = eva_model.forward_head(features, pre_logits=True)
                logits = model(image_features)
                pred = logits.data.max(1, keepdim=True)[1]
            output_file.write("%s,%d\n" % (f[:-5], pred.item()))

    output_file.close()

    print(
        "Successfully wrote "
        + args.outfile
        + ", you can upload this file to the kaggle competition website"
    )

if __name__ == "__main__":
    main()
