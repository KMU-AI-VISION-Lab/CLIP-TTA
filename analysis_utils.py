import os
from argparse import Namespace

import clip
import torch

from datasets import get_all_dataloaders
from utils import get_all_features


BACKBONES = {
    "rn50": "RN50",
    "rn101": "RN101",
    "vit_b32": "ViT-B/32",
    "vit_b16": "ViT-B/16",
    "vit_l14": "ViT-L/14",
}


def make_feature_args(dataset_name, root_path, cache_dir, load, batch_size=64, num_workers=8):
    return Namespace(
        dataset=dataset_name,
        root_path=root_path,
        cache_dir=cache_dir,
        load=load,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def ensure_cache_dir(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_clip_model(backbone, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(BACKBONES[backbone], device=device)
    clip_model.eval()
    return clip_model, preprocess


def load_dataset_and_features(dataset_name, root_path, backbone, cache_dir, load, device=None, batch_size=64, num_workers=8):
    cache_dir = ensure_cache_dir(cache_dir)
    clip_model, preprocess = load_clip_model(backbone, device=device)
    args = make_feature_args(dataset_name, root_path, cache_dir, load, batch_size=batch_size, num_workers=num_workers)
    _, _, test_loader, dataset = get_all_dataloaders(args, preprocess, num_workers=num_workers)
    features, labels, prototypes = get_all_features(args, test_loader, dataset, clip_model)
    return clip_model, preprocess, dataset, test_loader, features, labels, prototypes


def get_image_paths(image_dataset):
    if image_dataset is None:
        return None
    if hasattr(image_dataset, "samples"):
        return [sample[0] for sample in image_dataset.samples]
    if hasattr(image_dataset, "imgs"):
        return [sample[0] for sample in image_dataset.imgs]
    return None
