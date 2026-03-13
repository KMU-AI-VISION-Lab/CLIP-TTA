import os

import torchvision.datasets as datasets


def _first_existing_dir(paths):
    for path in paths:
        if path and os.path.isdir(path):
            return path
    return paths[0]


def get_imagenet_train_dir(root):
    return _first_existing_dir([
        os.path.join(root, "imagenet", "images", "train"),
        os.path.join(root, "imagenet", "train"),
        os.path.join(root, "ImageNet", "train"),
        os.path.join(root, "ImageNet"),
    ])


def get_imagenet_val_dir(root):
    return _first_existing_dir([
        os.path.join(root, "imagenet", "images", "val"),
        os.path.join(root, "imagenet", "val"),
        os.path.join(root, "ImageNet", "val"),
        os.path.join(root, "ImageNet"),
    ])


def get_imagenet_v2_dir(root):
    return _first_existing_dir([
        os.path.join(root, "imagenet-v2", "imagenetv2-matched-frequency-format-val"),
        os.path.join(root, "imagenet-v2", "images"),
        os.path.join(root, "imagenetv2", "images"),
        os.path.join(root, "ImageNetV2", "imagenetv2-matched-frequency-format-val"),
        os.path.join(root, "ImageNetV2"),
    ])


def get_imagenet_sketch_dir(root):
    return _first_existing_dir([
        os.path.join(root, "imagenet-sketch", "images"),
        os.path.join(root, "imagenet-sketch"),
        os.path.join(root, "ImageNet-Sketch"),
    ])


def get_imagenet_r_dir(root):
    return _first_existing_dir([
        os.path.join(root, "imagenet-r", "images"),
        os.path.join(root, "imagenet-r"),
        os.path.join(root, "imagenet-rendition", "images"),
        os.path.join(root, "imagenet-rendition"),
        os.path.join(root, "ImageNet-R"),
    ])


def get_imagenet_a_dir(root):
    return _first_existing_dir([
        os.path.join(root, "imagenet-adversarial", "images"),
        os.path.join(root, "imagenet-adversarial"),
        os.path.join(root, "ImageNet-A"),
    ])


def build_imagenet_folder_to_index(root):
    reference_dir = get_imagenet_train_dir(root)
    if not os.path.isdir(reference_dir):
        reference_dir = get_imagenet_val_dir(root)
    if not os.path.isdir(reference_dir):
        raise FileNotFoundError(
            "Could not find ImageNet train/val folders under the provided root. "
            "Expected one of 'imagenet/images/train' or 'imagenet/images/val'."
        )

    reference_dataset = datasets.ImageFolder(reference_dir)
    return dict(reference_dataset.class_to_idx)


class ImageFolderWithAlignedTargets(datasets.ImageFolder):
    def __init__(self, root, transform, folder_to_imagenet_idx):
        super().__init__(root, transform=transform)

        original_idx_to_folder = {value: key for key, value in self.class_to_idx.items()}
        aligned_samples = []
        aligned_targets = []

        for image_path, original_target in self.samples:
            folder_name = original_idx_to_folder[original_target]

            if folder_name.isdigit():
                aligned_target = int(folder_name)
            elif folder_name in folder_to_imagenet_idx:
                aligned_target = folder_to_imagenet_idx[folder_name]
            else:
                continue

            aligned_samples.append((image_path, aligned_target))
            aligned_targets.append(aligned_target)

        if not aligned_samples:
            raise RuntimeError(
                f"No valid ImageNet-aligned samples were found under '{root}'. "
                "Check the directory layout and class folder names."
            )

        self.samples = aligned_samples
        self.imgs = aligned_samples
        self.targets = aligned_targets
        self.aligned_class_to_idx = folder_to_imagenet_idx
        self.available_imagenet_indices = sorted(set(aligned_targets))
