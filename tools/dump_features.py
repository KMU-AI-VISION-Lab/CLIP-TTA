import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis_utils import get_image_paths, load_dataset_and_features
from datasets.imagenet import imagenet_classes


CANONICAL_DATASET_NAMES = {
    "imagenet": "ImageNet_v1",
    "imagenet_v2": "ImageNet_v2",
    "imagenet_sketch": "ImageNet_Sketch",
    "imagenet_a": "ImageNet_A",
    "imagenet_r": "ImageNet_R",
}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--root_path", default="/data2/TTA_dataset", type=str)
    parser.add_argument("--backbone", default="vit_b16", type=str, choices=["rn50", "rn101", "vit_b32", "vit_b16", "vit_l14"])
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--load", action="store_true", default=False, help="Load image features from cache_dir if present.")
    parser.add_argument("--device", default=None, type=str, help="Device passed to CLIP loading, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--batch_size", default=64, type=int, help="Feature extraction batch size.")
    parser.add_argument("--num_workers", default=8, type=int)
    return parser.parse_args()


def main():
    args = get_arguments()

    cache_dir = args.cache_dir or os.path.join("./caches", f"{args.dataset}_{args.backbone}")
    output_path = args.output or os.path.join(cache_dir, f"{args.dataset}_{args.backbone}_features.pt")

    # Load the dataset exactly once and extract frozen CLIP image embeddings.
    # These saved tensors are the inputs to the later geometry analysis script.
    _, _, dataset, _, features, labels, _ = load_dataset_and_features(
        dataset_name=args.dataset,
        root_path=args.root_path,
        backbone=args.backbone,
        cache_dir=cache_dir,
        load=args.load,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    export_labels = labels.long().cpu()
    export_classnames = list(dataset.classnames)
    if hasattr(dataset, "subset_to_imagenet_index"):
        # Some datasets only contain a subset of ImageNet classes (e.g. ImageNet-R).
        # For cross-dataset analysis we export labels in the original ImageNet-1k index space.
        subset_mapping = torch.tensor(dataset.subset_to_imagenet_index, dtype=torch.long)
        export_labels = subset_mapping[export_labels]
        export_classnames = list(imagenet_classes)

    image_paths = get_image_paths(getattr(dataset, "test", None))
    payload = {
        # features: [N, D] normalized CLIP image embeddings
        "features": features.float().cpu(),
        # labels: ImageNet-aligned class ids used by geometry_eval.py
        "labels": export_labels,
        "classnames": export_classnames,
        "dataset_name": CANONICAL_DATASET_NAMES.get(args.dataset, args.dataset),
        "backbone": args.backbone,
        "image_paths": image_paths,
    }

    if hasattr(dataset, "available_imagenet_indices"):
        payload["available_imagenet_indices"] = list(dataset.available_imagenet_indices)
    if hasattr(dataset, "subset_classnames"):
        payload["subset_classnames"] = list(dataset.subset_classnames)
    if hasattr(dataset, "subset_to_imagenet_index"):
        payload["subset_to_imagenet_index"] = list(dataset.subset_to_imagenet_index)
        payload["raw_labels"] = labels.long().cpu()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(payload, output_path)

    print(f"Saved features to: {output_path}")
    print(f"Features shape: {tuple(payload['features'].shape)}")
    print(f"Labels shape: {tuple(payload['labels'].shape)}")
    print(f"Num classnames: {len(payload['classnames'])}")


if __name__ == "__main__":
    main()
