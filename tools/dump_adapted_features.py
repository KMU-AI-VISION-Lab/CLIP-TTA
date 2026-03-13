import argparse
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis_utils import get_image_paths, load_dataset_and_features
from main import get_hp, set_random_seed
from sampler import BatchSampler, OnlineSampler


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--root_path", default="/data2/TTA_datatset", type=str)
    parser.add_argument("--backbone", default="vit_b16", type=str, choices=["rn50", "rn101", "vit_b32", "vit_b16", "vit_l14"])
    parser.add_argument("--method", default="StatA", type=str, choices=["StatA", "TransCLIP", "Dirichlet", "ZLaP"])
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--device", default=None, type=str, help="Device passed to CLIP loading, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--feature_batch_size", default=64, type=int, help="Feature extraction batch size before adaptation.")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_class_eff", default=None, type=int)
    parser.add_argument("--num_class_eff_min", default=None, type=int)
    parser.add_argument("--num_class_eff_max", default=None, type=int)
    parser.add_argument("--online", action="store_true", default=False)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--lambda_laplacian", default=1.0, type=float)
    parser.add_argument("--soft_beta", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = get_arguments()
    set_random_seed(args.seed)

    cache_dir = args.cache_dir or os.path.join("./caches", f"{args.dataset}_{args.backbone}")
    output_path = args.output or os.path.join(cache_dir, f"{args.dataset}_{args.backbone}_{args.method}_adapted.pt")

    # First extract the frozen CLIP features exactly as in the other analysis scripts.
    _, _, dataset, _, features, labels, clip_prototypes = load_dataset_and_features(
        dataset_name=args.dataset,
        root_path=args.root_path,
        backbone=args.backbone,
        cache_dir=cache_dir,
        load=args.load,
        device=args.device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )

    solver, method_args = get_hp(args, args.method)
    if solver is None:
        raise ValueError(f"Method '{args.method}' is not supported by this analysis script.")

    if args.online:
        num_batch = max(features.shape[0] // args.batch_size, 1)
        num_slots = min(num_batch, len(torch.unique(labels)))
        sampler = OnlineSampler(features, labels, args.gamma, num_slots, args.batch_size)
    else:
        # Offline mode samples one batch/task from the saved test features.
        sampler = BatchSampler(features, labels, args.batch_size, args.num_class_eff, args.num_class_eff_min, args.num_class_eff_max)

    sampled_indices = sampler.generate_indices()
    if sampled_indices is None:
        raise RuntimeError("Could not sample any indices from the dataset.")

    batch_features = features[sampled_indices]
    batch_labels = labels[sampled_indices]
    # The existing solvers operate on frozen CLIP features/logits here.
    # We save those outputs for analysis rather than modifying the training code path.
    zs_scores, adapted_scores = solver(batch_features, batch_labels, clip_prototypes, **method_args)
    zs_logits = 100.0 * batch_features @ clip_prototypes.squeeze().cpu()

    image_paths = get_image_paths(getattr(dataset, "test", None))
    sampled_paths = [image_paths[index] for index in sampled_indices] if image_paths is not None else None

    payload = {
        "dataset_name": args.dataset,
        "backbone": args.backbone,
        "method": args.method,
        "indices": torch.tensor(sampled_indices, dtype=torch.long),
        "labels": batch_labels.long().cpu(),
        "features": batch_features.float().cpu(),
        "zero_shot_logits": zs_logits.float().cpu(),
        "zero_shot_scores": zs_scores.float().cpu(),
        "adapted_scores": adapted_scores.float().cpu(),
        "image_paths": sampled_paths,
        "note": "This script stores frozen CLIP image embeddings and prediction behavior before/after adaptation. It does not extract adapted image embeddings.",
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(payload, output_path)

    print(f"Saved adapted analysis payload to: {output_path}")
    print(f"Num sampled instances: {len(sampled_indices)}")
    print(f"Feature shape: {tuple(payload['features'].shape)}")


if __name__ == "__main__":
    main()
