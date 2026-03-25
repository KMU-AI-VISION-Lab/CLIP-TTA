#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


DEFAULT_DATASETS = [
    "imagenet",
    "sun397",
    "fgvc",
    "eurosat",
    "stanford_cars",
    "food101",
    "oxford_pets",
    "oxford_flowers",
    "caltech101",
    "dtd",
    "ucf101",
]

IMAGENET_FAMILY_DATASETS = [
    "imagenet",
    "imagenet_a",
    "imagenet_r",
    "imagenet_sketch",
    "imagenet_v2",
]

DEFAULT_BACKBONES = ["rn50", "vit_b16", "vit_b32", "vit_l14"]
DEFAULT_METHODS = ["TPT", "TDA", "DOTA", "TransCLIP", "StatA"]


METHOD_SPECS = {
    "StatA": {"implemented": True, "supports_online": True, "supports_batch": True},
    "TransCLIP": {"implemented": True, "supports_online": True, "supports_batch": True},
    "TDA": {"implemented": True, "supports_online": True, "supports_batch": False},
    "TPT": {"implemented": False, "supports_online": False, "supports_batch": False},
    "DOTA": {"implemented": False, "supports_online": False, "supports_batch": False},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an experiment manifest for large StatA grid runs."
    )
    parser.add_argument("--output", required=True, help="Path to save the manifest JSON.")
    parser.add_argument("--root_path", required=True, help="Dataset root passed to main.py.")
    parser.add_argument(
        "--setting",
        default="online",
        choices=["online", "batch"],
        help="Shared evaluation setting for all experiments in this manifest.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to include.",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=DEFAULT_BACKBONES,
        help="Backbones to include.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Methods to include.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num_class_eff_min", type=int, default=2)
    parser.add_argument("--num_class_eff_max", type=int, default=10)
    parser.add_argument(
        "--results_root",
        default="outputs/stata_runs",
        help="Root directory used by the queue runner for logs and JSON outputs.",
    )
    parser.add_argument(
        "--preset",
        choices=["all", "imagenet_family"],
        default="all",
        help="Convenience preset for common dataset groups.",
    )
    return parser.parse_args()


def build_shared_config(args):
    if args.setting == "online":
        return {
            "n_tasks": args.n_tasks if args.n_tasks is not None else 100,
            "batch_size": args.batch_size if args.batch_size is not None else 128,
            "online": True,
            "gamma": args.gamma,
            "num_class_eff_min": None,
            "num_class_eff_max": None,
        }

    return {
        "n_tasks": args.n_tasks if args.n_tasks is not None else 1000,
        "batch_size": args.batch_size if args.batch_size is not None else 64,
        "online": False,
        "gamma": None,
        "num_class_eff_min": args.num_class_eff_min,
        "num_class_eff_max": args.num_class_eff_max,
    }


def get_status(method, setting):
    spec = METHOD_SPECS.get(method)
    if spec is None:
        return "missing", f"Unknown method '{method}'."
    if not spec["implemented"]:
        return "missing", f"Method '{method}' is not implemented in this repository."
    if setting == "online" and not spec["supports_online"]:
        return "unsupported", f"Method '{method}' does not support the online setting."
    if setting == "batch" and not spec["supports_batch"]:
        return "unsupported", f"Method '{method}' does not support the batch setting."
    return "pending", ""


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.preset == "imagenet_family":
        args.datasets = IMAGENET_FAMILY_DATASETS

    shared = build_shared_config(args)
    experiments = []

    for dataset in args.datasets:
        for backbone in args.backbones:
            for method in args.methods:
                status, note = get_status(method, args.setting)
                experiments.append(
                    {
                        "dataset": dataset,
                        "backbone": backbone,
                        "method": method,
                        "seed": args.seed,
                        "status": status,
                        "note": note,
                        "root_path": args.root_path,
                        "setting": args.setting,
                        "config": shared,
                    }
                )

    manifest = {
        "root_path": args.root_path,
        "results_root": args.results_root,
        "setting": args.setting,
        "datasets": args.datasets,
        "backbones": args.backbones,
        "methods": args.methods,
        "shared_config": shared,
        "experiments": experiments,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    status_counts = {}
    for experiment in experiments:
        status_counts[experiment["status"]] = status_counts.get(experiment["status"], 0) + 1

    print("Saved manifest:", output_path)
    print("Setting:", args.setting)
    print("Total experiments:", len(experiments))
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
