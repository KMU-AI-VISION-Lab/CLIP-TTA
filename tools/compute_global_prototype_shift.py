import argparse
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.geometry_eval import (
    build_class_index,
    build_inter_class_structure,
    load_feature_file,
    safe_vector_cosine,
    sample_class_features,
)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="./outputs/geometry_vit_b16", type=str)
    parser.add_argument("--cache_root", default=None, type=str)
    parser.add_argument("--result_files", nargs="+", default=None, help="Optional explicit main-result JSON paths.")
    parser.add_argument("--write_back", action="store_true", default=False, help="Write the computed global prototype fields back into each main result JSON.")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def find_result_files(output_root, explicit_files):
    if explicit_files:
        return sorted(explicit_files)

    candidates = []
    for pattern in [
        os.path.join(output_root, "main_results", "*.json"),
        os.path.join(output_root, "*.json"),
    ]:
        candidates.extend(glob.glob(pattern))

    results = []
    for path in sorted(set(candidates)):
        base = os.path.basename(path)
        if any(
            token in base
            for token in [
                "_inter_class_per_",
                "_viz_metrics.json",
                "_summary.json",
                "_pair_buckets_",
                "_radius_buckets_",
            ]
        ):
            continue
        results.append(path)
    return results


def infer_cache_root(output_root):
    base_name = os.path.basename(os.path.normpath(output_root))
    if base_name.startswith("geometry_"):
        return os.path.join("./caches", base_name)
    return "./caches/geometry_vit_b16"


def feature_file_from_result(cache_root, dataset_name, backbone):
    return os.path.join(cache_root, dataset_name, f"{dataset_name}_{backbone}_features.pt")


def rebuild_cross_dataset_sampling(result_payload, cache_root):
    source_dataset = result_payload["source_dataset"]
    target_dataset = result_payload["target_dataset"]
    backbone = result_payload["backbone"]
    seed = int(result_payload["seed"])
    num_classes = int(result_payload["num_classes"])
    samples_per_class = int(result_payload["samples_per_class"])
    expected_selected_classes = list(result_payload["selected_classes"])

    source_feature_file = feature_file_from_result(cache_root, source_dataset, backbone)
    target_feature_file = feature_file_from_result(cache_root, target_dataset, backbone)
    if not os.path.isfile(source_feature_file):
        raise FileNotFoundError(f"Missing source feature file: {source_feature_file}")
    if not os.path.isfile(target_feature_file):
        raise FileNotFoundError(f"Missing target feature file: {target_feature_file}")

    source_payload = load_feature_file(source_feature_file)
    target_payload = load_feature_file(target_feature_file)

    source_features = source_payload["features"].float().cpu().numpy()
    target_features = target_payload["features"].float().cpu().numpy()
    source_labels = source_payload["labels"].long().cpu()
    target_labels = target_payload["labels"].long().cpu()

    source_class_to_indices = build_class_index(source_labels)
    target_class_to_indices = build_class_index(target_labels)
    eligible_classes = sorted(set(source_class_to_indices).intersection(target_class_to_indices))
    eligible_classes = [
        class_index
        for class_index in eligible_classes
        if len(source_class_to_indices[class_index]) >= samples_per_class
        and len(target_class_to_indices[class_index]) >= samples_per_class
    ]

    rng = np.random.default_rng(seed)
    selected_classes = sorted(rng.choice(eligible_classes, size=num_classes, replace=False).tolist())
    if selected_classes != expected_selected_classes:
        raise ValueError(
            "Stored selected_classes do not match the classes reconstructed from the saved seed and feature files. "
            f"Expected {expected_selected_classes[:5]}..., got {selected_classes[:5]}..."
        )

    source_class_samples = {}
    target_class_samples = {}
    for class_index in selected_classes:
        source_class_samples[class_index] = sample_class_features(
            source_features, source_class_to_indices, class_index, samples_per_class, rng
        )
        target_class_samples[class_index] = sample_class_features(
            target_features, target_class_to_indices, class_index, samples_per_class, rng
        )

    return selected_classes, source_class_samples, target_class_samples


def compute_global_shift_record(result_payload, cache_root):
    if result_payload.get("mode") != "cross_dataset":
        return None

    selected_classes, source_class_samples, target_class_samples = rebuild_cross_dataset_sampling(
        result_payload=result_payload,
        cache_root=cache_root,
    )

    source_structure = build_inter_class_structure(source_class_samples, selected_classes)
    target_structure = build_inter_class_structure(target_class_samples, selected_classes)

    source_global_prototype = np.asarray(source_structure["global_prototype"], dtype=np.float64)
    target_global_prototype = np.asarray(target_structure["global_prototype"], dtype=np.float64)

    return {
        "source_dataset": result_payload["source_dataset"],
        "target_dataset": result_payload["target_dataset"],
        "backbone": result_payload["backbone"],
        "num_classes": int(result_payload["num_classes"]),
        "samples_per_class": int(result_payload["samples_per_class"]),
        "seed": int(result_payload["seed"]),
        "selected_classes": selected_classes,
        "source_global_prototype": [float(value) for value in source_global_prototype],
        "target_global_prototype": [float(value) for value in target_global_prototype],
        "global_prototype_shift_l2": float(np.linalg.norm(target_global_prototype - source_global_prototype)),
        "global_prototype_shift_cosine": safe_vector_cosine(source_global_prototype, target_global_prototype),
    }


def main():
    args = get_arguments()
    cache_root = args.cache_root or infer_cache_root(args.output_root)
    result_files = find_result_files(args.output_root, args.result_files)
    if not result_files:
        raise FileNotFoundError(f"No main result JSON files found under {args.output_root}")

    updated = 0
    skipped = 0
    for result_path in result_files:
        payload = load_json(result_path)
        if payload.get("mode") != "cross_dataset":
            skipped += 1
            continue

        record = compute_global_shift_record(payload, cache_root)
        sidecar_path = os.path.splitext(result_path)[0] + "_global_prototype_shift.json"
        save_json(sidecar_path, record)

        if args.write_back:
            payload.setdefault("inter_class_geometry", {})
            payload["inter_class_geometry"]["source_global_prototype"] = record["source_global_prototype"]
            payload["inter_class_geometry"]["target_global_prototype"] = record["target_global_prototype"]
            payload["inter_class_geometry"]["global_prototype_shift_l2"] = record["global_prototype_shift_l2"]
            payload["inter_class_geometry"]["global_prototype_shift_cosine"] = record["global_prototype_shift_cosine"]
            save_json(result_path, payload)

        print(
            f"{os.path.basename(result_path)}: "
            f"global_prototype_shift_l2={record['global_prototype_shift_l2']:.6f}, "
            f"global_prototype_shift_cosine={record['global_prototype_shift_cosine']:.6f}"
        )
        updated += 1

    print(f"Processed cross-dataset result files: {updated}")
    print(f"Skipped non-cross-dataset result files: {skipped}")
    print(f"Cache root: {cache_root}")
    if args.write_back:
        print("Existing main result JSON files were updated in place with the new global prototype fields.")
    else:
        print("Main result JSON files were left unchanged; sidecar JSON files were written next to them.")


if __name__ == "__main__":
    main()
