import argparse
import json
import os

import torch


def get_arguments():
    parser = argparse.ArgumentParser()
    # `dataset_name=feature_path` 형태를 여러 개 받습니다.
    # 예: ImageNet_v1=./caches/...pt ImageNet_A=./caches/...pt
    # 이렇게 받은 feature 파일들 사이의 "공통으로 존재하는 ImageNet class"만 남겨서
    # balanced group config를 만들게 됩니다.
    parser.add_argument("--feature_files", nargs="+", required=True, help="Items like ImageNet_v1=path/to/features.pt")
    parser.add_argument("--group_candidates_file", default="./configs/imagenet_supergroup_candidates.json", type=str)
    parser.add_argument("--groups", nargs="+", default=["dog", "bird", "vehicle"])
    parser.add_argument("--config_name", required=True, type=str)
    parser.add_argument("--output_dir", default="./configs/generated", type=str)
    parser.add_argument("--max_classes_per_group", default=None, type=int)
    parser.add_argument("--seed", default=1, type=int)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_feature_payload(path):
    # feature dump 결과는 PyTorch의 pickle 형식(.pt)이라 `torch.load`로 읽습니다.
    # map_location="cpu"를 쓰면 GPU가 없어도 안전하게 메모리로 가져올 수 있습니다.
    payload = torch.load(path, map_location="cpu")
    required = {"labels", "classnames", "dataset_name"}
    missing = required - set(payload.keys())
    if missing:
        raise KeyError(f"Missing keys in {path}: {sorted(missing)}")
    return payload


def parse_feature_items(items):
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected dataset_name=path format, got: {item}")
        dataset_name, path = item.split("=", maxsplit=1)
        parsed.append((dataset_name, path))
    return parsed


def available_indices_from_payload(payload):
    # subset dataset(A/R)처럼 ImageNet-1k 전체 중 일부만 가진 경우,
    # dump 시점에 `available_imagenet_indices`를 같이 저장해 두었습니다.
    # 이 값이 있으면 그걸 그대로 쓰는 것이 가장 정확합니다.
    if "available_imagenet_indices" in payload:
        return set(int(index) for index in payload["available_imagenet_indices"])
    # full dataset(v1/v2/sketch) 쪽은 보통 label에 등장한 class id 자체가 available set입니다.
    # `labels`는 이미지마다 class id가 하나씩 들어 있는 1차원 텐서/배열입니다.
    labels = payload["labels"].long().cpu().tolist()
    return set(int(index) for index in labels)


def choose_balanced_ids(common_indices, candidate_groups, groups, max_classes_per_group):
    selected_by_group = {}
    group_sizes = {}

    for group_name in groups:
        candidate_ids = [int(class_id) for class_id in candidate_groups[group_name]]
        # candidate group에서 실제 dataset 교집합에도 존재하는 class만 남깁니다.
        # 예를 들어 dog 후보가 20개 있어도, ImageNet_A와 공통으로 존재하는 dog class가 7개뿐이면
        # 여기서는 그 7개만 valid id가 됩니다.
        valid_ids = sorted(class_id for class_id in candidate_ids if class_id in common_indices)
        selected_by_group[group_name] = valid_ids
        group_sizes[group_name] = len(valid_ids)

    # group 간 밸런스를 맞추기 위해 가장 작은 group 크기에 맞춰 모두 자릅니다.
    # 예: dog 8개, bird 5개, vehicle 6개면 최종적으로 각 group에서 5개만 사용합니다.
    balanced_size = min(group_sizes.values())
    if max_classes_per_group is not None:
        balanced_size = min(balanced_size, max_classes_per_group)

    if balanced_size <= 0:
        raise ValueError(
            "At least one requested group has no classes in the dataset intersection. "
            f"Group sizes: {group_sizes}"
        )

    balanced_groups = {
        group_name: class_ids[:balanced_size]
        for group_name, class_ids in selected_by_group.items()
    }
    return balanced_groups, group_sizes, balanced_size


def main():
    args = get_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    feature_items = parse_feature_items(args.feature_files)
    candidate_groups = load_json(args.group_candidates_file)
    groups = [group.lower() for group in args.groups]

    for group in groups:
        if group not in candidate_groups:
            raise KeyError(f"Requested group '{group}' not found in {args.group_candidates_file}")

    payloads = []
    dataset_names = []
    feature_paths = []
    available_sets = []

    for dataset_name, feature_path in feature_items:
        payload = load_feature_payload(feature_path)
        payloads.append(payload)
        dataset_names.append(dataset_name)
        feature_paths.append(feature_path)
        available_sets.append(available_indices_from_payload(payload))

    # 여러 dataset이 동시에 실험에 들어갈 수 있으므로,
    # "모든 dataset에서 공통으로 존재하는 class id"의 교집합을 먼저 구합니다.
    # set.intersection(*available_sets)는 여러 집합의 공통 원소만 남기는 연산입니다.
    common_indices = set.intersection(*available_sets)
    if not common_indices:
        raise ValueError("No common ImageNet indices across the provided feature files.")

    balanced_groups, group_sizes, balanced_size = choose_balanced_ids(
        common_indices=common_indices,
        candidate_groups=candidate_groups,
        groups=groups,
        max_classes_per_group=args.max_classes_per_group,
    )

    classnames = payloads[0]["classnames"]
    balanced_classnames = {
        group_name: [classnames[class_id] for class_id in class_ids]
        for group_name, class_ids in balanced_groups.items()
    }

    # metadata는 사람이 실험 세팅을 다시 확인할 때 쓰는 설명 파일입니다.
    # "교집합이 몇 개였는지", "각 group이 balancing 전에 몇 개였는지"를 남겨 둡니다.
    metadata = {
        "config_name": args.config_name,
        "dataset_names": dataset_names,
        "feature_files": feature_paths,
        "groups": groups,
        "group_candidates_file": args.group_candidates_file,
        "common_available_indices_count": len(common_indices),
        "group_sizes_before_balancing": group_sizes,
        "balanced_group_size": balanced_size,
        "selected_class_ids_by_group": balanced_groups,
        "selected_class_names_by_group": balanced_classnames,
        "balanced_class_ids_union": sorted({class_id for class_ids in balanced_groups.values() for class_id in class_ids}),
    }

    # This JSON is directly reusable by the curated-group analysis tools.
    reusable_group_file = {
        group_name: class_ids
        for group_name, class_ids in balanced_groups.items()
    }

    group_output_path = os.path.join(args.output_dir, f"{args.config_name}_groups.json")
    metadata_output_path = os.path.join(args.output_dir, f"{args.config_name}_metadata.json")

    with open(group_output_path, "w", encoding="utf-8") as handle:
        json.dump(reusable_group_file, handle, indent=2)
    with open(metadata_output_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved reusable group file to: {group_output_path}")
    print(f"Saved metadata file to: {metadata_output_path}")
    print(f"Dataset names: {', '.join(dataset_names)}")
    print(f"Groups: {', '.join(groups)}")
    print(f"Common available indices: {len(common_indices)}")
    print(f"Balanced group size: {balanced_size}")
    for group_name in groups:
        print(
            f"{group_name}: before={group_sizes[group_name]}, "
            f"after={len(balanced_groups[group_name])}"
        )


if __name__ == "__main__":
    main()
