import os


def resolve_dataset_dir(root, preferred_dir, aliases):
    candidates = [preferred_dir] + list(aliases)
    for candidate in candidates:
        candidate_path = os.path.join(root, candidate)
        if os.path.exists(candidate_path):
            return candidate_path

    return os.path.join(root, preferred_dir)
