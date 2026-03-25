#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run manifest experiments with one queued job per GPU."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON.")
    parser.add_argument(
        "--gpus",
        nargs="+",
        required=True,
        help="GPU ids to assign, for example: --gpus 0 1 2 3",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch main.py.",
    )
    parser.add_argument(
        "--poll_seconds",
        type=float,
        default=10.0,
        help="Seconds between process status checks.",
    )
    parser.add_argument(
        "--rerun_completed",
        action="store_true",
        help="Rerun jobs even if the result JSON already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without launching them.",
    )
    return parser.parse_args()


def safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_")


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_run_dir(results_root, experiment):
    config = experiment["config"]
    run_parts = [
        safe_name(experiment["setting"]),
        safe_name(experiment["dataset"]),
        safe_name(experiment["backbone"]),
        safe_name(experiment["method"]),
        f"seed{experiment['seed']}",
        f"bs{config['batch_size']}",
    ]
    if experiment["setting"] == "online":
        run_parts.append(f"gamma{config['gamma']}")
    else:
        run_parts.append(f"nce{config['num_class_eff_min']}_{config['num_class_eff_max']}")
    return Path(results_root).joinpath(*run_parts)


def build_command(repo_root, python_exec, experiment, results_json):
    config = experiment["config"]
    cmd = [
        python_exec,
        "main.py",
        "--root_path",
        experiment["root_path"],
        "--dataset",
        experiment["dataset"],
        "--method",
        experiment["method"],
        "--backbone",
        experiment["backbone"],
        "--seed",
        str(experiment["seed"]),
        "--batch_size",
        str(config["batch_size"]),
        "--n_tasks",
        str(config["n_tasks"]),
        "--results_json",
        str(results_json),
    ]
    if config["online"]:
        cmd.extend(["--online", "--gamma", str(config["gamma"])])
    else:
        if config["num_class_eff_min"] is not None:
            cmd.extend(["--num_class_eff_min", str(config["num_class_eff_min"])])
        if config["num_class_eff_max"] is not None:
            cmd.extend(["--num_class_eff_max", str(config["num_class_eff_max"])])
    return cmd


def start_job(repo_root, python_exec, results_root, experiment, gpu_id, dry_run=False):
    run_dir = build_run_dir(results_root, experiment)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_json = run_dir / "results.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"

    cmd = build_command(repo_root, python_exec, experiment, results_json)
    display_cmd = " ".join(cmd)

    if dry_run:
        print(f"[DRY RUN][GPU {gpu_id}] {display_cmd}")
        return {
            "process": None,
            "gpu_id": gpu_id,
            "command": cmd,
            "run_dir": str(run_dir),
            "results_json": str(results_json),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

    stdout_file = open(stdout_path, "w", encoding="utf-8")
    stderr_file = open(stderr_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    process = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=stdout_file,
        stderr=stderr_file,
        env=env,
    )

    print(f"[START][GPU {gpu_id}] {display_cmd}")
    return {
        "process": process,
        "gpu_id": gpu_id,
        "command": cmd,
        "run_dir": str(run_dir),
        "results_json": str(results_json),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_file": stdout_file,
        "stderr_file": stderr_file,
    }


def close_files(job):
    for key in ["stdout_file", "stderr_file"]:
        handle = job.get(key)
        if handle is not None and not handle.closed:
            handle.close()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    results_root = Path(manifest["results_root"]).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    queue = []
    for experiment in manifest["experiments"]:
        if experiment["status"] != "pending":
            continue

        run_dir = build_run_dir(results_root, experiment)
        results_json = run_dir / "results.json"
        if results_json.exists() and not args.rerun_completed:
            experiment["status"] = "completed"
            experiment["note"] = f"Skipped because results already exist at {results_json}."
            continue

        queue.append(experiment)

    print("Manifest:", manifest_path)
    print("Results root:", results_root)
    print("GPUs:", ", ".join(args.gpus))
    print("Runnable experiments:", len(queue))

    running = {}
    pending = list(queue)

    while pending or running:
        for gpu_id in args.gpus:
            if gpu_id in running or not pending:
                continue

            experiment = pending.pop(0)
            job = start_job(
                repo_root=Path.cwd(),
                python_exec=args.python,
                results_root=results_root,
                experiment=experiment,
                gpu_id=gpu_id,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                experiment["status"] = "running"
                experiment["note"] = f"Assigned to GPU {gpu_id}."
            running[gpu_id] = (experiment, job)
            if not args.dry_run:
                save_manifest(manifest_path, manifest)

        if args.dry_run:
            break

        finished_gpus = []
        for gpu_id, (experiment, job) in running.items():
            return_code = job["process"].poll()
            if return_code is None:
                continue

            close_files(job)
            finished_gpus.append(gpu_id)
            results_json = Path(job["results_json"])
            if return_code == 0 and results_json.exists():
                experiment["status"] = "completed"
                experiment["note"] = f"Completed successfully. Results: {results_json}"
                print(f"[DONE][GPU {gpu_id}] {experiment['dataset']} {experiment['backbone']} {experiment['method']}")
            else:
                experiment["status"] = "failed"
                experiment["note"] = (
                    f"Exit code {return_code}. Check {job['stdout_path']} and {job['stderr_path']}."
                )
                print(f"[FAIL][GPU {gpu_id}] {experiment['dataset']} {experiment['backbone']} {experiment['method']}")

        for gpu_id in finished_gpus:
            del running[gpu_id]
        if finished_gpus:
            save_manifest(manifest_path, manifest)

        if running:
            time.sleep(args.poll_seconds)

    save_manifest(manifest_path, manifest)
    status_counts = {}
    for experiment in manifest["experiments"]:
        status_counts[experiment["status"]] = status_counts.get(experiment["status"], 0) + 1

    print("Final status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
