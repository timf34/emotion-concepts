"""Move results between the laptop and pods through a private HF dataset (CPU work stays on the laptop,
GPU work on pods; pods never need the laptop's files except through this).

Subsets are top-level folders of RESULTS_DIR:
  inputs  for pods : stories, spiral (transcripts + judgments)
  outputs from pods: vectors, probe, selfother, analysis
The OpenRouter cache and logs are never synced.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from dprobe.config import HF_RESULTS_REPO, RESULTS_DIR

INPUT_SUBSETS = ("stories", "spiral")
OUTPUT_SUBSETS = ("vectors", "probe", "selfother", "steer")


def _api() -> HfApi:
    return HfApi(token=os.environ.get("HF_TOKEN"))


def sync_up(subsets: tuple[str, ...] = INPUT_SUBSETS, repo: str = HF_RESULTS_REPO, models: tuple[str, ...] | None = None):
    api = _api()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    for sub in subsets:
        root = RESULTS_DIR / sub
        if not root.exists():
            print(f"[sync] skip {sub}: not present")
            continue
        folders = [root] if models is None else [root / m for m in models if (root / m).exists()]
        for folder in folders:
            rel = folder.relative_to(RESULTS_DIR)
            print(f"[sync] up {folder} -> {repo}/{rel}")
            api.upload_folder(
                folder_path=str(folder), repo_id=repo, repo_type="dataset", path_in_repo=str(rel),
                ignore_patterns=["*.tmp", "*.log", "token_level*"], commit_message=f"sync {rel}",
            )


def sync_down(subsets: tuple[str, ...] = INPUT_SUBSETS, repo: str = HF_RESULTS_REPO, models: tuple[str, ...] | None = None):
    patterns = []
    for sub in subsets:
        if models is None:
            patterns.append(f"{sub}/**")
        else:
            patterns += [f"{sub}/{m}/**" for m in models] + [f"{sub}/{m}_*/**" for m in models]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[sync] down {repo} {patterns} -> {RESULTS_DIR}")
    snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns, local_dir=str(RESULTS_DIR), token=os.environ.get("HF_TOKEN"))
    for sub in subsets:
        p = RESULTS_DIR / sub
        print(f"[sync]   {sub}: {sum(1 for _ in p.rglob('*') if _.is_file()) if p.exists() else 0} files")
