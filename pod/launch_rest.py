"""Create a RunPod pod that runs pod/rest_job.sh unattended, through the RunPod REST API (no `rp` CLI needed).

    uv run python pod/launch_rest.py gptoss_120b                     # extract + axis, 2x H200 (else 2x B200, 4x H100)
    uv run python pod/launch_rest.py gptoss_120b --stages extract --gpus "NVIDIA H200:2"
    uv run python pod/launch_rest.py --status [name]                  # list pods (or one) without secrets
    uv run python pod/launch_rest.py --log dprobe-vec-gptoss-120b     # tail the log the pod pushes to the results dataset

The pod clones --branch of the public repo, gets HF_TOKEN and RUNPOD_API_KEY from .env as pod environment variables,
pushes results and its log to the results dataset (DPROBE_HF_REPO), and stops itself when the job ends.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
# bf16 gpt-oss-120b is ~234 GB: tried in order, first that has capacity wins
DEFAULT_GPUS = ["NVIDIA H200:2", "NVIDIA B200:2", "NVIDIA H100 80GB HBM3:4"]


def _req(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json",
                                          "User-Agent": "dprobe-launcher/1.0"})   # Cloudflare rejects urllib's default agent
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode()
            return json.loads(txt) if txt else {}
    except urllib.error.HTTPError as e:
        return {"error": e.code, "detail": e.read().decode()[:500]}


def status(name: str | None = None):
    pods = _req("GET", "/pods")
    if not isinstance(pods, list):
        sys.exit(f"RunPod API error: {pods}")
    for p in pods if isinstance(pods, list) else []:
        if name and p.get("name") != name:
            continue
        print(f"{p['id']}  {p['name']:28s} {p.get('desiredStatus'):8s} gpus={p.get('gpuCount')} ${p.get('costPerHr')}/h  started {p.get('lastStartedAt')}")


def log(name: str, n: int = 60):
    from huggingface_hub import hf_hub_download

    from dprobe.config import HF_RESULTS_REPO

    p = hf_hub_download(HF_RESULTS_REPO, f"logs/{name}/job.log", repo_type="dataset", token=os.environ.get("HF_TOKEN"), force_download=True)
    with open(p, errors="replace") as f:
        lines = f.read().replace("\r", "\n").splitlines()   # tqdm bars use \r
    print("\n".join(lines[-n:]))


def launch(model: str, stages: str, branch: str, repo: str, gpus: list[str], disk: int, name: str | None, skip_smoke: bool):
    name = name or f"dprobe-vec-{model.replace('_', '-')}"
    env = {"HF_TOKEN": os.environ["HF_TOKEN"], "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"], "MODEL": model,
           "STAGES": stages, "POD_NAME": name, "SHUTDOWN": "stop", "SKIP_SMOKE": "1" if skip_smoke else "0"}
    if os.environ.get("DPROBE_HF_REPO"):
        env["DPROBE_HF_REPO"] = os.environ["DPROBE_HF_REPO"]
    # the start command must never exit: RunPod restarts a container whose command ends (it would rerun the job)
    cmd = (f"git clone -q --depth 1 -b {branch} {repo} /dprobe && cd /dprobe && bash pod/rest_job.sh; "
           f"echo 'start command finished'; sleep infinity")
    for g in gpus:
        gtype, count = g.rsplit(":", 1)
        body = {"name": name, "imageName": IMAGE, "computeType": "GPU", "cloudType": "SECURE", "gpuTypeIds": [gtype],
                "gpuCount": int(count), "containerDiskInGb": disk, "volumeInGb": 0, "minRAMPerGPU": 100, "env": env,
                "dockerStartCmd": ["bash", "-lc", cmd], "ports": ["22/tcp"]}
        out = _req("POST", "/pods", body)
        if "id" in out:
            print(f"launched {name} ({out['id']}) on {count}x {gtype} at ${out.get('costPerHr')}/h")
            return out
        print(f"  {count}x {gtype}: {out.get('error')} {out.get('detail', '')[:200]}")
    sys.exit("no GPU configuration available")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--stages", default="extract,axis")
    ap.add_argument("--branch", default="crossfamily")
    ap.add_argument("--repo", default="https://github.com/timf34/emotion-concepts.git")
    ap.add_argument("--gpus", default=",".join(DEFAULT_GPUS), help="comma-separated TYPE:COUNT, tried in order")
    ap.add_argument("--disk", type=int, default=250)
    ap.add_argument("--name")
    ap.add_argument("--skip_smoke", action="store_true")
    ap.add_argument("--status", nargs="?", const="")
    ap.add_argument("--log")
    a = ap.parse_args()
    if a.status is not None:
        status(a.status or None)
    elif a.log:
        log(a.log)
    else:
        launch(a.model, a.stages, a.branch, a.repo, a.gpus.split(","), a.disk, a.name, a.skip_smoke)
