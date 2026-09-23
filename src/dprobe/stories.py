"""Story generation (laptop side, OpenRouter).

Three story sets per model, all written by the model itself (Anthropic's on-policy method):
  emotions/   one file per emotion word   {topic, text}
  syndromes/  one file per syndrome        {topic, text}
  neutral/    stories.json                  [text]

Outputs land in RESULTS_DIR/stories/<model_key>/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from dprobe.config import (
    DATA_DIR,
    EMOTIONS,
    PROMPTS_DIR,
    RESULTS_DIR,
    SYNDROMES,
    StoryGenConfig,
    get_model,
)
from dprobe.openrouter import OpenRouterClient, run

# Tolerates [Story 1], **Story 1:**, Story 1., ### story 1 ...  (from emotion-concepts-in-llms, MIT)
_STORY_HEADER_RE = re.compile(r"(?im)^[ \t]*#{0,6}[ \t]*[\*\[\(_]*[ \t]*story[ \t]*\d+[ \t]*[\*\]\)_:.\-]*[ \t]*$")


def parse_stories(response: str | None, min_chars: int = 300) -> list[str]:
    if not response:
        return []
    matches = list(_STORY_HEADER_RE.finditer(response))
    if not matches:
        body = response.strip()
        return [body] if len(body) >= min_chars else []
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
        body = response[m.end():end].strip()
        body = re.sub(r"^\*+\s*|\s*\*+$", "", body).strip()
        if len(body) >= min_chars:
            out.append(body)
    return out


def _load_topics() -> list[str]:
    with open(DATA_DIR / "topics.json") as f:
        return json.load(f)


def _gen_extra(spec) -> dict:
    # Gemma 4 IT exposes a thinking mode on some providers; keep it off so stories are plain text.
    return {"reasoning": {"enabled": False}} if spec.family == "gemma4" else {}


def _stories_dir(model_key: str) -> Path:
    d = RESULTS_DIR / "stories" / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _generate_set(
    client: OpenRouterClient,
    spec,
    labels: list[str],
    build_prompt,          # (label, topic, n) -> str
    out_dir: Path,
    cfg: StoryGenConfig,
    target_per_pair: int,
):
    topics = _load_topics()
    for label in labels:
        out_path = out_dir / f"{label.replace(' ', '_')}.json"
        existing: dict[str, list[str]] = {}
        if out_path.exists():
            with open(out_path) as f:
                for row in json.load(f):
                    existing.setdefault(row["topic"], []).append(row["text"])
        pending = [t for t in topics if len(existing.get(t, [])) < target_per_pair]
        if not pending:
            print(f"[stories] {spec.key}/{label}: complete ({sum(len(v) for v in existing.values())} stories)")
            continue
        print(f"[stories] {spec.key}/{label}: {len(pending)} topics pending")
        for attempt in range(3):
            if not pending:
                break
            msgs = [[{"role": "user", "content": build_prompt(label, t, target_per_pair - len(existing.get(t, [])))}] for t in pending]
            grid = await client.batch(
                spec.openrouter_id, msgs, n=1, temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                extra_body=_gen_extra(spec), progress_every=100,
            )
            still = []
            for t, row in zip(pending, grid):
                parsed = parse_stories(row[0], cfg.min_story_chars)
                if parsed:
                    existing.setdefault(t, []).extend(parsed[: target_per_pair - len(existing.get(t, []))])
                if len(existing.get(t, [])) < target_per_pair:
                    still.append(t)
            pending = still
            rows = [{"topic": t, "text": s} for t in topics for s in existing.get(t, [])]
            tmp = out_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(rows, f, indent=1)
            tmp.replace(out_path)          # atomic: a concurrent sync never sees a half-written file
            if pending and attempt < 2:
                # a retry must miss the cache: vary the prompt with a nonce so the key changes
                build_prompt_prev = build_prompt
                build_prompt = lambda l, t, n, _b=build_prompt_prev, _a=attempt: _b(l, t, n) + f"\n\n(variation {_a + 1})"
        n_total = sum(len(v) for v in existing.values())
        print(f"[stories] {spec.key}/{label}: {n_total} stories across {len(existing)} topics" + (f"; {len(pending)} topics short" if pending else ""))


def generate_emotion_stories(model_key: str, emotions: list[str] | None = None, cfg: StoryGenConfig | None = None):
    cfg = cfg or StoryGenConfig()
    spec = get_model(model_key)
    if spec.openrouter_id is None:
        raise ValueError(f"{model_key} is not on OpenRouter; use stories_from={spec.stories_from}")
    with open(PROMPTS_DIR / "story_generation.yaml") as f:
        tmpl = yaml.safe_load(f)["prompt"]
    emotions = emotions or EMOTIONS
    client = OpenRouterClient(max_concurrency=cfg.max_concurrency)

    def build(label, topic, n):
        return tmpl.format(n_stories=n, topic=topic, emotion=label)

    (_stories_dir(model_key) / "emotions").mkdir(exist_ok=True)
    run(_generate_set(client, spec, emotions, build, _stories_dir(model_key) / "emotions", cfg, cfg.n_per_call))


def generate_syndrome_stories(model_key: str, syndromes: list[str] | None = None, cfg: StoryGenConfig | None = None):
    cfg = cfg or StoryGenConfig()
    spec = get_model(model_key)
    with open(PROMPTS_DIR / "syndromes.yaml") as f:
        y = yaml.safe_load(f)
    tmpl, defs = y["prompt"], y["syndromes"]
    syndromes = syndromes or SYNDROMES
    client = OpenRouterClient(max_concurrency=cfg.max_concurrency)

    def build(label, topic, n):
        d = defs[label]
        return tmpl.format(n_stories=n, topic=topic, description=d["description"].strip(), banned=d["banned"])

    (_stories_dir(model_key) / "syndromes").mkdir(exist_ok=True)
    run(_generate_set(client, spec, syndromes, build, _stories_dir(model_key) / "syndromes", cfg, cfg.n_per_call))


def generate_neutral_stories(model_key: str, cfg: StoryGenConfig | None = None):
    cfg = cfg or StoryGenConfig()
    spec = get_model(model_key)
    with open(PROMPTS_DIR / "neutral_story_generation.yaml") as f:
        tmpl = yaml.safe_load(f)["prompt"]
    client = OpenRouterClient(max_concurrency=cfg.max_concurrency)

    def build(label, topic, n):
        return tmpl.format(n_stories=n, topic=topic)

    (_stories_dir(model_key) / "neutral").mkdir(exist_ok=True)
    run(_generate_set(client, spec, ["neutral"], build, _stories_dir(model_key) / "neutral", cfg, cfg.neutral_n_per_topic))


def load_story_set(model_key: str, kind: str) -> dict[str, list[str]]:
    """kind in {'emotions','syndromes','neutral'} -> {label: [text, ...]} (follows stories_from for base models)."""
    spec = get_model(model_key)
    src = spec.stories_from or model_key
    d = RESULTS_DIR / "stories" / src / kind
    out: dict[str, list[str]] = {}
    for p in sorted(d.glob("*.json")):
        try:
            with open(p) as f:
                rows = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[stories] WARNING skipping unreadable {p.name}: {e}")
            continue
        out[p.stem.replace("_", " ") if kind == "emotions" else p.stem] = [r["text"] for r in rows]
    if not out:
        raise FileNotFoundError(f"no {kind} stories under {d}")
    return out


def story_summary(model_key: str) -> dict:
    spec = get_model(model_key)
    src = spec.stories_from or model_key
    base = RESULTS_DIR / "stories" / src
    summary = {}
    for kind in ("emotions", "syndromes", "neutral"):
        d = base / kind
        if d.exists():
            summary[kind] = {p.stem: len(json.load(open(p))) for p in sorted(d.glob("*.json"))}
    return summary
