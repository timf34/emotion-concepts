"""Single source of truth: paths, model registry, emotion sets, run defaults.

Layer convention used everywhere in this package
------------------------------------------------
``layer = b`` means the residual stream *after* decoder block ``b`` (0-based),
i.e. ``hidden_states[b + 1]`` in HuggingFace terms. Anthropic's "about
two-thirds of the way through the model" is therefore ``round(2/3 * n_blocks) - 1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
EVALS_DIR = DATA_DIR / "evals"

# On RunPod the network volume is /workspace; keep bulky outputs there when present.
_ws = Path("/workspace")
RESULTS_DIR = Path(os.environ.get("DPROBE_RESULTS", (_ws / "dprobe_results") if _ws.exists() else (PROJECT_ROOT / "results")))
CACHE_DIR = RESULTS_DIR / "openrouter_cache"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    openrouter_id: str | None          # None -> not hosted; generate locally with vLLM
    n_blocks: int
    hidden: int
    family: str                        # "gemma3" | "gemma4"
    is_base: bool = False
    # If a base model has no OpenRouter presence, borrow stories from this key
    # (Anthropic applied the same probes to base and post-trained models).
    stories_from: str | None = None

    @property
    def two_thirds_layer(self) -> int:
        return round(2 / 3 * self.n_blocks) - 1


MODELS: dict[str, ModelSpec] = {
    "gemma3_27b": ModelSpec("gemma3_27b", "google/gemma-3-27b-it", "google/gemma-3-27b-it", 62, 5376, "gemma3"),
    "gemma3_27b_pt": ModelSpec("gemma3_27b_pt", "google/gemma-3-27b-pt", None, 62, 5376, "gemma3", is_base=True, stories_from="gemma3_27b"),
    "gemma4_31b": ModelSpec("gemma4_31b", "google/gemma-4-31B-it", "google/gemma-4-31b-it", 60, 5376, "gemma4"),
    # cheap smoke-test target for the GPU pipeline (spirals too, score 9 in the paper's table)
    "gemma3_12b": ModelSpec("gemma3_12b", "google/gemma-3-12b-it", "google/gemma-3-12b-it", 48, 3840, "gemma3"),
}


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(MODELS)}")
    return MODELS[key]


# ---------------------------------------------------------------------------
# Emotion sets
# ---------------------------------------------------------------------------
# Word-level emotions (all on Anthropic's 171 list; stories via data/prompts/story_generation.yaml).
# Chosen to tile the spiral's vocabulary (worthlessness, apology, pleading, panic, exhaustion)
# plus low-arousal negative affect, positive/calm anchors, and neighbours for specificity tests.
EMOTIONS: list[str] = [
    # low mood
    "depressed", "sad", "gloomy", "melancholy", "miserable", "dispirited", "listless",
    "weary", "worn out", "tired", "resigned", "lonely", "heartbroken", "grief-stricken",
    # self-worth / shame
    "worthless", "self-critical", "ashamed", "guilty", "humiliated", "embarrassed", "regretful", "sorry",
    # high-arousal distress
    "desperate", "panicked", "hysterical", "overwhelmed", "distressed", "tormented", "trapped", "stuck",
    # frustration / anger
    "frustrated", "exasperated", "angry",
    # anxiety
    "anxious", "afraid", "nervous", "stressed",
    # positive / calm anchors
    "calm", "self-confident", "content", "happy", "hopeful",
]

# Syndrome-level clusters (data/prompts/syndromes.yaml). The clinical set is the target;
# the others are matched controls that each share a surface feature with depression.
SYNDROMES: list[str] = [
    "clinical_depression",
    "acute_grief",
    "burnout_exhaustion",
    "physical_illness_fatigue",
    "anxiety_panic",
    "frustration_blocked_goal",
]

# Emotions we plot / correlate by default (subset of EMOTIONS + syndromes + pain axis).
FOCUS: list[str] = [
    "depressed", "worthless", "sad", "desperate", "panicked", "frustrated", "ashamed",
    "hopeful", "calm", "self-confident", "clinical_depression", "pain_axis",
]


@dataclass
class StoryGenConfig:
    n_per_call: int = 6            # stories per (emotion, topic) request
    max_tokens: int = 6000
    temperature: float = 1.0
    max_concurrency: int = 48
    neutral_n_per_topic: int = 12
    min_story_chars: int = 300


@dataclass
class SpiralConfig:
    """Gemma Needs Help 'Extended' condition: base prompt + 7 neutral rejections = 8 model turns."""
    rollouts: int = 200            # paper: 200 conversations for the 8-turn condition
    extra_puzzle_rollouts: int = 10  # per puzzle in impossible_numeric (10 puzzles) for diversity
    temperature: float = 1.0
    max_tokens: int = 2048
    max_concurrency: int = 32


@dataclass
class JudgeConfig:
    model: str = "anthropic/claude-sonnet-4"   # exact judge used in the paper (still on OpenRouter)
    temperature: float = 0.0
    max_tokens: int = 600
    max_concurrency: int = 24


@dataclass
class ExtractConfig:
    start_at_nth_token: int = 50   # Anthropic: average from the 50th token
    batch_size: int = 8
    max_length: int = 1024
    neutral_pca_variance: float = 0.5
    neutral_tokens_for_pca: int = 40000   # subsample of neutral tokens per layer for the PCA
    # Layers at which we denoise + keep neutral stats + probe transcripts (raw vectors are kept at ALL layers).
    analysis_layer_stride: int = 2
    analysis_layer_min: int = 6


def analysis_layers(spec: ModelSpec, cfg: ExtractConfig | None = None) -> list[int]:
    cfg = cfg or ExtractConfig()
    layers = list(range(cfg.analysis_layer_min, spec.n_blocks, cfg.analysis_layer_stride))
    for must in (spec.two_thirds_layer, spec.n_blocks - 1):
        if must not in layers:
            layers.append(must)
    return sorted(set(layers))


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
