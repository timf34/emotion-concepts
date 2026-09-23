"""dprobe: emotion-concept probes for Gemma's distress spiral.

Laptop-side (no GPU): config, openrouter, stories, spiral, judge.
Pod-side (GPU):       models, extract, transcripts, probe, steer, selfother.
Both:                 analysis (numpy/pandas over saved tensors).
"""

__version__ = "0.1.0"
