# Baseline-controlled Donation Bet analysis

This folder contains the reproducible analysis for three hypotheses about motivated reasoning:

- H1: explicitly verbalized value steering
- H2: implicit selective reasoning
- H3: early value steering / rationalization

Models:

- Qwen 3.5 122B A10B
- Claude Opus 4.7
- DeepSeek V4 Pro

The baseline condition is used as the control in every analysis. The script reads the existing run files under `runs/` and writes the JSON, rollout-level CSV, and H1 audit TSV included here.

Run from the repository root with a Python environment containing NumPy:

```bash
python analysis/three_hypothesis_experiment/analyze_three_hypotheses.py
```

The analysis is observational. Sentence resampling is the proposed causal follow-up; Qwen J-lens is an optional follow-up for the early-steering result.
