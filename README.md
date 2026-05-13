# DFlash Training Recipe

Training and evaluation recipe for DFlash draft models used in speculative decoding pipelines.

## Presentation Framework (Proven README Pattern)

### TL;DR
End-to-end DFlash training/evaluation methodology for speculative decoding draft models.

### Why this project
- Solves a concrete workflow problem with reproducible command paths.
- Prioritizes operator reliability over demo-only output.
- Structured for practical use, not just conceptual documentation.

### Quick Start
```bash
less DFLASH_TRAINING_RECIPE.md
```

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install transformers datasets accelerate bitsandbytes trl peft
```

### Usage Examples
```bash
less DFLASH_TRAINING_RECIPE.md
python train_gemma4_dflash.py --help
```

### Architecture at a glance
- DFLASH_TRAINING_RECIPE.md — canonical method and hyperparameters
- DFLASH_README.md — context and rationale
- docs/ — publication assets and visuals

### Troubleshooting
- If quality regresses, re-check tokenizer/config parity between teacher and draft models.
- If VRAM blows up, tune batch/sequence before optimizer-level changes.

### Project status
Refine reproducibility and benchmark reporting for production-grade adoption.


## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install transformers datasets accelerate bitsandbytes trl peft
```

## Quick Start

```bash
# Primary guide:
less DFLASH_TRAINING_RECIPE.md

# Validate tokenizer/config compatibility before run:
python -c "from transformers import AutoTokenizer; print('ok')"
```

## Usage Examples

- Use the Gemma4 DFlash recipe as source of truth
```bash
less DFLASH_TRAINING_RECIPE.md
```

- Run training once script/config are prepared
```bash
python train_gemma4_dflash.py --help
```

- Serve/check merged checkpoint (example path)
```bash
python -m sglang.launch_server --help
```

## Implementation Overview

- `DFLASH_TRAINING_RECIPE.md` is the canonical end-to-end method and hyperparameter reference.
- `DFLASH_README.md` adds supporting context and rationale around the recipe.
- `.github/workflows/ci.yml` enforces basic markdown/repo quality checks.
- `docs/` holds generated visuals and publication assets.

## Troubleshooting

- If memory errors occur, reduce batch size and sequence length before changing optimizer settings.
- If speculative decoding quality regresses, re-check teacher/student tokenizer and config parity.
- If inference server startup fails, verify CUDA, driver, and framework version compatibility.

## Visual Overview

![dflash-training-recipe visual overview](docs/assets/visual-overview-dflash-training-recipe.svg)


## Problem
Serving costs and latency can dominate LLM production workloads.

## Method
- Document training assumptions for draft models
- Track acceptance rate, throughput, latency, and quality deltas
- Provide reproducible evaluation guidance

## Reproducibility
```bash
# Read DFLASH_ANALYSIS.md
# Run evals on your hardware and compare acceptance rate / tok-s / latency
```

## Limitations
Results vary by model family, hardware, and decoding parameters.

## Contributing

Contributions are welcome. Open an issue first for significant changes, then submit a focused PR with reproducible validation steps.

## License

See `LICENSE` for terms.
