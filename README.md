# DFlash Training Recipe

Training and evaluation recipe for DFlash draft models used in speculative decoding pipelines.

## Installation

2) Clone and enter the repo
```bash
git clone <repo-url>
cd <repo-name>
```

## Quick Start

```bash
# See repository-specific setup below
```

## Usage Examples

- Run default training workflow
```bash
python scripts/train.py  # adjust config path
```

## Implementation Overview

This repository is implemented primarily in **Mixed** and organized around explicit runtime entrypoints plus supporting modules.

### Key Directories

- `.github/`
- `docs/`

### Key Files

- `README.md`
- `LICENSE`
- `.github/workflows/ci.yml`

### Entrypoints


## Troubleshooting

- If startup fails, run the primary command with verbose flags and capture stderr logs.
- If dependencies conflict, remove lock artifacts and reinstall in a clean shell.
- If tests fail intermittently, run a single test target first, then full suite.
- Ensure environment variables are loaded before running build/train commands.

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
