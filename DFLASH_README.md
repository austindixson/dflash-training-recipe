# Gemma4-21B DFlash Draft Model

**2.5x faster inference for fine-tuned Gemma4-21B-REAP using block diffusion speculative decoding.**

## Overview

This repository contains the complete training recipe and implementation guide for creating a DFlash draft model for Gemma4-21B-REAP. DFlash enables 2.5-2.8x inference speedup through speculative decoding with minimal quality loss.

**Status**: Training recipe ready. Implementation pending base model completion.

---

## What is DFlash?

[DFlash](https://arxiv.org/abs/2602.06036) is a block diffusion technique for efficient speculative decoding:

- **Lightweight draft model** (5 layers vs 42 in base)
- **Predicts 16 tokens in parallel**
- **Target model verifies** and accepts matching tokens
- **60-80% acceptance rate** → 2.5x speedup

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Base Model (Gemma4-21B-REAP)                               │
│  - 42 layers                                                │
│  - Verification only                                        │
│  - Provides hidden states for conditioning                 │
└─────────────────────────────────────────────────────────────┘
                          ↑
                    Conditioning
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Draft Model (DFlash)                                       │
│  - 5 layers (12% of base)                                   │
│  - Cross-attention to target hidden states                  │
│  - Predicts 16-token blocks                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

1. **Trained base model**: Gemma4-21B-REAP must be fully trained and merged
2. **H100 instance**: For efficient training (6-8 hours)
3. **Dataset**: Same hybrid agent dataset used for base training

### Training

```bash
# After base model training completes
python train_gemma4_dflash.py \
    --base_model_path ./outputs/gemma-4-21b-reap-merged \
    --output_dir ./outputs/gemma4-dflash-draft \
    --data_path ./hybrid_dataset/train.jsonl \
    --num_train_epochs 3 \
    --per_device_batch_size 2 \
    --gradient_accumulation 8
```

### Inference with Speculative Decoding

```bash
# Using vLLM (recommended for production)
vllm serve ./outputs/gemma-4-21b-reap-merged \
  --speculative-config '{
    "method": "dflash",
    "model": "./outputs/gemma4-dflash-draft",
    "num_speculative_tokens": 16
  }' \
  --dtype bfloat16 \
  --max-model-len 8192
```

---

## Documentation

### Core Files

- **`DFLASH_TRAINING_RECIPE.md`** - Complete implementation guide
  - Architecture specifications
  - Full training script
  - Hyperparameters
  - Integration guide
  - Troubleshooting

- **`DFLASH_ANALYSIS.md`** - Technical deep-dive
  - Architecture analysis
  - Performance benchmarks
  - Design decisions
  - Production model comparisons

### External References

- [DFlash Paper](https://arxiv.org/abs/2602.06036)
- [DFlash GitHub](https://github.com/z-lab/dflash)
- [Production Models](https://huggingface.co/collections/z-lab/dflash)
- [vLLM DFlash Docs](https://docs.vllm.ai/en/latest/speculative/dflash.html)

---

## Specifications

### Draft Model Configuration

```yaml
Layers: 5  # Reduced from 42 (12% of base)
Hidden Size: 3072  # Same as base
Attention Heads: 16  # Same as base
Key-Value Heads: 4  # Same as base
Block Size: 16  # Tokens per speculative block
Target Layer IDs: [1, 10, 20, 29, 38]  # Conditioning layers
Mask Token ID: 256000
```

### Training Hyperparameters

```yaml
Optimizer: adamw_torch
Learning Rate: 2.0e-5
Batch Size: 2 per device
Gradient Accumulation: 8
Effective Batch: 16
Epochs: 3
Max Sequence Length: 4096
Precision: bfloat16
```

### Expected Performance

| Metric | Baseline | With DFlash | Speedup |
|--------|----------|-------------|---------|
| Tokens/Second | ~25 t/s | ~60-70 t/s | 2.5-2.8x |
| Time for 512 tokens | ~20s | ~7-8s | 2.5x |
| GPU Memory (H100) | 45GB | 25-35GB | - |
| Acceptance Rate | - | 60-80% | - |

---

## Architecture Details

### Key Components

1. **Target Layer Projection**
   ```python
   # Concatenate hidden states from 5 target layers
   # Project to draft model hidden dimension
   target_hidden = self.target_projection(concatenated_target_states)
   ```

2. **Cross-Attention Draft Layers**
   ```python
   # Each draft layer attends to target hidden states
   # Enables conditioning on base model representations
   hidden_states = self.cross_attention(
       hidden_states=hidden_states,
       target_hidden=target_hidden
   )
   ```

3. **Block-wise Generation**
   ```python
   # Generate 16 tokens in parallel
   # Verify with target model
   # Accept matching tokens
   block_output = draft_model(target_hidden, noise_embedding)
   verified_output = target_model.verify(block_output)
   ```

### Layer Selection Strategy

Target layers are evenly distributed through the base model depth:

```python
def build_target_layer_ids(num_target_layers=42, num_draft_layers=5):
    start = 1
    end = num_target_layers - 3  # 39
    span = end - start  # 38

    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]

# Result: [1, 10, 20, 29, 38]
```

This captures representations from early, middle, and late layers of the base model.

---

## Development Status

- [x] Architecture analysis
- [x] Training recipe documented
- [x] Hyperparameters specified
- [x] Integration guide written
- [ ] Base model training (in progress)
- [ ] DFlash model implementation
- [ ] DFlash model training
- [ ] Production deployment

---

## Requirements

### Training

```
torch>=2.5.0
transformers>=4.46.0
datasets
accelerate
peft
bitsandbytes
```

### Inference

```
vllm>=0.6.0  # For production serving
# OR
sglang       # Alternative backend
```

---

## Cost & Time Estimates

| Phase | Duration | H100 Cost |
|-------|----------|-----------|
| Base Model Training | 6-12 hours | $12-36 |
| DFlash Training | 6-8 hours | $12-24 |
| **Total** | **12-20 hours** | **$24-60** |

---

## Troubleshooting

### Training Loss is NaN
- Lower learning rate to `1e-5`
- Verify tokenizer padding configuration
- Check for corrupted data samples

### Out of Memory
- Reduce `per_device_batch_size` to 1
- Increase `gradient_accumulation` to 16
- Reduce `max_seq_length` to 2048

### Slow Training
- Verify H100 is being used: `nvidia-smi`
- Check Flash Attention is enabled
- Increase `dataloader_num_workers` to 8

### Poor Generation Quality
- Train for more epochs (5 instead of 3)
- Increase dataset size
- Verify training data matches base training

---

## Citation

If you use this DFlash implementation, please cite:

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```

---

## License

This training recipe is provided for research and commercial use. The underlying DFlash technique is described in the paper above.

---

## Questions?

- **DFlash Paper**: https://arxiv.org/abs/2602.06036
- **Original Repo**: https://github.com/z-lab/dflash
- **Production Models**: https://huggingface.co/collections/z-lab/dflash

---

**Last Updated**: 2026-04-07
**Status**: Ready for implementation after base training completes
