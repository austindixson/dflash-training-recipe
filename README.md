# DFlash Training Recipe

**Complete guide for creating DFlash draft models to enable 2.5x faster LLM inference through speculative decoding.**

---

## 📚 Documentation

This repository contains comprehensive documentation for understanding and implementing DFlash speculative decoding:

### **[DFLASH_ANALYSIS.md](./DFLASH_ANALYSIS.md)**
Technical deep-dive into DFlash architecture
- Architecture overview and design principles
- Performance benchmarks and comparisons
- How DFlash achieves 2.5-2.8x speedup
- Production model analysis
- Use cases and applications

**Key topics:**
- Block diffusion mechanism
- Cross-attention conditioning
- Speculative verification
- Token acceptance rates (60-80%)
- Comparison to other acceleration techniques

### **[DFLASH_TRAINING_RECIPE.md](./DFLASH_TRAINING_RECIPE.md)**
Complete implementation guide for creating a DFlash draft model
- Full training script implementation
- Architecture specifications
- Hyperparameters and configuration
- Integration with vLLM/SGLang
- Production deployment

**Includes:**
- Gemma4-21B specific training recipe
- 5-layer draft model architecture
- 16-token block size configuration
- Target layer selection strategy
- Code examples and scripts

### **[DFLASH_README.md](./DFLASH_README.md)**
Quick reference and overview
- What is DFlash?
- How it works
- Quick start examples
- Performance expectations
- Links to resources

---

## 🎯 What is DFlash?

**DFlash** is a block diffusion technique for efficient speculative decoding:

- **Lightweight draft model** (5 layers vs 40+ in base)
- **Predicts 16 tokens in parallel**
- **Target model verifies** and accepts matching tokens
- **60-80% acceptance rate** → 2.5x speedup

### Architecture

```
┌─────────────────────────────────────────┐
│  Base Model (e.g., Gemma4-21B-REAP)       │
│  - 42 layers (verification only)          │
│  - Provides hidden states for conditioning │
└──────────────┬──────────────────────────────┘
               │ Conditioning
               ↓
┌─────────────────────────────────────────┐
│  Draft Model (DFlash)                    │
│  - 5 layers (12% of base)                │
│  - Cross-attention to target states      │
│  - Predicts 16-token blocks               │
└─────────────────────────────────────────┘
```

---

## 🚀 Use Cases

### 1. After Training Your Model

You've just finished training a large language model and want to deploy it with faster inference:

1. **Use this training recipe** to create a DFlash draft model
2. **Deploy with vLLM** for 2.5x faster inference
3. **Reduce latency** from 20s to 7-8s for 512-token responses

### 2. Understanding Speculative Decoding

Learn how modern LLM inference is accelerated beyond simple quantization:

- **Block diffusion** (DFlash's innovation)
- **Speculative verification** (accept/reject tokens)
- **Cross-attention conditioning** (how draft knows target context)

### 3. Research & Implementation

Study the architecture for your own implementations:

- **DFLASH_ANALYSIS.md**: Understand the theory
- **DFLASH_TRAINING_RECIPE.md**: Get implementation details
- **Adapt the training script** for your specific model

---

## 📊 Performance Benchmarks

| Metric | Baseline | With DFlash | Speedup |
|--------|----------|-------------|---------|
| Tokens/Second | ~25 t/s | ~60-70 t/s | **2.5-2.8x** |
| Time for 512 tokens | ~20s | ~7-8s | **2.5x** |
| GPU Memory (H100) | 45GB | 25-35GB | - |
| Acceptance Rate | - | 60-80% | - |

---

## 🔧 Quick Start

### 1. Understand DFlash

Read **[DFLASH_ANALYSIS.md](./DFLASH_ANALYSIS.md)** to understand:
- How block diffusion works
- Why it's faster than traditional methods
- Performance characteristics

### 2. Train Your Draft Model

Follow **[DFLASH_TRAINING_RECIPE.md](./DFLASH_TRAINING_RECIPE.md)** to:
- Configure your draft model architecture
- Train with 4-bit quantization
- Optimize for your base model

### 3. Deploy with vLLM

```bash
vllm serve your-model \
  --speculative-config '{
    "method": "dflash",
    "model": "your-dflash-model",
    "num_speculative_tokens": 16
  }' \
  --dtype bfloat16 \
  --max-model-len 8192
```

---

## 📖 Documentation Structure

```
dflash-training-recipe/
├── README.md (this file)
├── DFLASH_ANALYSIS.md          # Technical deep-dive
├── DFLASH_TRAINING_RECIPE.md   # Implementation guide
└── DFLASH_README.md             # Quick reference
```

**Reading Order:**
1. Start here (README.md)
2. DFLASH_README.md - Quick overview
3. DFLASH_ANALYSIS.md - Understand the architecture
4. DFLASH_TRAINING_RECIPE.md - Implement it yourself

---

## 🎓 Target Audience

- **ML Engineers** deploying LLMs with inference constraints
- **Researchers** studying speculative decoding techniques
- **DevOps Engineers** optimizing serving infrastructure
- **LLM Hobbyists** experimenting with model acceleration

---

## 🌟 Key Features

### DFLASH_ANALYSIS.md
- Architecture breakdown
- Performance comparison tables
- Acceptance rate analysis
- Real-world benchmarks
- Design decisions explained

### DFLASH_TRAINING_RECIPE.md
- Complete training script
- Model configuration specs
- Gemma4-21B example
- vLLM/SGLang integration
- Troubleshooting guide

### DFLASH_README.md
- What is DFlash?
- Architecture diagram
- Quick start examples
- Performance expectations
- Resource links

---

## 💡 Common Questions

**Q: Can I use DFlash with any model?**
A: DFlash works best with decoder-only transformer models. The training recipe shows how to adapt it for different architectures.

**Q: Do I need an H100 GPU?**
A: For training: H100 recommended (16-week timeline). For inference: Any GPU with sufficient VRAM for your base model.

**Q: How long does training take?**
A: 6-8 hours on H100 for the draft model, assuming base model is already trained.

**Q: What's the speedup in production?**
A: 2.5-2.8x faster than baseline, depending on task and acceptance rate.

---

## 📚 Resources

- [DFlash Paper](https://arxiv.org/abs/2602.06036)
- [DFlash GitHub](https://github.com/z-lab/dflash)
- [Production Models](https://huggingface.co/collections/z-lab/dflash)
- [vLLM DFlash Docs](https://docs.vllm.ai/en/latest/speculative/dflash.html)
- [SGLang Guide](https://sglang.ai/docs/speculative_decoding)

---

## 📝 Citation

If you use DFlash, please cite:

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```

---

## 📄 License

This documentation is provided for educational and commercial use. DFlash technique is described in the paper above.

---

**Repository:** austindixson/dflash-training-recipe
**Status:** ✅ Complete
**Last Updated:** 2026-04-08
**Purpose:** Complete DFlash training recipe for LLM acceleration
