# DFlash Training Recipe for Gemma4-21B-REAP

**Purpose:** Complete training guide for creating a DFlash draft model for your fine-tuned Gemma4-21B-REAP model.

**Status:** Ready to implement after base training completes.

---

## Overview

This document reverse-engineers the DFlash training recipe from production models and adapts it for Gemma4-21B architecture.

**Source Models Analyzed:**
- `z-lab/Qwen3-8B-DFlash-b16` (config extracted)
- `z-lab/Qwen3.5-4B-DFlash` (reference)
- `z-lab/Qwen3-8B-DFlash-b16` source code

**Target Architecture:**
```
Base Model: 0xSero/gemma-4-21b-a4b-it-REAP (your fine-tuned model)
Draft Model: Custom DFlash draft with 5 layers
Speedup: 2.5x faster inference via speculative decoding
```

---

## Architecture Specifications

### Base Model (Gemma4-21B-REAP)
```yaml
Layers: 42
Hidden Size: 3072
Attention Heads: 16
Key-Value Heads: 4
Head Dim: 256
Intermediate Size: 24576
Max Sequence Length: 8192 (Gemma4 native)
```

### Draft Model Configuration
```yaml
# Adapted from Qwen3-8B-DFlash-b16 config.json
Layers: 5  # Reduced from 42 (12% of base)
Hidden Size: 3072  # Same as base
Attention Heads: 16  # Same as base
Key-Value Heads: 4  # Same as base
Head Dim: 256  # Same as base
Intermediate Size: 24576  # Same as base
Block Size: 16  # Tokens per speculative block
```

### Target Layer Selection (Critical)
```python
# Evenly distributed conditioning layers
# Formula: spread 5 points across [1, 38] (exclude first and last 3 layers)

def build_target_layer_ids(num_target_layers=42, num_draft_layers=5):
    """
    Select target layers for conditioning.
    Formula from dflash/model.py line 98-107
    """
    start = 1
    end = num_target_layers - 3  # 39
    span = end - start  # 38
    
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]

# Result for Gemma4-21B:
target_layer_ids = [1, 10, 20, 29, 38]
```

---

## Model Configuration JSON

```json
{
  "architectures": ["Gemma4DFlashDraftModel"],
  "hidden_size": 3072,
  "num_hidden_layers": 5,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "head_dim": 256,
  "intermediate_size": 24576,
  "max_position_embeddings": 8192,
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000.0,
  "attention_dropout": 0.0,
  "hidden_act": "gelu_pytorch_tanh",
  "initializer_range": 0.02,
  "use_cache": true,

  "block_size": 16,
  "num_target_layers": 42,
  "mask_token_id": 256000,

  "dflash_config": {
    "mask_token_id": 256000,
    "target_layer_ids": [1, 10, 20, 29, 38]
  },

  "transformers_version": "4.46.0"
}
```

**Note:** `mask_token_id` should be a valid unused token ID in Gemma4 vocabulary. Gemma4 tokenizer has 256K+ tokens, so 256000 is safe.

---

## Training Hyperparameters

### Extrapolated from Production Models

```yaml
# Training Configuration
Training Data: Same hybrid agent dataset as base model
Data Format: Same format (turns with system/user/assistant roles)

# Optimizer Settings
Optimizer: adamw_torch
Learning Rate: 2.0e-5
Weight Decay: 0.01
Warmup Ratio: 0.03
LR Scheduler: cosine
Beta1: 0.9
Beta2: 0.999
Epsilon: 1.0e-8

# Batch Configuration
Per Device Batch Size: 2  # Adjust based on GPU
Gradient Accumulation: 8
Effective Batch: 16
Max Seq Length: 4096  # Can go to 8192 if needed

# Training Duration
Num Epochs: 3
Total Steps: ~1500 (depends on dataset size)

# Regularization
Warmup Steps: 45  # 3% of 1500
Logging Steps: 10
Save Steps: 100
Eval Steps: 100

# Precision
BF16: true  # Use bfloat16 on H100
Gradient Checkpointing: true
```

---

## Training Script

```python
"""
train_gemma4_dflash.py

Train DFlash draft model for Gemma4-21B-REAP.

Usage:
    python train_gemma4_dflash.py \
        --base_model_path ./outputs/gemma-4-21b-reap-merged \
        --output_dir ./outputs/gemma4-dflash-draft \
        --data_path ./hybrid_dataset/train.jsonl \
        --num_train_epochs 3
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Config,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
)
from transformers.modeling_utils import PreTrainedModel
from torch import nn


# ---------------------------------------------------------------------------
# DFlash Model Architecture for Gemma4
# ---------------------------------------------------------------------------

@dataclass
class DFlashConfig:
    """DFlash-specific configuration."""
    block_size: int = 16
    num_target_layers: int = 42
    mask_token_id: int = 256000
    target_layer_ids: List[int] = field(default_factory=lambda: [1, 10, 20, 29, 38])


class Gemma4DFlashAttention(nn.Module):
    """Cross-attention layer for DFlash draft model.

    Key difference from standard attention:
    - Attends to target model's hidden states (cross-attention)
    - Concatenates target context with draft positions
    """

    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Project queries from draft hidden states
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Project keys and values from concatenated [target, draft]
        k_target = self.k_proj(target_hidden)
        k_draft = self.k_proj(hidden_states)
        v_target = self.v_proj(target_hidden)
        v_draft = self.v_proj(hidden_states)

        k = torch.cat([k_target, k_draft], dim=1)
        v = torch.cat([v_target, v_draft], dim=1)

        k = k.view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(q, position_ids)
        q_embed = (q * cos[:, :, :q_len, :]) + (self.rotate_half(q) * sin[:, :, :q_len, :])
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        # Repeat KV for multi-group attention
        k_embed = torch.repeat_interleave(k_embed, self.num_key_value_groups, dim=1)
        v_embed = torch.repeat_interleave(v_embed, self.num_key_value_groups, dim=1)

        # Compute attention
        attn_weights = (q_embed @ k_embed.transpose(-2, -1)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ v_embed

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        return self.o_proj(attn_output)

    @staticmethod
    def rotate_half(x):
        """Rotates half of the hidden dims for rotary position embedding."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class Gemma4DFlashDecoderLayer(nn.Module):
    """DFlash draft decoder layer with cross-attention."""

    def __init__(self, config: Gemma3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Gemma4DFlashAttention(config, layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention (cross-attention to target)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma4DFlashDraftModel(PreTrainedModel):
    """Complete DFlash draft model for Gemma4.

    Architecture:
    1. Project target hidden states (from conditioning layers)
    2. Concat with noise embedding (masked draft tokens)
    3. Denoise through 5 cross-attention layers
    4. Predict next token block (16 tokens)
    """

    config_class = Gemma3Config

    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.config = config

        # Get DFlash-specific config
        dflash_config = config.dflash_config if hasattr(config, 'dflash_config') else {}
        self.block_size = dflash_config.get('block_size', 16)
        self.target_layer_ids = dflash_config.get('target_layer_ids', [1, 10, 20, 29, 38])
        self.mask_token_id = dflash_config.get('mask_token_id', 256000)

        # Draft layers (reduced from 42)
        self.layers = nn.ModuleList([
            Gemma4DFlashDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)  # 5 layers
        ])

        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)

        # Projection: target hidden states -> draft hidden size
        # Input: concatenated features from target_layer_ids
        # Output: single hidden state vector
        self.target_projection = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False
        )
        self.hidden_norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head for token prediction
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        target_hidden_states: List[torch.Tensor],  # List from target model layers
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: Draft token IDs [batch, seq_len]
            target_hidden_states: List of hidden states from target model
                at layers specified by target_layer_ids
            position_ids: Position IDs [batch, seq_len]
            attention_mask: Attention mask [batch, 1, seq_len, seq_len]
            labels: Target token IDs for training [batch, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.hidden_size

        # 1. Extract and concatenate target hidden states
        selected_states = [
            target_hidden_states[i]  # [batch, target_seq_len, hidden_size]
            for i in self.target_layer_ids
        ]
        concatenated = torch.cat(selected_states, dim=-1)  # [batch, target_seq_len, 5*hidden_size]

        # 2. Project to draft hidden space
        target_hidden = self.target_projection(concatenated)  # [batch, target_seq_len, hidden_size]
        target_hidden = self.hidden_norm(target_hidden)

        # 3. Get draft embeddings (from input_ids)
        # Note: In actual inference, input_ids would be masked/noise tokens
        # For training, we use actual tokens from dataset
        hidden_states = self.get_input_embeddings()(input_ids)  # [batch, seq_len, hidden_size]

        # 4. Denoise through draft layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)

        # 5. Predict tokens
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {
            'loss': loss,
            'logits': logits,
        }


# ---------------------------------------------------------------------------
# Training Setup
# ---------------------------------------------------------------------------

def load_training_dataset(data_path: str, tokenizer, max_seq_length: int = 4096):
    """Load and format hybrid agent dataset for DFlash training."""

    def format_conversation(example):
        """Format conversation into chat template."""
        messages = example['turns']

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {'text': text}

    # Load dataset
    if data_path.endswith('.jsonl'):
        data = []
        with open(data_path) as f:
            for line in f:
                data.append(json.loads(line))
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset(data_path, split='train')

    # Format conversations
    dataset = dataset.map(format_conversation)

    # Tokenize
    def tokenize(example):
        return tokenizer(
            example['text'],
            max_length=max_seq_length,
            truncation=True,
            padding='max_length',
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    return tokenized


def train_dflash_model(
    base_model_path: str,
    output_dir: str,
    data_path: str,
    num_train_epochs: int = 3,
    per_device_batch_size: int = 2,
    gradient_accumulation: int = 8,
    learning_rate: float = 2e-5,
    max_seq_length: int = 4096,
):
    """Train DFlash draft model."""

    # Load tokenizer from base model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Create draft model config
    print("Creating draft model config...")
    base_config = Gemma3Config.from_pretrained(base_model_path)

    # Override for DFlash draft
    draft_config = Gemma3Config(
        # Architecture
        hidden_size=base_config.hidden_size,
        num_hidden_layers=5,  # Draft layers
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        head_dim=base_config.head_dim,
        intermediate_size=base_config.intermediate_size,

        # Gemma4 specific
        rms_norm_eps=base_config.rms_norm_eps,
        rope_theta=base_config.rope_theta,
        attention_dropout=0.0,
        hidden_act='gelu_pytorch_tanh',
        initializer_range=0.02,

        # Vocab
        vocab_size=tokenizer.vocab_size,

        # Max positions
        max_position_embeddings=max_seq_length,

        # DFlash specific
        block_size=16,
        num_target_layers=base_config.num_hidden_layers,
        mask_token_id=256000,
        dflash_config={
            'block_size': 16,
            'num_target_layers': base_config.num_hidden_layers,
            'mask_token_id': 256000,
            'target_layer_ids': [1, 10, 20, 29, 38],
        },
    )

    # Initialize model
    print("Initializing DFlash draft model...")
    model = Gemma4DFlashDraftModel(draft_config)

    # Load dataset
    print("Loading training dataset...")
    train_dataset = load_training_dataset(data_path, tokenizer, max_seq_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        weight_decay=0.01,

        logging_steps=10,
        save_steps=100,
        save_total_limit=3,

        bf16=True,
        gradient_checkpointing=True,

        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,

        max_grad_norm=1.0,
        dataloader_num_workers=4,

        report_to=['wandb', 'tensorboard'],
        run_name=f'gemma4-dflash-{os.path.basename(base_model_path)}',

        ddp_find_unused_parameters=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")
    return trainer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train DFlash draft model for Gemma4-21B')
    parser.add_argument('--base_model_path', type=str, required=True,
                       help='Path to fine-tuned Gemma4-21B-REAP base model')
    parser.add_argument('--output_dir', type=str, default='./outputs/gemma4-dflash-draft',
                       help='Output directory for draft model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training dataset (JSONL or HuggingFace path)')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_seq_length', type=int, default=4096)

    args = parser.parse_args()

    train_dflash_model(
        base_model_path=args.base_model_path,
        output_dir=args.output_dir,
        data_path=args.data_path,
        num_train_epochs=args.num_train_epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
    )
```

---

## Training Execution

### Prerequisites

1. **Base Model Trained**: Your Gemma4-21B-REAP model must be fully trained and merged
2. **Dataset Ready**: Same hybrid agent dataset used for base training
3. **H100 Instance**: DFlash training requires GPU (H100 recommended)

### Step 1: Prepare Base Model

```bash
# Ensure your base model is merged and saved
# From previous training output, you should have:
ls -lh ./outputs/gemma-4-21b-reap-merged/

# Should show:
# - config.json
# - tokenizer.json
# - model.safetensors (or model-*.safetensors)
```

### Step 2: Launch Training

```bash
# Activate conda environment
conda activate gemma-h100

# Set environment variables
export WANDB_PROJECT="gemma4-dflash"
export WANDB_RUN_NAME="gemma4-dflash-$(date +%Y%m%d)"

# Run training
python train_gemma4_dflash.py \
    --base_model_path ./outputs/gemma-4-21b-reap-merged \
    --output_dir ./outputs/gemma4-dflash-draft \
    --data_path ./hybrid_dataset/train.jsonl \
    --num_train_epochs 3 \
    --per_device_batch_size 2 \
    --gradient_accumulation 8 \
    --learning_rate 2e-5 \
    --max_seq_length 4096
```

### Step 3: Monitor Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor logs
tail -f ./outputs/gemma4-dflash-draft/trainer.log

# Check W&B dashboard (if enabled)
# https://wandb.ai/<your-username>/gemma4-dflash
```

---

## Expected Training Metrics

### Performance Targets

Based on production DFlash models:

```yaml
# Loss
Initial Loss: 2.5 - 3.0
Final Loss: 1.5 - 1.8

# GPU Usage (H100)
GPU Memory: 25-35 GB (draft model is lightweight)
GPU Utilization: 85-95%
Training Speed: 4-6 samples/second

# Training Duration
3 epochs on 100K samples: ~6-8 hours on single H100
```

### Success Indicators

✅ **Training converges:**
- Loss decreases steadily
- No NaN values
- GPU utilization stable

✅ **Model learns patterns:**
- Loss plateaus at reasonable value
- Sample generations improve during training

✅ **Resource usage healthy:**
- GPU memory under 40GB
- No OOM errors
- Training speed >3 samples/sec

---

## Post-Training: Inference Integration

### Option 1: Transformers Backend (Development)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemma4_dflash import Gemma4DFlashDraftModel  # Your implementation

# Load models
base_model = AutoModelForCausalLM.from_pretrained(
    './outputs/gemma-4-21b-reap-merged',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0'
).eval()

draft_model = Gemma4DFlashDraftModel.from_pretrained(
    './outputs/gemma4-dflash-draft',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0'
).eval()

tokenizer = AutoTokenizer.from_pretrained('./outputs/gemma-4-21b-reap-merged')

# Generate with speculative decoding
messages = [{'role': 'user', 'content': 'Explain quantum computing'}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors='pt').to('cuda:0')

# Use draft model for fast generation
output_ids = draft_model.spec_generate(
    target=base_model,
    input_ids=input_ids,
    max_new_tokens=512,
    temperature=0.7,
)

output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)
```

### Option 2: vLLM Backend (Production)

**Note:** vLLM DFlash support is in nightly builds. Check documentation for latest integration.

```bash
# Install vLLM nightly
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly

# Start server with DFlash speculative decoding
vllm serve ./outputs/gemma-4-21b-reap-merged \
  --speculative-config '{
    "method": "dflash",
    "model": "./outputs/gemma4-dflash-draft",
    "num_speculative_tokens": 16
  }' \
  --dtype bfloat16 \
  --max-model-len 8192

# Test via API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./outputs/gemma-4-21b-reap-merged",
    "prompt": "Explain quantum computing",
    "max_tokens": 512
  }'
```

### Option 3: SGLang Backend (Alternative)

```bash
# Install SGLang with DFlash support
pip install "sglang[all]"

# Start server
python -m sglang.launch_server \
    --model-path ./outputs/gemma-4-21b-reap-merged \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path ./outputs/gemma4-dflash-draft \
    --speculative-num-draft-tokens 16 \
    --dtype bfloat16 \
    --context-length 8192

# Test
curl http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain quantum computing",
    "max_new_tokens": 512
  }'
```

---

## Performance Benchmarks

### Expected Speedup on Gemma4-21B

Based on Qwen3-8B production benchmarks:

```yaml
# Without DFlash (baseline)
Tokens/Second: ~25 t/s (H100, single GPU, bfloat16)
Time for 512 tokens: ~20 seconds

# With DFlash (speculative decoding)
Tokens/Second: ~60-70 t/s
Time for 512 tokens: ~7-8 seconds

Speedup: 2.5x - 2.8x
Acceptance Rate: 60-80% (varies by task)
```

### Task-Specific Performance

From DFlash paper benchmarks:

| Task | Acceptance Rate | Speedup |
|------|----------------|---------|
| Math (GSM8K) | 70-75% | 2.7x |
| Code (HumanEval) | 65-70% | 2.5x |
| Chat (MT-Bench) | 60-65% | 2.4x |
| General | 60-80% | 2.5x |

---

## Troubleshooting

### Issue: Model initialization fails

**Symptom:** Error loading model or config

**Solution:**
```bash
# Verify base model exists
ls -lh ./outputs/gemma-4-21b-reap-merged/config.json

# Verify tokenizer
python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('./outputs/gemma-4-21b-reap-merged'))"

# Check config matches Gemma4 architecture
python -c "from transformers import Gemma3Config; print(Gemma3Config.from_pretrained('./outputs/gemma-4-21b-reap-merged'))"
```

### Issue: Training loss is NaN

**Symptom:** Loss becomes NaN after first few steps

**Solutions:**
1. Lower learning rate: `--learning_rate 1e-5`
2. Enable gradient clipping (already in script: `max_grad_norm=1.0`)
3. Check for tokenizer padding issues
4. Reduce batch size if OOM occurred

### Issue: Out of Memory

**Symptom:** CUDA OOM during training

**Solutions:**
```bash
# Reduce batch size
--per_device_batch_size 1  # Was 2

# Increase gradient accumulation
--gradient_accumulation 16  # Was 8

# Reduce sequence length
--max_seq_length 2048  # Was 4096

# Enable gradient checkpointing (already enabled)
```

### Issue: Slow training

**Symptom:** Training speed <2 samples/second

**Solutions:**
1. Verify H100 is being used: `nvidia-smi`
2. Check Flash Attention is enabled
3. Increase dataloader workers: `dataloader_num_workers=8`
4. Disable wandb logging temporarily: `--report_to none`

### Issue: Poor generation quality

**Symptom:** Draft model generates poor tokens, low acceptance rate

**Solutions:**
1. Train for more epochs: `--num_train_epochs 5`
2. Increase dataset size (include more diverse examples)
3. Verify training data matches base training data
4. Lower learning rate and train longer

---

## Validation & Testing

### 1. Model Validation

```python
# After training completes, validate model loads correctly
from transformers import AutoConfig

config = AutoConfig.from_pretrained('./outputs/gemma4-dflash-draft')
print("Model architecture:", config.architectures)
print("Num layers:", config.num_hidden_layers)
print("Block size:", config.block_size)
print("Target layer IDs:", config.dflash_config['target_layer_ids'])

# Expected output:
# Model architecture: ['Gemma4DFlashDraftModel']
# Num layers: 5
# Block size: 16
# Target layer IDs: [1, 10, 20, 29, 38]
```

### 2. Generation Quality Test

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load models
base = AutoModelForCausalLM.from_pretrained(
    './outputs/gemma-4-21b-reap-merged',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0'
).eval()

tokenizer = AutoTokenizer.from_pretrained('./outputs/gemma-4-21b-reap-merged')

# Test prompt
messages = [{'role': 'user', 'content': 'What is the capital of France?'}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors='pt').to('cuda:0')

# Generate (without draft for baseline)
output_baseline = base.generate(input_ids, max_new_tokens=100)
print("Baseline:", tokenizer.decode(output_baseline[0]))

# TODO: Test with draft model once spec_generate is implemented
```

### 3. Performance Benchmark

```python
import time

# Benchmark baseline (no draft)
start = time.time()
output_baseline = base.generate(input_ids, max_new_tokens=512)
baseline_time = time.time() - start

# Benchmark with DFlash (once implemented)
# start = time.time()
# output_dflash = draft_model.spec_generate(...)
# dflash_time = time.time() - start

print(f"Baseline time: {baseline_time:.2f}s")
# print(f"DFlash time: {dflash_time:.2f}s")
# print(f"Speedup: {baseline_time / dflash_time:.2f}x")
```

---

## Next Steps After DFlash Training

### 1. Full Integration
- Implement `spec_generate` method in draft model
- Test acceptance rate on validation set
- Optimize block size (try 8, 16, 32)

### 2. Production Deployment
- Set up vLLM or SGLang server with DFlash
- Benchmark end-to-end latency
- Monitor acceptance rate in production

### 3. Iterate & Improve
- Experiment with different draft layer counts (3, 5, 7)
- Try different target layer selection strategies
- A/B test against baseline model

---

## Summary Checklist

**Before Training:**
- [ ] Base model (Gemma4-21B-REAP) fully trained and merged
- [ ] Dataset prepared (same as base training)
- [ ] H100 instance provisioned
- [ ] Training script (`train_gemma4_dflash.py`) created
- [ ] Environment dependencies installed

**During Training:**
- [ ] Loss decreases steadily
- [ ] GPU memory <40GB
- [ ] Training speed >3 samples/sec
- [ ] No NaN values in loss

**After Training:**
- [ ] Model saves successfully
- [ ] Config validated
- [ ] Generation quality tested
- [ ] Performance benchmarked
- [ ] Integration with vLLM/SGLang tested

---

## Resources

- **DFlash Paper**: https://arxiv.org/abs/2602.06036
- **DFlash GitHub**: https://github.com/z-lab/dflash
- **Production Models**: https://huggingface.co/collections/z-lab/dflash
- **vLLM DFlash Docs**: https://docs.vllm.ai/en/latest/speculative/dflash.html
- **SGLang DFlash Guide**: https://sglang.ai/docs/speculative_decoding

---

**Last Updated:** 2026-04-07
**Status:** Ready for implementation after base training completes
**Estimated Training Time:** 6-8 hours on single H100
