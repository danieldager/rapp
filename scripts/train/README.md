# Language Model Training

This directory contains training scripts for LSTM and GPT-2 language models on SPIDR tokenized speech data.

## Quick Start

```bash
# Train LSTM (default)
sbatch scripts/train/run.slurm

# Train LSTM (explicit)
sbatch scripts/train/run.slurm lstm

# Train GPT-2
sbatch scripts/train/run.slurm gpt2
```

## LSTM Training: Key Findings (Feb 2026)

After extensive debugging with 30+ experimental runs, we discovered the optimal configuration for LSTM convergence:

### Critical Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model Size** | 256/256/2 (emb/hidden/layers) | Smaller is better for this task |
| **Optimizer** | Adam (not AdamW!) | Critical - AdamW plateaus at ~5.1 loss |
| **Learning Rate** | 1e-2 | Much higher than typical (10-100x) |
| **Beta2** | 0.99 | Slightly lower than default 0.999 |
| **Gradient Clip** | 5.0 | Essential for stability |
| **Dropout** | 0.0 | Not needed |
| **Weight Decay** | 0.0 | Not needed |
| **LR Schedule** | Constant (no warmup) | Immediate high LR works best |
| **Batch Size** | 32×1×3 = 96 effective | Per-device × accum × GPUs |

### Results

- **Step 40**: Loss ~1.9
- **Step 100**: Loss ~1.7
- **Step 1000**: Loss ~1.65
- **0% plateau rate** (tested across 5 random seeds)

### Why AdamW Fails for LSTMs

Even with `weight_decay=0.0`, AdamW behaves differently than Adam:

- **AdamW**: Decoupled weight decay optimized for Transformers' dense gradient flow
- **Adam**: Coupled weight decay works better with LSTMs' sparse recurrent gradients

In our tests, Adam converged to ~1.9 loss by step 40, while AdamW plateaued at ~5.1 loss.

### Key Insights

1. **Smaller models converge better**: 256/256/2 >> 200/1024/3 (from paper)
2. **High LR is essential**: 1e-2 vs paper's 1e-4
3. **No warmup needed**: Model converges immediately with constant high LR
4. **Orthogonal initialization**: Helps but not as critical as optimizer choice
5. **Forget gate bias**: Initialize to 1.0 (standard LSTM practice)

### Debugging History

See `debug_compare_models.py` for the definitive experiment that confirmed these findings.

## GPT-2 Training Configuration

Configuration from paper:

| Parameter | Value |
|-----------|-------|
| **Model** | 768/12/12 (emb/layers/heads) |
| **Optimizer** | AdamW |
| **Learning Rate** | 1e-4 |
| **Beta2** | 0.98 |
| **Gradient Clip** | 0.0 (disabled) |
| **Weight Decay** | 0.01 |
| **LR Schedule** | inverse_sqrt with 1000 warmup |
| **Batch Size** | 32×4×3 = 384 effective |
| **Max Steps** | 100,000 |

## Model Loading

Load the most recent trained model:

```python
from scripts.train.utils import load_latest_model

# Load latest LSTM
model, config, path = load_latest_model("lstm")

# Load specific checkpoint
model, config, path = load_latest_model("lstm", checkpoint="checkpoint-5000")

# Load latest GPT-2
model, config, path = load_latest_model("gpt2", checkpoint="checkpoint-10000")
```

## Files

- **train.py**: Main training script with configuration dictionaries at top
- **run.slurm**: SLURM batch script (accepts architecture as first arg)
- **models.py**: LSTM and config classes with proper initialization
- **datasets.py**: WebDataset loaders for streaming tokenized data
- **utils.py**: Helper functions (model loading, training summary, timestamps)
- **debug_compare_models.py**: Preserved for posterity - shows LSTM vs GPT-2 comparison

## Data

- Training: `/scratch2/ddager/rapp/tokens/chunks30_spidr_base/`
- Evaluation: `/scratch2/ddager/rapp/tokens/chunks-eval_spidr_base/`
- Vocab size: 258 tokens (256 + BOS + EOS)
- Sequence length: 128 tokens

## Notes

- Models save with timestamps: `lstm_h256_l2_d0.0_03feb14`
- Checkpoints saved every 500 steps (LSTM) or 1000 steps (GPT-2)
- Early stopping patience: 5 (LSTM) or 3 (GPT-2)
- BF16 training enabled for speed
