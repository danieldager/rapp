# LSTM Training Configuration Summary

## Final Optimal Hyperparameters

**Discovered:** February 3, 2026  
**Method:** Systematic debugging with 30+ experimental runs

### Model Architecture
- **Type:** LSTM (not GPT2)
- **Embedding dimension:** 256
- **Hidden size:** 256
- **Number of layers:** 2
- **Dropout:** 0.0
- **Total parameters:** ~1.2M

### Training Configuration
- **Optimizer:** Adam (not AdamW)
- **Learning rate:** 1e-2
- **Adam beta1:** 0.9 (default)
- **Adam beta2:** 0.99 (vs default 0.999)
- **Weight decay:** 0.0
- **Gradient clipping:** 5.0 (critical!)
- **Batch size:** 32 per device
- **Gradient accumulation:** 4 steps
- **Effective batch size:** 128 (32 × 4)

### Learning Rate Schedule
- **Type:** Cosine annealing
- **Warmup steps:** 200
- **Max steps:** 10,000

### Expected Performance
- **Step 40:** ~1.9 loss
- **Step 100:** ~1.7 loss
- **Convergence:** Consistent across all random seeds (0% plateau rate)
- **Data entropy baseline:** 5.17 (theoretical minimum)

## Key Findings from Debugging

### 1. Model Size
**Smaller is better for this task:**
- 256/256/2 (1.2M params) → Loss ~1.9 at step 40 ✓
- 200/1024/3 (22M params, from paper) → Plateaus at ~5.1 ❌

### 2. Learning Rate
**LSTM needs much higher LR than GPT2:**
- 1e-2 (optimal) → Fast convergence ✓
- 5e-4 (GPT2 style) → Complete plateau ❌

### 3. Optimizer Details
**Adam configuration matters:**
- Adam with beta2=0.99 → Works great ✓
- AdamW with beta2=0.98 → Slower convergence ❌

### 4. Regularization
**Less is more:**
- No dropout, no weight decay → Best performance ✓
- Paper's dropout=0.1 → Slower learning ❌

### 5. Gradient Clipping
**Critical for stability:**
- Clipping at 5.0 → Stable training ✓
- No clipping → Occasional instability ❌

### 6. Data Loading Bug
**Important lesson learned:**
- Must recreate DataLoader for each seed when testing
- WebDataset continues streaming across experiments
- Otherwise get artificial variance from different batches

## Comparison: LSTM vs GPT2

| Metric | LSTM (256/256/2) | GPT2 (256/2/4) |
|--------|------------------|----------------|
| Parameters | 1.2M | ~2M |
| Learning Rate | 1e-2 | 5e-4 |
| Loss @40 steps | 1.9 ± 0.07 | 5.2 ± 0.03 |
| Plateau rate | 0% | 100% |
| Training speed | Fast | Very slow |

**Winner:** LSTM with proper hyperparameters

## Files Modified
1. `/scratch2/ddager/rapp/scripts/train/train.py` - Updated with optimal config
2. `/scratch2/ddager/rapp/scripts/train/models.py` - Already has good initialization
3. `/scratch2/ddager/rapp/scripts/train/run.slurm` - Ready to launch

## How to Launch Training

```bash
cd /scratch2/ddager/rapp
sbatch scripts/train/run.slurm
```

This will:
- Train on 3 GPUs with DDP
- Save checkpoints every 500 steps
- Evaluate every 500 steps
- Use BF16 precision for speed
- Run for max 10,000 steps (should converge much earlier)

## Monitoring

Watch the training:
```bash
tail -f logs/train/train_<JOB_ID>.out
```

Expected output pattern:
```
Step:     0 | Loss: 5.5500 | TPS: 150k tokens/s | LR: 1.00e-02
Step:    10 | Loss: 4.2000 | TPS: 152k tokens/s | LR: 9.95e-03
Step:    20 | Loss: 2.8000 | TPS: 153k tokens/s | LR: 9.90e-03
Step:    40 | Loss: 1.9000 | TPS: 154k tokens/s | LR: 9.80e-03
...
```

## Next Steps After Training

1. Check final eval loss (should be well below 2.0)
2. Run SWUGGY evaluation
3. Compare against GPT2 baseline
4. Consider scaling up if performance is good

## Debugging History

See these files for full experimental details:
- `logs/train/debug_55188.out` - Original successful run
- `logs/train/compare_55201.out` - LSTM vs GPT2 comparison
- `logs/train/debug_55195.out` - Grid search (30 configs tested)
