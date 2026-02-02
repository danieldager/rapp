# sWuggy Evaluation

Evaluate language models on word-nonword classification using phonetically matched pairs.

The **sWuggy score** measures how well a language model distinguishes between real words and phonetically similar nonwords based on sequence probabilities.

## Usage

### 1. Prepare Dataset

```bash
sbatch scripts/swuggy/run.slurm utils
```

Loads raw parquet files with embedded audio, encodes to tokens using SPIDR, and saves to tar files.

**Output:** `tokens/swuggy_spidr_base/*.tar`, `metadata/swuggy.parquet`

### 2. Evaluate Model

```bash
sbatch scripts/swuggy/run.slurm evaluate
```

Calculates log probabilities for all samples.

**Edit config in script:**
```python
config = {
    "metadata_path": "metadata/swuggy.parquet",
    "tokens_dir": "tokens/swuggy_spidr_base",
    "checkpoint_path": "weights/lstm_h1024_l3_d0.1/checkpoint-1400",
    "output_path": "metadata/swuggy_evaluated.parquet",
    "device": "cuda",
}
```

**Output:** `metadata/swuggy_evaluated.parquet` with `log_prob` column

### 3. Calculate Scores

```bash
sbatch scripts/swuggy/run.slurm analysis
# or run locally:
python scripts/swuggy/analysis.py
```

Computes same-voice and cross-voice accuracy.

---

### 3. Analysis (Optional)

Perform detailed analysis on the results.

```bash
python scripts/swuggy/analysis.py
```

**Features:**
- Compare accuracy across different voices
- Analyze by word length / phoneme complexity

## Scoring Metrics

**Same-Voice**: Compares positive vs negative with matching voices (most conservative)

**Cross-Voice**: Compares all positives vs all negatives per word_id (more challenging)

**Per-Voice**: Same-voice accuracy broken down by individual voices

## How It Works

For each sequence, calculate log probability using next-token prediction:
1. Forward pass → get logits
2. Shift for autoregressive prediction
3. Log-softmax → probabilities
4. Sum log P(token_i | context)

Higher log probability = model thinks sequence is more likely to be a real word.
