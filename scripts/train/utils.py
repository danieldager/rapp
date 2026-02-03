"""Utility functions for training."""

from datetime import datetime
from pathlib import Path
import torch


def get_model_timestamp():
    """Get timestamp for model naming (format: DDmmmHH, e.g., 03feb14)."""
    return datetime.now().strftime("%d%b%H").lower()


def find_latest_model_dir(base_pattern, weights_dir="./weights"):
    """Find most recent model directory matching a pattern.
    
    Args:
        base_pattern: Base name (e.g., "lstm_h256_l2_d0.0")
        weights_dir: Weights directory path
        
    Returns:
        Path to most recent matching directory, or None if not found
    """
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        return None
    
    matching_dirs = [
        d for d in weights_path.iterdir()
        if d.is_dir() and d.name.startswith(base_pattern)
    ]
    
    if not matching_dirs:
        return None
    
    matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(matching_dirs[0])


def load_latest_model(arch="lstm", checkpoint=None, weights_dir="./weights"):
    """Load most recent model of given architecture.
    
    Args:
        arch: "lstm" or "gpt2"
        checkpoint: Checkpoint name (e.g., "checkpoint-1000") or None
        weights_dir: Weights directory path
        
    Returns:
        (model, config, model_path)
    """
    from transformers import AutoModelForCausalLM, AutoConfig
    from scripts.train.models import LSTM, LSTMConfig
    
    base_pattern = "lstm_h256_l2_d0.0" if arch == "lstm" else "gpt2_e768_l12_h12"
    
    model_dir = find_latest_model_dir(base_pattern, weights_dir)
    if model_dir is None:
        raise FileNotFoundError(f"No models found matching: {base_pattern}")
    
    print(f"Loading from: {model_dir}")
    model_path = f"{model_dir}/{checkpoint}" if checkpoint else model_dir
    
    if arch == "lstm":
        config = LSTMConfig.from_pretrained(model_path)
        model = LSTM.from_pretrained(model_path)
    elif arch == "gpt2":
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return model, config, model_path


def print_training_summary(config, args, model, max_tokens):
    """Print training configuration summary."""
    total_params = sum(p.numel() for p in model.parameters())
    param_size_mb = (total_params * (2 if args.bf16 else 4)) / (1024**2)
    
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    tokens_per_step = effective_batch * (config.n_positions if hasattr(config, "n_positions") else max_tokens)
    
    model_type = "GPT2" if hasattr(config, "n_layer") else "LSTM"
    
    print(f"""
{'='*70}
{model_type} TRAINING SUMMARY
{'='*70}

MODEL:
  Parameters:      {total_params/1e6:.2f}M
  Weight Size:     {param_size_mb:.2f} MB ({'BF16' if args.bf16 else 'FP32'})

BATCH:
  GPUs:            {world_size}
  Per Device:      {args.per_device_train_batch_size}
  Accumulation:    {args.gradient_accumulation_steps}
  Effective:       {effective_batch}
  Tokens/Step:     {tokens_per_step/1e6:.2f}M

OPTIMIZATION:
  Optimizer:       {args.optim}
  Learning Rate:   {args.learning_rate}
  Betas:           ({args.adam_beta1}, {args.adam_beta2})
  Weight Decay:    {args.weight_decay}
  Grad Clip:       {args.max_grad_norm}

SCHEDULE:
  Type:            {args.lr_scheduler_type}
  Warmup Steps:    {args.warmup_steps}
  Max Steps:       {args.max_steps}

CHECKPOINTING:
  Save Every:      {args.save_steps} steps
  Eval Every:      {args.eval_steps} steps
  Keep Last:       {args.save_total_limit}
  Output:          {args.output_dir}

{'='*70}
""")
