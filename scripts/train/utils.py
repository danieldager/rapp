"""Utility functions for training."""

import torch


def print_training_summary(config, args, model, max_tokens):
    """Prints a comprehensive summary of the model and training parameters."""
    # Parameter Counting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Size Estimation (Weights only, approx)
    # BF16 = 2 bytes per param, FP32 = 4 bytes
    param_size_mb = (total_params * (2 if args.bf16 else 4)) / (1024**2)

    # Batch Logic
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
    )

    if hasattr(config, "n_positions"):
        tokens_per_batch = effective_batch_size * config.n_positions
    else:
        tokens_per_batch = effective_batch_size * max_tokens

    # Build model config details
    model_type = "GPT2" if hasattr(config, "n_layer") else "LSTM"
    config_dict = config.to_dict()
    
    # Filter out non-essential config keys for cleaner display
    exclude_keys = {'transformers_version', 'torch_dtype', '_name_or_path', 'architectures'}
    config_items = [(k, v) for k, v in sorted(config_dict.items()) 
                    if k not in exclude_keys and not k.startswith('_')]
    
    config_str = "\n    ".join([f"- {k:25s} = {v}" for k, v in config_items])

    # Build training args details
    # Group args by category
    optimization_args = {
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'max_grad_norm': args.max_grad_norm,
        'weight_decay': args.weight_decay,
        'adam_beta1': args.adam_beta1,
        'adam_beta2': args.adam_beta2,
        'adam_epsilon': args.adam_epsilon,
    }
    
    scheduling_args = {
        'lr_scheduler_type': args.lr_scheduler_type,
        'warmup_steps': args.warmup_steps,
        'max_steps': args.max_steps,
    }
    
    precision_args = {
        'bf16': args.bf16,
        'fp16': args.fp16,
        'dataloader_num_workers': args.dataloader_num_workers,
    }
    
    logging_args = {
        'logging_steps': args.logging_steps,
        'save_strategy': args.save_strategy,
        'save_steps': args.save_steps,
        'save_total_limit': args.save_total_limit,
        'eval_strategy': args.eval_strategy,
        'eval_steps': args.eval_steps,
        'metric_for_best_model': args.metric_for_best_model,
        'greater_is_better': args.greater_is_better,
        'load_best_model_at_end': args.load_best_model_at_end,
    }
    
    ddp_args = {
        'ddp_find_unused_parameters': args.ddp_find_unused_parameters,
        'remove_unused_columns': args.remove_unused_columns,
        'ignore_data_skip': args.ignore_data_skip,
    }

    opt_str = "\n    ".join([f"- {k:35s} = {v}" for k, v in optimization_args.items()])
    sched_str = "\n    ".join([f"- {k:35s} = {v}" for k, v in scheduling_args.items()])
    prec_str = "\n    ".join([f"- {k:35s} = {v}" for k, v in precision_args.items()])
    log_str = "\n    ".join([f"- {k:35s} = {v}" for k, v in logging_args.items()])
    ddp_str = "\n    ".join([f"- {k:35s} = {v}" for k, v in ddp_args.items()])

    summary_box = f"""
    {'='*70}
    PRE-TRAINING SUMMARY - {model_type}
    {'='*70}
    
    MODEL STATISTICS:
    - Total Parameters:     {total_params/1e6:.2f}M
    - Trainable Parameters: {trainable_params/1e6:.2f}M
    - Est. Weight Size:     {param_size_mb:.2f} MB ({'BF16' if args.bf16 else 'FP32'})
    
    MODEL CONFIGURATION ({model_type}):
    {config_str}

    COMPUTED BATCH METRICS:
    - Devices (GPUs):       {world_size}
    - EFFECTIVE BATCH SIZE: {effective_batch_size} samples
    - TOKENS PER STEP:      {tokens_per_batch/1e6:.2f}M tokens
    
    OPTIMIZATION:
    {opt_str}
    
    SCHEDULING:
    {sched_str}
    
    PRECISION & SPEED:
    {prec_str}
    
    LOGGING & CHECKPOINTING:
    {log_str}
    
    DISTRIBUTED (DDP):
    {ddp_str}
    
    OUTPUT:
    - Output Dir:           {args.output_dir}
    {'='*70}
    """
    print(summary_box)
