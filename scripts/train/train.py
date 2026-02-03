"""Language Model Training Script

Supports both LSTM and GPT-2 architectures with architecture-specific optimal configurations.

LSTM OPTIMAL CONFIG (discovered Feb 2026 via systematic debugging):
- Model: 256/256/2 (embedding_dim/hidden_size/num_layers)
- Optimizer: Adam lr=1e-2, beta2=0.99
- Gradient clipping: 5.0
- Dropout: 0.0
- Weight decay: 0.0
- Batch size: 32 × 1 accumulation × 3 GPUs = 96 effective
- Scheduler: Constant (no warmup)
- Results: Consistent convergence to ~1.9 loss at 40 steps, ~1.7 at 100 steps

GPT-2 CONFIG (from paper):
- Model: 768/12/12 (n_embd/n_layer/n_head)
- Optimizer: AdamW lr=1e-4, beta2=0.98
- Batch size: 32 × 4 accumulation × 3 GPUs = 384 effective
- Scheduler: inverse_sqrt with 1000 warmup steps
- Max steps: 100,000
"""

import os
import argparse
from time import time

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import PrinterCallback

from scripts.train.datasets import (
    TokenDataset,
    EvalDataset,
    collate_fn,
    MAX_TOKENS,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    VOCAB_SIZE,
)

from scripts.train.models import LSTM, LSTMConfig
from scripts.train.utils import print_training_summary, get_model_timestamp


# ============================================================================
# MODEL & TRAINING CONFIGURATIONS
# ============================================================================

LSTM_MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "embedding_dim": 256,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.0,
    "bos_token_id": BOS_TOKEN_ID,
    "eos_token_id": EOS_TOKEN_ID,
}

LSTM_TRAINING_CONFIG = {
    "overwrite_output_dir": True,
    "disable_tqdm": True,
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "optim": "adamw_torch",
    "learning_rate": 1e-2,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "max_grad_norm": 5.0,
    "weight_decay": 0.0,
    "lr_scheduler_type": "constant",
    "warmup_steps": 0,
    "max_steps": 10000,
    "bf16": True,
    "dataloader_num_workers": 4,
    "logging_steps": 10,
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 3,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "load_best_model_at_end": True,
    "ddp_find_unused_parameters": False,
    "remove_unused_columns": False,
    "label_names": ["labels"],
    "ignore_data_skip": True,
}

GPT2_MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "n_positions": 1024,
    "n_ctx": 1024,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12,
    "n_inner": 3072,
    "activation_function": "gelu_new",
    "loss_type": "ForCausalLMLoss",
    "resid_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "attn_pdrop": 0.0,
    "bos_token_id": BOS_TOKEN_ID,
    "eos_token_id": EOS_TOKEN_ID,
}

GPT2_TRAINING_CONFIG = {
    "overwrite_output_dir": True,
    "disable_tqdm": False,
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optim": "adamw_torch",
    "learning_rate": 1e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "max_grad_norm": 0.0,
    "weight_decay": 0.01,
    "lr_scheduler_type": "inverse_sqrt",
    "warmup_steps": 1000,
    "max_steps": 100000,
    "bf16": True,
    "dataloader_num_workers": 4,
    "logging_steps": 100,
    "save_strategy": "steps",
    "save_steps": 1000,
    "save_total_limit": 20,
    "eval_strategy": "steps",
    "eval_steps": 1000,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "load_best_model_at_end": True,
    "ddp_find_unused_parameters": False,
    "remove_unused_columns": False,
    "label_names": ["labels"],
    "ignore_data_skip": True,
}


# --- CUSTOM CALLBACK FOR LOGGING ---
class CustomCallback(TrainerCallback):
    """Prints a clear message at the start of training to confirm things are working."""

    def __init__(self, use_lstm=False):
        self.start_time = None
        self.use_lstm = use_lstm

    def on_step_begin(self, args, state, control, **kwargs):
        if self.start_time is None:
            self.start_time = time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if os.environ.get("RANK", "0") == "0":
            if logs is not None and ("loss" in logs or "eval_loss" in logs):
                steps_done = state.global_step

                # Training step logging
                if "loss" in logs:
                    elapsed = time() - self.start_time if self.start_time else 0

                    # Heuristic: HF Trainer sometimes logs the sum of losses over gradients accumulation steps
                    # We normalize it here for readability if it looks like a sum (i.e. > 20 for a language model is clearly wrong)
                    current_loss = logs["loss"]
                    if self.use_lstm and args.gradient_accumulation_steps > 1:
                        current_loss /= args.gradient_accumulation_steps

                    # Calculate tokens per second
                    # (Effective Batch Size * Context Window) / (Seconds per step)
                    effective_batch_size = (
                        args.per_device_train_batch_size
                        * args.gradient_accumulation_steps
                        * args.world_size
                    )
                    tokens_per_step = effective_batch_size * MAX_TOKENS  # MAX_TOKENS

                    # This is an approximation of steps per second
                    steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
                    tokens_per_sec = steps_per_sec * tokens_per_step

                    print(
                        f"Step: {steps_done:5d} | Loss: {current_loss:.4f} | "
                        f"TPS: {tokens_per_sec/1000:.1f}k tokens/s | "
                        f"LR: {logs['learning_rate']:.2e}"
                    )

                # Evaluation step logging
                elif "eval_loss" in logs:
                    print(
                        f"EVAL Step: {steps_done:5d} | Eval Loss: {logs['eval_loss']:.4f}"
                    )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train language model")
    parser.add_argument(
        "arch",
        type=str,
        choices=["lstm", "gpt2"],
        help="Model architecture to train (lstm or gpt2)",
    )
    args = parser.parse_args()

    train_dir = "/scratch2/ddager/rapp/tokens/chunks30_spidr_base/"
    eval_dir = "/scratch2/ddager/rapp/tokens/chunks-eval_spidr_base/"

    train_dataset = TokenDataset(train_dir)
    eval_dataset = EvalDataset(eval_dir)

    if os.environ.get("RANK", "0") == "0":
        print(f"Architecture: {args.arch.upper()}")
        print(f"Eval samples: {len(eval_dataset)}")

    # Build model and config based on architecture
    timestamp = get_model_timestamp()
    
    if args.arch == "lstm":
        config = LSTMConfig(**LSTM_MODEL_CONFIG)
        model = LSTM(config)
        run_name = f"lstm_h{config.hidden_size}_l{config.num_layers}_d{config.dropout}_{timestamp}"
        training_config = LSTM_TRAINING_CONFIG
        early_stopping_patience = 5
    elif args.arch == "gpt2":
        config = GPT2Config(**GPT2_MODEL_CONFIG)
        model = GPT2LMHeadModel(config)
        run_name = f"gpt2_e{config.n_embd}_l{config.n_layer}_h{config.n_head}_{timestamp}"
        training_config = GPT2_TRAINING_CONFIG
        early_stopping_patience = 3
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=f"./weights/{run_name}",
        **training_config,
    )

    # Create custom optimizer for LSTM
    # Adam (not AdamW) is critical for LSTM convergence - see debug results
    if args.arch == "lstm":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
        )
        optimizers = (optimizer, None)
    elif args.arch == "gpt2":
        optimizers = (None, None)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        optimizers=optimizers,
        callbacks=[
            CustomCallback(use_lstm=(args.arch == "lstm")),
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
        ],
    )
    trainer.pop_callback(PrinterCallback)

    if os.environ.get("RANK", "0") == "0":
        print_training_summary(config, training_args, model, MAX_TOKENS)

    trainer.train()
