### TODO: maybe baysian optimization of hyperparameters later ?

import os
import torch
from time import time

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


def print_training_summary(config, args, model):
    """Prints a clear summary of the model and training parameters."""
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
        tokens_per_batch = effective_batch_size * MAX_TOKENS

    if hasattr(config, "n_layer"):
        arch_str = f"- Layers: {config.n_layer} | Heads: {config.n_head} | Embed: {config.n_embd}"
    else:
        arch_str = f"- Layers: {config.num_layers} | Hidden: {config.hidden_size} | Embed: {config.embedding_dim}"

    summary_box = f"""
    {'='*60}
    PRE-TRAINING SUMMARY
    {'='*60}
    MODEL ARCHITECTURE:
    {arch_str}
    - Total Parameters:     {total_params/1e6:.2f}M
    - Trainable Parameters: {trainable_params/1e6:.2f}M
    - Est. Weight Size:     {param_size_mb:.2f} MB ({'BF16' if args.bf16 else 'FP32'})

    TRAINING CONFIGURATION:
    - Devices Found:        {world_size}
    - Batch Size/Device:    {args.per_device_train_batch_size}
    - Grad Accumulation:    {args.gradient_accumulation_steps}
    - EFFECTIVE BATCH SIZE: {effective_batch_size} samples
    - TOKENS PER STEP:      {tokens_per_batch/1e6:.2f}M tokens
    
    STRATEGY:
    - Max Steps:            {args.max_steps}
    - Learning Rate:        {args.learning_rate}
    - Warmup Steps:         {args.warmup_steps}
    - Output Dir:           {args.output_dir}
    {'='*60}
    """
    print(summary_box)


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
    train_dir = "/scratch2/ddager/rapp/tokens/lv_spidr_base/"
    eval_dir = "/scratch2/ddager/rapp/tokens/chunks-eval_spidr_base/"
    USE_LSTM = True

    train_dataset = TokenDataset(train_dir)
    eval_dataset = EvalDataset(eval_dir)

    if os.environ.get("RANK", "0") == "0":
        print(f"Eval samples:  {len(eval_dataset)}")

    if USE_LSTM:
        config = LSTMConfig(
            vocab_size=VOCAB_SIZE,
            embedding_dim=1024,
            hidden_size=1024,
            num_layers=3,
            dropout=0.1,
            bos_token_id=BOS_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
        )
        model = LSTM(config)
        run_name = f"lstm_h{config.hidden_size}_l{config.num_layers}_d{config.dropout}"
    else:
        config = GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_positions=MAX_TOKENS,
            n_ctx=MAX_TOKENS,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=3072,
            activation_function="gelu_new",
            loss_type="ForCausalLMLoss",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            bos_token_id=BOS_TOKEN_ID,
            eos_token_id=EOS_TOKEN_ID,
        )
        model = GPT2LMHeadModel(config)
        run_name = f"gpt2_e{config.n_embd}_l{config.n_layer}_h{config.n_head}"

    training_args = TrainingArguments(
        output_dir=f"./weights/{run_name}",
        overwrite_output_dir=True,
        disable_tqdm=True,
        # Optimization
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        weight_decay=0.01,
        # Scheduling
        # lr_scheduler_type="cosine",
        lr_scheduler_type="inverse_sqrt",
        warmup_steps=500,
        max_steps=10000,
        # Precision & Speed
        bf16=True,
        dataloader_num_workers=4,
        # Logging & Saving
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        # Evaluation
        eval_strategy="steps",
        eval_steps=50,
        metric_for_best_model="loss",
        greater_is_better=False,
        load_best_model_at_end=True,  # TODO: do I need ?
        # DDP
        ddp_find_unused_parameters=False,  # Optimization for standard models
        remove_unused_columns=False,  # Keeps all columns model receives in batch
        label_names=["labels"],  # Ensures Trainer knows which key is the target
        ignore_data_skip=True,  # Faster restart for IterableDatasets
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[
            CustomCallback(use_lstm=USE_LSTM),
            EarlyStoppingCallback(early_stopping_patience=5),
        ],
    )
    trainer.pop_callback(PrinterCallback)

    if os.environ.get("RANK", "0") == "0":
        print_training_summary(config, training_args, model)

    trainer.train()
