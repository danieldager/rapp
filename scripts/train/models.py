from torch import nn

from transformers import (
    PretrainedConfig, 
    PreTrainedModel,
)

from scripts.train.datasets import (
    VOCAB_SIZE,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
)

class LSTMConfig(PretrainedConfig):
    model_type = "lstm"

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        embedding_dim=768,
        hidden_size=512,
        num_layers=1,
        dropout=0.1,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=EOS_TOKEN_ID,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout


class LSTM(PreTrainedModel):
    config_class = LSTMConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )
        self.output = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        embeddings = self.embedding(input_ids)

        # Standard LSTM forward pass (no packing since we use fixed blocks)
        lstm_output, _ = self.lstm(embeddings)
        logits = self.output(lstm_output)

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        return {"loss": loss, "logits": logits} if loss is not None else logits