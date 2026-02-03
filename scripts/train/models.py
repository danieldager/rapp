from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from scripts.train.datasets import VOCAB_SIZE, BOS_TOKEN_ID, EOS_TOKEN_ID


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
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with LSTM-specific best practices."""
        if isinstance(module, (nn.Embedding, nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    # Orthogonal init for recurrent weights helps gradient flow
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.zero_()
                    # Forget gate bias = 1.0 (standard LSTM practice)
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        embeddings = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embeddings)
        logits = self.output(lstm_output)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        return {"loss": loss, "logits": logits} if loss is not None else logits
