import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Tuple
from dataclasses import dataclass

from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer, BertForMaskedLM, add_start_docstrings
from transformers.models.bert.modeling_bert import BertLayer
from transformers.modeling_outputs import MaskedLMOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class MaskedLMOutputWithLogs(MaskedLMOutput):
    logs: Optional[Dict[str, any]] = None


class BertForTextCompression(BertForMaskedLM):
    def __init__(
            self,
            config,
            n_cls_tokens: int = 1,
            use_decoder: bool = True,
            n_decoder_layers: int = 2,
            enable_decoder_mlm: bool = True,
            decoder_mlm_coef: float = 1.0,
    ):
        super().__init__(config)
        if use_decoder:
            self.decoder = nn.ModuleList(
                [BertLayer(config) for _ in range(n_decoder_layers)]
            )
            self.decoder.apply(self._init_weights)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.n_cls_tokens = n_cls_tokens
        self.use_decoder = use_decoder
        self.n_decoder_layers = n_decoder_layers
        self.enable_decoder_mlm = enable_decoder_mlm
        self.decoder_mlm_coef = decoder_mlm_coef

    def forward(self, **model_input):
        # encoder forward
        encoder_output: MaskedLMOutput = super().forward(
            input_ids=model_input['input_ids'],
            attention_mask=model_input['attention_mask'],
            labels=model_input['labels'],
            output_hidden_states=True,
            return_dict=True
        )
        cls_hiddens = encoder_output.hidden_states[-1][:, 0: self.n_cls_tokens]

        # add last layer mlm loss
        logs = dict()
        loss = encoder_output.loss
        logs["encoder_mlm_loss"] = encoder_output.loss.item()

        # decoder forward
        if self.use_decoder and self.enable_decoder_mlm:
            # Get the embedding of decoder inputs
            decoder_input_embeddings = self.bert.embeddings.word_embeddings(model_input['decoder_input_ids'])
            decoder_input_embeddings = torch.cat([cls_hiddens, decoder_input_embeddings[:, self.n_cls_tokens:]], dim=1)
            decoder_input_embeddings = self.bert.embeddings(inputs_embeds=decoder_input_embeddings)
            decoder_attention_mask = self.get_extended_attention_mask(
                model_input['decoder_attention_mask'],
                model_input['decoder_attention_mask'].shape,
                model_input['decoder_attention_mask'].device
            )
            hiddens = decoder_input_embeddings
            for layer in self.decoder:
                layer_out = layer(
                    hiddens,
                    decoder_attention_mask,
                )
                hiddens = layer_out[0]
            # add head-layer mlm loss
            head_mlm_loss = self.mlm_loss(hiddens, model_input['decoder_labels']) * self.decoder_mlm_coef
            logs["head_mlm_loss"] = head_mlm_loss.item()
            loss += head_mlm_loss

        return MaskedLMOutputWithLogs(
            loss=loss,
            logits=encoder_output.logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
            logs=logs,
        )

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        return masked_lm_loss



