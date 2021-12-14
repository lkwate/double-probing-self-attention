import copy
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertLayer
from transformers.modeling_utils import apply_chunking_to_forward
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder
from loguru import logger

class CrossBertLayer(nn.Module):
    def __init__(self, layer: BertLayer):
        super().__init__()
        self.chunk_size_feed_forward = layer.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = layer.attention
        self.intermediate = layer.intermediate
        self.output = layer.output

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        cross_attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )
        attention_output = cross_attention_output[0]
        outputs = cross_attention_output[1:]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        
        outputs = (layer_output,) + outputs
        return outputs
        
        
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

def slice_transformers(model_name: str):
    base_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    cross_model = BertEncoder(base_model.config)
    cross_model.layer = nn.ModuleList([CrossBertLayer(layer) for layer in base_model.encoder.layer])
    return base_model, cross_model
