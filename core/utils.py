import copy
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertEncoder
from loguru import logger


def slice_transformers(model_name: str, pivot: int):
    base_model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, add_cross_attention=True, is_decoder=True
    )
    cross_model = BertEncoder(base_model.config)
    cross_model.load_state_dict(copy.deepcopy(base_model.encoder.state_dict()))
    cross_model.layer = cross_model.layer[pivot:]
    base_model.encoder.layer = base_model.encoder.layer[:pivot]
    return base_model, cross_model
