import re
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Tuple, Union

from torch import Tensor

from ...layers.activations import Activation
from ...layers.attention import RelativePositionAttentionType
from ..hf_hub import _process_hf_keys
from .config import DeBERTaConfig

HF_KEY_TO_CURATED_KEY = MappingProxyType(
    {
        "embeddings.word_embeddings.weight": "embeddings.piece_embeddings.weight",
        "embeddings.token_type_embeddings.weight": "embeddings.type_embeddings.weight",
        "embeddings.position_embeddings.weight": "embeddings.position_embeddings.weight",
        "embeddings.LayerNorm.weight": "embeddings.embed_output_layer_norm.weight",
        "embeddings.LayerNorm.bias": "embeddings.embed_output_layer_norm.bias",
        # The relative position embedding parameter is stored in the first attention
        # layer and shared with the other layers. Same with its layer norm.
        "encoder.rel_embeddings.weight": "layers.0.mha.attention_scorer.rel_position_embeddings.weight",
        "encoder.LayerNorm.weight": "layers.0.mha.attention_scorer.rel_position_embeddings_layer_norm.weight",
        "encoder.LayerNorm.bias": "layers.0.mha.attention_scorer.rel_position_embeddings_layer_norm.bias",
        # The convolution layer is stored directy in the encoder.
        "encoder.conv.conv.weight": "conv_layer.kernel.weight",
        "encoder.conv.conv.bias": "conv_layer.kernel.bias",
        "encoder.conv.LayerNorm.weight": "conv_output_layer_norm.weight",
        "encoder.conv.LayerNorm.bias": "conv_output_layer_norm.bias",
    }
)


HF_CONFIG_KEY_MAPPING: Dict[str, Union[str, Tuple[str, Callable]]] = {
    "attention_probs_dropout_prob": "attention_probs_dropout_prob",
    "hidden_act": ("activation", Activation),
    "hidden_dropout_prob": "hidden_dropout_prob",
    "hidden_size": "hidden_width",
    "intermediate_size": "intermediate_width",
    "layer_norm_eps": "layer_norm_eps",
    "max_position_embeddings": "n_positions",
    "num_attention_heads": "n_attention_heads",
    "num_hidden_layers": "n_hidden_layers",
    "type_vocab_size": "n_types",
    "vocab_size": "n_pieces",
    "position_buckets": "n_rel_position_buckets",
    "pos_att_type": "rel_position_attention_type",
    "norm_rel_ebd": "use_rel_position_layer_norm",
    "conv_kernel_size": "conv_kernel_size",
    "conv_act": ("conv_activation", Activation),
    "share_att_key": "share_content_pos_projection",
    # The keys below are not used by our config per-se
    # but need to be validated nevertheless.
    "relative_attention": "relative_attention",
    "max_relative_positions": "max_relative_positions",
    "position_biased_input": "position_biased_input",
}


def convert_hf_config(hf_config: Any) -> DeBERTaConfig:
    if hf_config["model_type"] == "deberta":
        raise ValueError("Only DeBERTA v2/v3 models are supported")

    kwargs = _process_hf_keys("DeBERTa", hf_config, HF_CONFIG_KEY_MAPPING)

    # These keys are not always present in the config, so
    # they need to be handled separately.
    optional_keys = {
        "conv_groups": "n_conv_groups",
    }
    kwargs.update(
        {ct: hf_config[hf] for hf, ct in optional_keys.items() if hf in hf_config}
    )

    if kwargs["conv_kernel_size"] == 0:
        kwargs["conv_kernel_size"] = None
        kwargs["conv_activation"] = None
        kwargs["n_conv_groups"] = None

    if not kwargs["position_biased_input"]:
        kwargs["n_positions"] = None
    if not kwargs["n_types"]:
        kwargs["n_types"] = None

    kwargs["use_rel_position_layer_norm"] = (
        kwargs["use_rel_position_layer_norm"] != "none"
    )
    rel_pos_attn_type = kwargs["rel_position_attention_type"]
    if isinstance(rel_pos_attn_type, list):
        rel_pos_attn_type = "|".join(rel_pos_attn_type)
    kwargs["rel_position_attention_type"] = RelativePositionAttentionType.from_string(
        rel_pos_attn_type
    )

    if (
        not kwargs["relative_attention"]
        or kwargs["rel_position_attention_type"] == None
    ):
        raise ValueError(
            "DeBERTa models without relative/disentangled "
            "attention are not supported"
        )
    elif kwargs["max_relative_positions"] != -1:
        raise ValueError(
            "DeBERTa models without an inferred number of "
            "maximum relative positions are not supported"
        )

    # Unused kwargs.
    kwargs.pop("relative_attention")
    kwargs.pop("max_relative_positions")
    kwargs.pop("position_biased_input")

    return DeBERTaConfig(
        embedding_width=hf_config["hidden_size"],
        model_max_length=hf_config["max_position_embeddings"],
        **kwargs,
    )


def convert_hf_state_dict(params: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    out = {}

    for name, parameter in params.items():
        if "encoder.layer." not in name:
            continue

        # Remove the prefix and rename the internal 'layers' variable.
        name = re.sub(r"^encoder\.", "", name)
        name = re.sub(r"^layer", "layers", name)

        # The HF model has one more level of indirection for the output layers in their
        # attention heads and the feed-forward network layers.
        name = re.sub(r"\.attention\.self\.(query|key|value)_proj", r".mha.\1", name)
        name = re.sub(
            r"\.attention\.self\.pos_(query|key)_proj",
            r".mha.attention_scorer.rel_position_\1",
            name,
        )
        name = re.sub(r"\.attention\.(output)\.dense", r".mha.\1", name)
        name = re.sub(
            r"\.attention\.output\.LayerNorm", r".attn_residual_layer_norm", name
        )
        name = re.sub(r"\.(intermediate)\.dense", r".ffn.\1", name)
        name = re.sub(
            r"(\.\d+)\.output\.LayerNorm", r"\1.ffn_residual_layer_norm", name
        )
        name = re.sub(r"(\.\d+)\.(output)\.dense", r"\1.ffn.\2", name)

        out[name] = parameter

    for hf_name, curated_name in HF_KEY_TO_CURATED_KEY.items():
        if hf_name in params:
            out[curated_name] = params[hf_name]

    return out
