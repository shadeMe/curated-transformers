from dataclasses import dataclass
from typing import Optional

from ...layers.activations import Activation
from ...layers.attention import RelativePositionAttentionType
from ..config import (
    TransformerAttentionLayerConfig,
    TransformerEmbeddingLayerConfig,
    TransformerFeedForwardLayerConfig,
    TransformerLayerConfig,
)


@dataclass
class DeBERTaEmbeddingLayerConfig(TransformerEmbeddingLayerConfig):
    """
    DeBERTa v2/v3 (`He et al., 2020`_, `He et al., 2021`_) embedding configuration.

    .. _He et al., 2020 : https://arxiv.org/abs/2006.03654
    .. _He et al., 2021 : https://arxiv.org/abs/2111.09543
    """

    n_rel_position_buckets: int
    use_rel_position_layer_norm: bool


@dataclass
class DeBERTaAttentionLayerConfig(TransformerAttentionLayerConfig):
    """
    DeBERTa v2/v3 (`He et al., 2020`_, `He et al., 2021`_) attention configuration.

    .. _He et al., 2020 : https://arxiv.org/abs/2006.03654
    .. _He et al., 2021 : https://arxiv.org/abs/2111.09543
    """

    rel_position_attention_type: RelativePositionAttentionType
    share_content_pos_projection: bool


@dataclass
class DeBERTaConvolutionLayerConfig:
    """
    DeBERTa v2/v3 (`He et al., 2020`_, `He et al., 2021`_) convolution layer configuration.

    .. _He et al., 2020 : https://arxiv.org/abs/2006.03654
    .. _He et al., 2021 : https://arxiv.org/abs/2111.09543
    """

    activation: Activation
    kernel_size: int
    n_groups: int


@dataclass
class DeBERTaLayerConfig(TransformerLayerConfig):
    """
    DeBERTa v2/v3 (`He et al., 2020`_, `He et al., 2021`_) transformer layer configuration.

    .. _He et al., 2020 : https://arxiv.org/abs/2006.03654
    .. _He et al., 2021 : https://arxiv.org/abs/2111.09543
    """

    attention: DeBERTaAttentionLayerConfig
    convolution: Optional[DeBERTaConvolutionLayerConfig]


@dataclass
class DeBERTaConfig:
    """
    DeBERTa v2/v3 (`He et al., 2020`_, `He et al., 2021`_) model configuration.

    .. _He et al., 2020 : https://arxiv.org/abs/2006.03654
    .. _He et al., 2021 : https://arxiv.org/abs/2111.09543
    """

    embedding: DeBERTaEmbeddingLayerConfig
    layer: DeBERTaLayerConfig
    model_max_length: int

    def __init__(
        self,
        *,
        embedding_width: int = 1536,
        hidden_width: int = 1536,
        intermediate_width: int = 6144,
        n_attention_heads: int = 24,
        n_hidden_layers: int = 24,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        activation: Activation = Activation.GELU,
        n_pieces: int = 128100,
        n_types: Optional[int] = None,
        n_positions: Optional[int] = 512,
        model_max_length: int = 512,
        layer_norm_eps: float = 1e-7,
        n_rel_position_buckets: int = 256,
        use_rel_position_layer_norm: bool = False,
        rel_position_attention_type: RelativePositionAttentionType = RelativePositionAttentionType.CONTENT_TO_POSITION
        | RelativePositionAttentionType.POSITION_TO_CONTENT,
        share_content_pos_projection: bool = True,
        conv_kernel_size: Optional[int] = 3,
        conv_activation: Optional[Activation] = Activation.GELU,
        n_conv_groups: Optional[int] = 1,
    ):
        """
        :param embedding_width:
            Width of the embedding representations.
        :param hidden_width:
            Width of the transformer hidden layers.
        :param intermediate_width:
            Width of the intermediate projection layer in the
            point-wise feed-forward layer.
        :param n_attention_heads:
            Number of self-attention heads.
        :param n_hidden_layers:
            Number of hidden layers.
        :param attention_probs_dropout_prob:
            Dropout probabilty of the self-attention layers.
        :param hidden_dropout_prob:
            Dropout probabilty of the point-wise feed-forward and
            embedding layers.
        :param activation:
            Activation used by the pointwise feed-forward layers.
        :param n_pieces:
            Size of main vocabulary.
        :param n_types:
            Size of token type vocabulary.
        :param n_positions:
            Maximum length of absolute position embeddings. If
            ``None``, absolute position embeddings are not applied.
        :param model_max_length:
            Maximum length of model inputs.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        :param n_rel_position_buckets:
            Number of position buckets for the relative position embeddings.
            This is independent of the absolute position embeddings. The size
            of these embeddings will be ``2 * n_rel_position_buckets``.
        :param use_rel_position_layer_norm:
            Use layer normalization for the relative position
            embeddings.
        :param rel_position_attention_type:
            Bit-wise combination of relative position attentions to apply.
        :parame share_content_pos_projection:
            Share the same projection matrices for the content and position
            hidden states.
        :param conv_kernel_size:
            Size of the convolution kernel. If ``None``, the convolution
            layer will not be applied.
        :param conv_activation:
            Activation used by the convolution layer. If ``None``, the convolution
            layer will not be applied.
        :param n_conv_groups:
            Number of groups in the convolution layer. If ``None``, the convolution
            layer will not be applied.
        """
        convolution = (
            DeBERTaConvolutionLayerConfig(
                kernel_size=conv_kernel_size,
                activation=conv_activation,
                n_groups=n_conv_groups,
            )
            if conv_kernel_size is not None
            and conv_activation is not None
            and n_conv_groups is not None
            else None
        )

        self.embedding = DeBERTaEmbeddingLayerConfig(
            embedding_width=embedding_width,
            n_pieces=n_pieces,
            n_types=n_types,
            n_positions=n_positions,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
            n_rel_position_buckets=n_rel_position_buckets,
            use_rel_position_layer_norm=use_rel_position_layer_norm,
        )
        self.layer = DeBERTaLayerConfig(
            attention=DeBERTaAttentionLayerConfig(
                hidden_width=hidden_width,
                dropout_prob=attention_probs_dropout_prob,
                n_key_value_heads=n_attention_heads,
                n_query_heads=n_attention_heads,
                rotary_embeddings=None,
                use_alibi=False,
                use_bias=True,
                use_parallel_attention=False,
                rel_position_attention_type=rel_position_attention_type,
                share_content_pos_projection=share_content_pos_projection,
            ),
            feedforward=TransformerFeedForwardLayerConfig(
                hidden_width=hidden_width,
                intermediate_width=intermediate_width,
                activation=activation,
                use_bias=True,
                use_gate=False,
            ),
            convolution=convolution,
            n_hidden_layers=n_hidden_layers,
            layer_norm_eps=layer_norm_eps,
            dropout_prob=hidden_dropout_prob,
        )
        self.model_max_length = model_max_length
