from functools import partial
from typing import Any, Mapping, Optional, Set, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm

from ...layers.attention import (
    AttentionHeads,
    AttentionMask,
    DisentangledAttention,
    QkvMode,
    SelfAttention,
)
from ...layers.feedforward import PointwiseFeedForward
from ...layers.transformer import (
    EmbeddingDropouts,
    EmbeddingLayerNorms,
    EncoderLayer,
    TransformerDropouts,
    TransformerEmbeddings,
    TransformerLayerNorms,
)
from ...sharing import Shareable, SharedDataDescriptor, SharedDataType
from ..hf_hub import FromHFHub
from ..output import ModelOutput
from ..transformer import TransformerEncoder
from ._hf import convert_hf_config, convert_hf_state_dict
from .config import DeBERTaConfig
from .conv import DeBERTaConvolutionLayer

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="DeBERTaEncoder")


class DeBERTaEncoder(TransformerEncoder, FromHFHub, Shareable):
    """
    DeBERTa v2/v3 (`He et al., 2020`_, `He et al., 2021`_) encoder.

    .. _He et al., 2020 : https://arxiv.org/abs/2006.03654
    .. _He et al., 2021 : https://arxiv.org/abs/2111.09543
    """

    def __init__(self, config: DeBERTaConfig, *, device: Optional[torch.device] = None):
        """
        Construct a DeBERTa encoder.

        :param config:
            Encoder configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The encoder.
        """
        super().__init__()
        Shareable.__init__(self)

        # We need to play around with the parameters a bit
        # to emulate the non-standard operations in the HF
        # implementation.
        use_embedding_proj = (
            config.embedding.embedding_width != config.layer.feedforward.hidden_width
        )
        embedding_dropouts = (
            EmbeddingDropouts(
                proj_output_dropout=Dropout(config.embedding.dropout_prob)
            )
            if use_embedding_proj
            else EmbeddingDropouts(
                embed_output_dropout=Dropout(config.embedding.dropout_prob)
            )
        )
        embedding_layer_norms = (
            EmbeddingLayerNorms(
                proj_output_layer_norm=LayerNorm(
                    config.embedding.embedding_width, config.embedding.layer_norm_eps
                )
            )
            if use_embedding_proj
            else EmbeddingLayerNorms(
                embed_output_layer_norm=LayerNorm(
                    config.embedding.embedding_width, config.embedding.layer_norm_eps
                )
            )
        )

        self.embeddings = TransformerEmbeddings(
            dropouts=embedding_dropouts,
            embedding_width=config.embedding.embedding_width,
            hidden_width=config.layer.feedforward.hidden_width,
            layer_norms=embedding_layer_norms,
            n_pieces=config.embedding.n_pieces,
            n_positions=config.embedding.n_positions,
            n_types=config.embedding.n_types,
            device=device,
        )

        self.max_seq_len = config.model_max_length
        self.share_attn_content_pos_projection = (
            config.layer.attention.share_content_pos_projection
        )

        layer_norm = partial(
            LayerNorm,
            config.layer.feedforward.hidden_width,
            config.layer.layer_norm_eps,
            device=device,
        )
        self.layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    attention_layer=SelfAttention(
                        attention_heads=AttentionHeads.uniform(
                            config.layer.attention.n_query_heads
                        ),
                        attention_scorer=DisentangledAttention(
                            dropout_prob=config.layer.attention.dropout_prob,
                            hidden_width=config.layer.feedforward.hidden_width,
                            n_rel_position_buckets=config.embedding.n_rel_position_buckets,
                            layer_norm_eps=config.layer.layer_norm_eps,
                            position_attention_type=config.layer.attention.rel_position_attention_type,
                            use_bias=True,
                            device=device,
                        ),
                        hidden_width=config.layer.feedforward.hidden_width,
                        qkv_mode=QkvMode.SEPARATE,
                        rotary_embeds=None,
                        use_bias=config.layer.attention.use_bias,
                        device=device,
                    ),
                    feed_forward_layer=PointwiseFeedForward(
                        activation=config.layer.feedforward.activation.module(),
                        hidden_width=config.layer.feedforward.hidden_width,
                        intermediate_width=config.layer.feedforward.intermediate_width,
                        use_bias=config.layer.feedforward.use_bias,
                        use_gate=config.layer.feedforward.use_gate,
                        device=device,
                    ),
                    dropouts=TransformerDropouts.layer_output_dropouts(
                        config.layer.dropout_prob
                    ),
                    layer_norms=TransformerLayerNorms(
                        attn_residual_layer_norm=layer_norm(),
                        ffn_residual_layer_norm=layer_norm(),
                    ),
                    use_parallel_attention=config.layer.attention.use_parallel_attention,
                )
                for _ in range(config.layer.n_hidden_layers)
            ]
        )

        if config.layer.convolution is None:
            self.conv_layer = None
            self.conv_output_layer_norm = None
        else:
            self.conv_layer = DeBERTaConvolutionLayer(
                activation=config.layer.convolution.activation.module(),
                dropout_prob=config.layer.dropout_prob,
                hidden_width=config.layer.feedforward.hidden_width,
                kernel_size=config.layer.convolution.kernel_size,
                n_groups=config.layer.convolution.n_groups,
                device=device,
            )
            self.conv_output_layer_norm = layer_norm()

    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        positions: Optional[Tensor] = None,
        type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        embeddings = self.embeddings(piece_ids, positions=positions, type_ids=type_ids)
        layer_output = embeddings

        # Apply the convolution layer to the embedding outputs
        # and add it to the output of the first layer.
        if self.conv_layer is None:
            conv_output = None
        else:
            conv_output = self.conv_layer(embeddings)

        layer_outputs = []
        for i, layer in enumerate(self.layers):
            layer_output, _ = layer(layer_output, attention_mask)
            # TODO: Does this work with TorchScript tracing?
            # Alternatively, we can subclass EncoderLayer and
            # drop the conv layer into just the first transformer layer.
            if i == 0 and self.conv_output_layer_norm is not None:
                layer_output = self.conv_output_layer_norm(conv_output + layer_output)

            layer_outputs.append(layer_output)

        return ModelOutput(all_outputs=[embeddings, *layer_outputs])

    def shared_data(self) -> Set[SharedDataDescriptor]:
        out: Set[SharedDataDescriptor] = set()

        for i in range(len(self.layers)):
            # Share the relative position embeddings and its layer norm.
            # They are only stored in the first attention layer.
            if i > 0:
                out.add(
                    SharedDataDescriptor(
                        source="layers.0.mha.attention_scorer.rel_position_embeddings",
                        target=f"layers.{i}.mha.attention_scorer.rel_position_embeddings",
                        type=SharedDataType.MODULE,
                    )
                )
                out.add(
                    SharedDataDescriptor(
                        source="layers.0.mha.attention_scorer.rel_position_embeddings_layer_norm",
                        target=f"layers.{i}.mha.attention_scorer.rel_position_embeddings_layer_norm",
                        type=SharedDataType.MODULE,
                    )
                )

            # Share the query/key projection matrices for the positions
            # in all attention layer.
            if self.share_attn_content_pos_projection:
                out.add(
                    SharedDataDescriptor(
                        source=f"layers.{i}.mha.query",
                        target=f"layers.{i}.mha.attention_scorer.rel_position_query",
                        type=SharedDataType.MODULE,
                    )
                )
                out.add(
                    SharedDataDescriptor(
                        source=f"layers.{i}.mha.key",
                        target=f"layers.{i}.mha.attention_scorer.rel_position_key",
                        type=SharedDataType.MODULE,
                    )
                )

        return out

    @classmethod
    def convert_hf_state_dict(cls, params: Mapping[str, Tensor]):
        return convert_hf_state_dict(params)

    @classmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        config = convert_hf_config(hf_config)
        return cls(config, device=device)
