from typing import Any, Mapping, Optional, Set, Type, TypeVar

import torch
from torch import Tensor
from torch.nn import Linear

from ...quantization.quantizable import Quantizable
from ..hf_hub import FromHFHub
from ..hf_hub.conversion import state_dict_from_hf, state_dict_to_hf
from ..transformer import TransformerCausalLM
from ._hf import CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS, convert_hf_config
from .config import FalconConfig
from .decoder import FalconDecoder

# Only provided as typing.Self in Python 3.11+.
Self = TypeVar("Self", bound="FalconCausalLM")


class FalconCausalLM(TransformerCausalLM[FalconConfig], FromHFHub, Quantizable):
    """
    Falcon (`Penedo et al., 2019`_) causal language model.

    .. _Penedo et al., 2019: https://arxiv.org/abs/2306.01116
    """

    def __init__(
        self, config: FalconConfig, *, device: Optional[torch.device] = None
    ) -> None:
        """
        Construct a Falcon causal LM.

        :param config:
            Causal LM configuration.
        :param device:
            Device to which the module is to be moved.
        :returns:
            The causal LM.
        """
        super().__init__(config)

        self.decoder = FalconDecoder(config, device=device)
        self.output_embeddings = Linear(
            in_features=config.layer.feedforward.hidden_width,
            out_features=config.embedding.n_pieces,
            bias=False,
            device=device,
        )

    @classmethod
    def state_dict_from_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_from_hf(params, CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def state_dict_to_hf(
        cls: Type[Self], params: Mapping[str, Tensor]
    ) -> Mapping[str, Tensor]:
        return state_dict_to_hf(params, CAUSAL_LM_HF_PARAM_KEY_TRANSFORMS)

    @classmethod
    def from_hf_config(
        cls: Type[Self],
        *,
        hf_config: Any,
        device: Optional[torch.device] = None,
    ) -> Self:
        config = convert_hf_config(hf_config)
        return cls(config, device=device)

    @classmethod
    def modules_to_not_quantize(cls) -> Set[str]:
        # Ignore the output embedding matrix.
        return {"output_embeddings"}
