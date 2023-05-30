from typing import Any, Callable, Dict, Mapping, Optional
from torch import Tensor
from torch.nn import Module

from .quantizable import Quantizable
from .bnb import quantize as bnb_quantize
from ..util.serde import TensorToParameterConverterT


def prepare_module_for_quantization(
    module: Module, config: Dict[str, Any]
) -> Optional[TensorToParameterConverterT]:
    """Prepares a module for quantiazation and returns an optional callback
    to generate quantized parameter tensors.

    :param module:
        Top-level module to quantize. Should implement `Quantizable`.
    :param config:
        Configuration for the quantizer.
    :returns:
        A callable that converts a non-quantized tensor to a quantized
        parameter.
    """
    if not isinstance(module, Quantizable):
        raise ValueError(f"Module of type `{type(module)}` is not quantizable")
    qmodel: Quantizable = module
    quantizable_module_prefixes = qmodel.modules_to_quantize()

    return bnb_quantize(module, quantizable_module_prefixes, config)
