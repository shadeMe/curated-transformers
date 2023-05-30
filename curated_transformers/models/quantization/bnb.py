from typing import Any, Callable, Dict, Mapping, Optional, Set
import torch
from torch import Tensor
from torch.nn import Parameter, Module


from ..._compat import has_bitsandbytes, bitsandbytes as bnb
from ..util.pytorch import apply_to_module, ModuleIterator
from ..util.serde import default_tensor_to_parameter_converter


# TODO use Pydantic for validation
SUPPORTED_CONFIG_OPTIONS = {
    "mode": ["training", "inference"],
    "width": ["int8"],
    "outlier_threshold": None,
}

DEFAULT_CONFIG = {
    "mode": "inference",
    "width": "int8",
    "outlier_threshold": 0.6,
}


def quantize(
    module: Module, quantizable_module_prefixes: Set[str], config: Dict[str, Any]
) -> Optional[Callable[[Module, str, Tensor], Parameter]]:
    if not has_bitsandbytes:
        raise ValueError(
            "The `bitsandbytes` Python library is required for quantization support"
        )
    _validate_config(config)
    is_inference = config["mode"] == "inference"
    width = config["width"]
    outlier_threshold = config["outlier_threshold"]

    if width == "int8":
        _replace_modules_for_8bit(
            module, quantizable_module_prefixes, is_inference, outlier_threshold
        )
        return _convert_tensor_to_int8_parameter
    else:
        # Unreachable.
        raise NotImplementedError


def _replace_modules_for_8bit(
    module: Module,
    quantizable_module_prefixes: Set[str],
    for_inference: bool,
    outlier_threshold: float,
):
    def apply(itr: ModuleIterator):
        if itr.prefix not in quantizable_module_prefixes:
            return

        assert itr.parent is not None
        if not isinstance(itr.module, torch.nn.Linear):
            raise ValueError(f"Cannot quantize module of type `{type(itr.module)}`")

        quantized_module = bnb.nn.Linear8bitLt(
            input_features=itr.module.in_features,
            output_features=itr.module.out_features,
            bias=itr.module.bias is None,
            has_fp16_weights=not for_inference,
            threshold=outlier_threshold,
        )
        itr.parent._modules[itr.name] = quantized_module

    apply_to_module(module, apply)


def _convert_tensor_to_int8_parameter(
    module: Module, parameter_name: str, tensor: Tensor
) -> Parameter:
    if not isinstance(module, bnb.nn.Linear8bitLt):
        return default_tensor_to_parameter_converter(module, parameter_name, tensor)

    # TODO do we need to do anything else depending on `old_param.has_fp16_weights`?
    old_param = module._parameters[parameter_name]
    assert old_param is not None and isinstance(old_param, bnb.nn.Int8Params)
    return bnb.nn.Int8Params(tensor, requires_grad=old_param.requires_grad)


def _validate_config(config: Dict[str, Any]):
    if len(config) != len(SUPPORTED_CONFIG_OPTIONS):
        raise ValueError(
            f"bitsandbytes quantization must contain {len(SUPPORTED_CONFIG_OPTIONS)} keys, but has {len(config)} instead"
        )

    for k, v in config.items():
        if k not in SUPPORTED_CONFIG_OPTIONS.keys():
            raise ValueError(f"Unknown bitsandbytes quantization config key `{k}`")
        elif v is not None:
            supported_values = SUPPORTED_CONFIG_OPTIONS.get(k)
            if v not in supported_values:  # type: ignore[operator]
                raise ValueError(
                    f"Unknown bitsandbytes quantization config value `{v}` for key `{k}`, expected one of `{supported_values}`"
                )
