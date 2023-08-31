import pytest
from curated_transformers.models.deberta.encoder import DeBERTaEncoder

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import JITMethod, assert_encoder_output_equals_hf

MODELS = [
    # "microsoft/deberta-v3-xsmall",
    "explosion-testing/deberta-v3-test",
    # "explosion-testing/deberta-v3-shared-attn-proj-test",
]


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("model_name", MODELS)
def test_encoder(torch_device, model_name):
    assert_encoder_output_equals_hf(
        DeBERTaEncoder,
        model_name,
        torch_device,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("model_name", MODELS)
def test_encoder_with_torch_compile(torch_device, model_name):
    assert_encoder_output_equals_hf(
        DeBERTaEncoder,
        model_name,
        torch_device,
        jit_method=JITMethod.TorchCompile,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("model_name", MODELS)
def test_encoder_with_torchscript_trace(torch_device, model_name):
    assert_encoder_output_equals_hf(
        DeBERTaEncoder,
        model_name,
        torch_device,
        jit_method=JITMethod.TorchScriptTrace,
    )
