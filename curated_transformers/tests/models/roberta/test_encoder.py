import pytest

from curated_transformers.models.roberta.encoder import RoBERTaEncoder

from ...compat import has_hf_transformers, has_torch_compile
from ...conftest import TORCH_DEVICES
from ..util import (
    JITMethod,
    assert_encoder_output_equals_hf,
    assert_model_hf_serialization_roundtrip,
)


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_encoder(torch_device, with_torch_sdp):
    assert_encoder_output_equals_hf(
        RoBERTaEncoder,
        "explosion-testing/roberta-test",
        torch_device,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.skipif(not has_torch_compile, reason="requires torch.compile")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_encoder_with_torch_compile(torch_device, with_torch_sdp):
    assert_encoder_output_equals_hf(
        RoBERTaEncoder,
        "explosion-testing/roberta-test",
        torch_device,
        jit_method=JITMethod.TorchCompile,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.slow
@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
@pytest.mark.parametrize("with_torch_sdp", [False, True])
def test_encoder_with_torchscript_trace(torch_device, with_torch_sdp):
    assert_encoder_output_equals_hf(
        RoBERTaEncoder,
        "explosion-testing/roberta-test",
        torch_device,
        jit_method=JITMethod.TorchScriptTrace,
        with_torch_sdp=with_torch_sdp,
    )


@pytest.mark.skipif(not has_hf_transformers, reason="requires huggingface transformers")
@pytest.mark.parametrize("torch_device", TORCH_DEVICES)
def test_encoder_hf_serializtion_roundtrip(torch_device):
    assert_model_hf_serialization_roundtrip(
        RoBERTaEncoder, "explosion-testing/roberta-test", torch_device
    )
