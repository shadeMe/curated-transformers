from typing import Optional

import torch
from torch import Tensor
from torch.nn import Conv1d, Dropout, Module


class DeBERTaConvolutionLayer(Module):
    """
    DeBERTa v2/v3 (`He et al., 2020`_, `He et al., 2021`_) convolution
    layer. Used to induce n-gram knowledge of the subword encodings.

    .. _He et al., 2020 : https://arxiv.org/abs/2006.03654
    .. _He et al., 2021 : https://arxiv.org/abs/2111.09543
    """

    def __init__(
        self,
        *,
        activation: Module,
        dropout_prob: float,
        hidden_width: int,
        kernel_size: int,
        n_groups: int,
        device: Optional[torch.device] = None,
    ):
        """
        Construct an DeBERTAa convolution layer.

        :param activation:
            Activation used by the convolution layer.
        :param dropout_prob:
            Dropout applied to pre-activations of the
            convolution layer.
        :param hidden_width:
            Hidden width of the transformer.
        :param kernel_size:
            Size of the convolving kernel.
        :param n_groups:
            Number of blocked connections from input channels
            to output channels.
        :param device:
            Device on which the module is to be initialized.
        """
        super().__init__()

        self.kernel = Conv1d(
            in_channels=hidden_width,
            out_channels=hidden_width,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=n_groups,
            device=device,
        )
        self.activation = activation
        self.dropout = Dropout(p=dropout_prob)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Apply the convolution layer to the given piece hidden representations.

        :param hidden_states:
            Hidden representations of piece identifiers. Usually
            the output of the embedding layer.

            *Shape:* ``(batch_size, seq_len, width)``
        :returns:
            Output of the convolution layer.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        # TODO: do we need to use the attention mask here?

        # Swap non-batch axes to conform to the kernel's expected input shape
        # and swap it back in the output.
        out = self.kernel(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.activation(self.dropout(out))
        return out
