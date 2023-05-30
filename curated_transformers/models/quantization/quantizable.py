from typing import Set
from abc import ABC, abstractmethod


class Quantizable(ABC):
    """Mixin class for models that are quantizable.

    A module using this mixin provides the necessary configuration
    and parameter information to quantize it on-the-fly during the
    module loading phase.
    """

    @classmethod
    @abstractmethod
    def modules_to_quantize(cls) -> Set[str]:
        """Returns a set of prefixes that specify which modules are to
        be quantized.

        :returns:
            Set of module prefixes. If empty, all submodules can be
            quantized.
        """
        raise NotImplementedError
