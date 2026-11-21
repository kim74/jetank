"""
Port for a neural-network runtime with one input tensor and one or more output tensors.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from jetank.domain.perception import NetworkOutputs


@runtime_checkable
class InferenceEngine(Protocol):
    @property
    def input_shape(self) -> tuple[int, ...]:
        """Exact shape the input array must have, e.g. ``(1, 3, 640, 640)``."""
        ...

    def infer(self, batch: NDArray[np.float32]) -> NetworkOutputs:
        """Run the network on ``batch`` and return all of its output tensors.

        Raises:
            ValueError: ``batch`` does not have :attr:`input_shape`.
            InferenceError: The network could not be run.
        """
        ...

    def close(self) -> None:
        """Release device memory. Must be idempotent."""
        ...
