"""
Port for a neural-network runtime with one input and one output tensor.

Layer: interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@runtime_checkable
class InferenceEngine(Protocol):
    @property
    def input_shape(self) -> tuple[int, ...]:
        """Exact shape the input array must have, e.g. ``(1, 3, 640, 640)``."""
        ...

    def infer(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run the network on ``batch`` and return its output tensor."""
        ...

    def close(self) -> None:
        """Release device memory. Must be idempotent."""
        ...
