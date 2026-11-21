"""
TensorRT runtime for a serialized engine (built by ``scripts/export_detector.sh``).

Layer: drivers

GPU buffers and the CUDA stream come from PyTorch (NVIDIA's JetPack build in the runtime
venv): the ``cuda-python`` bindings on PyPI need a newer driver than JetPack 6 has. Engines
are built for one TensorRT version and one GPU, so build them on the Jetson itself.

``tensorrt`` and ``torch`` are imported when an engine is opened, not at import time, so the
rest of the package (and the tests) work on machines without them.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from jetank.domain.perception import NetworkOutputs
from jetank.errors import DriverClosedError, InferenceError

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class TensorRtEngine:
    """:class:`~jetank.interfaces.InferenceEngine` running a TensorRT engine on the GPU.

    The engine must have one ``float32`` input and ``float32`` outputs with fixed shapes
    (export with ``dynamic=False``; FP16 engines keep ``float32`` inputs and outputs).

    Args:
        engine_path: The serialized engine.

    Raises:
        InferenceError: The file is missing, TensorRT, PyTorch or the GPU is not available,
            or the engine's inputs and outputs are not as described.
    """

    def __init__(self, engine_path: Path) -> None:
        if not engine_path.is_file():
            raise InferenceError(f"no TensorRT engine at {engine_path}")
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:
            raise InferenceError(f"TensorRT and PyTorch are needed to run {engine_path}") from exc
        if not torch.cuda.is_available():
            raise InferenceError("PyTorch sees no GPU")
        self._torch = torch
        # TensorRT and torch objects are untyped (no stubs).
        self._trt_logger: Any = trt.Logger(trt.Logger.WARNING)
        runtime: Any = trt.Runtime(self._trt_logger)
        self._engine: Any = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise InferenceError(
                f"TensorRT could not load {engine_path} (built for another TensorRT or GPU?)"
            )
        self._context: Any = self._engine.create_execution_context()
        self._stream: Any = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        self._input_name = ""
        self._input: Any = None
        self._outputs: dict[str, Any] = {}
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            shape = tuple(int(d) for d in self._engine.get_tensor_shape(name))
            if self._engine.get_tensor_dtype(name) != trt.float32 or min(shape) < 1:
                raise InferenceError(f"{engine_path}: tensor {name} {shape} is not fixed float32")
            buffer = torch.empty(shape, dtype=torch.float32, device="cuda")
            self._context.set_tensor_address(name, buffer.data_ptr())
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if self._input is not None:
                    raise InferenceError(f"{engine_path} has more than one input")
                self._input_name, self._input = name, buffer
            else:
                self._outputs[name] = buffer
        if self._input is None or not self._outputs:
            raise InferenceError(f"{engine_path} needs one input and at least one output")
        self._closed = False
        logger.info(
            "loaded %s: input %s %s, outputs %s",
            engine_path,
            self._input_name,
            self.input_shape,
            {name: tuple(t.shape) for name, t in self._outputs.items()},
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self._input.shape)

    def infer(self, batch: NDArray[np.float32]) -> NetworkOutputs:
        if self._closed:
            raise DriverClosedError("TensorRT engine is closed")
        if batch.shape != self.input_shape:
            raise ValueError(f"input must have shape {self.input_shape}, got {batch.shape}")
        torch = self._torch
        try:
            with torch.cuda.stream(self._stream):
                self._input.copy_(torch.from_numpy(np.ascontiguousarray(batch, np.float32)))
                if not self._context.execute_async_v3(self._stream.cuda_stream):
                    raise InferenceError("TensorRT failed to run the engine")
                host = {name: t.to("cpu", non_blocking=False) for name, t in self._outputs.items()}
            self._stream.synchronize()
        except RuntimeError as exc:  # CUDA errors surface as RuntimeError in torch
            raise InferenceError(f"inference failed: {exc}") from exc
        return NetworkOutputs(tensors={name: t.numpy() for name, t in host.items()})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._outputs.clear()
        self._input = None
        self._context = None
        self._engine = None
        self._torch.cuda.empty_cache()
        logger.debug("TensorRT engine closed")

    def __enter__(self) -> TensorRtEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
