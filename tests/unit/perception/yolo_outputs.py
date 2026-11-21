"""Synthetic YOLO outputs with known answers, for the decoder and analyzer tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jetank.domain.perception import NetworkOutputs

CLASSES = ("toy_animal", "floor")
CANDIDATES = 50
PROTO = 160  # mask prototypes at 640 x 640 input


@dataclass(frozen=True)
class Candidate:
    """One candidate box in input pixels: centre, size, class scores, mask coefficients."""

    cx: float
    cy: float
    w: float
    h: float
    scores: tuple[float, ...]
    coefficients: tuple[float, ...] = ()


def outputs(*candidates: Candidate, masks: bool = True, classes: int = 2) -> NetworkOutputs:
    """The raw output of a YOLO model that proposes ``candidates`` and nothing else.

    With ``masks``, prototype channel 0 is +8 in the bottom half of the input and -8 in the
    top half, so a candidate with coefficient 1 on channel 0 has a bottom-half mask.
    """
    rows = 4 + classes + (32 if masks else 0)
    prediction = np.zeros((1, rows, CANDIDATES), dtype=np.float32)
    for i, c in enumerate(candidates):
        prediction[0, :4, i] = (c.cx, c.cy, c.w, c.h)
        prediction[0, 4 : 4 + classes, i] = c.scores
        if masks and c.coefficients:
            prediction[0, 4 + classes : 4 + classes + len(c.coefficients), i] = c.coefficients
    tensors = {"output0": prediction}
    if masks:
        protos = np.zeros((1, 32, PROTO, PROTO), dtype=np.float32)
        protos[0, 0, : PROTO // 2] = -8.0
        protos[0, 0, PROTO // 2 :] = 8.0
        tensors["output1"] = protos
    return NetworkOutputs(tensors=tensors)
