"""
Building blocks shared by every configuration section.

Layer: config
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import ConfigDict

Backend = Literal["hardware", "fake"]

#: Settings for every section model: immutable, no unknown keys, defaults validated too.
SECTION: Final = ConfigDict(frozen=True, extra="forbid", validate_default=True)
