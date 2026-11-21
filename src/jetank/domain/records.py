"""
Records saved with a camera image: what taught grasps and recorded scenes have in common.

Layer: domain

A record's ID is derived from the moment it was taken and doubles as its file name stem, so
it is restricted to digits and dashes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import AwareDatetime, BaseModel, ConfigDict, StringConstraints

if TYPE_CHECKING:
    from datetime import datetime

#: Pattern of a record ID: date, time and microseconds, e.g. ``20261121-131542-538584``.
RECORD_ID_PATTERN: Final = r"^[0-9]{8}-[0-9]{6}-[0-9]{6}$"
#: Identifier of a record; also its file name stem, so it is restricted to safe characters.
RecordId = Annotated[str, StringConstraints(pattern=RECORD_ID_PATTERN)]

#: A short name an operator types, e.g. an animal ("lion", "löwe") or a room ("living room"):
#: letters, digits, spaces, "-" and "_", lower case.
LabelName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, to_lower=True, min_length=1, max_length=40, pattern=r"^\w[\w \-]*$"
    ),
]


class ImageRecord(BaseModel):
    """Base of every record stored as JSON next to a camera image.

    Attributes:
        record_id: Unique ID, derived from the time the record was taken.
        recorded_at: Wall-clock time the record was taken, with time zone.
        image_file: Name of the camera image saved with the record, if there was one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: RecordId
    recorded_at: AwareDatetime
    image_file: str | None = None


def is_record_id(text: str) -> bool:
    """Whether ``text`` is a well-formed record ID (and so safe to use as a file name stem)."""
    return re.fullmatch(RECORD_ID_PATTERN, text) is not None


def record_id_for(moment: datetime) -> str:
    """The record ID for a record taken at ``moment`` (microseconds keep IDs apart)."""
    return moment.strftime("%Y%m%d-%H%M%S-%f")
