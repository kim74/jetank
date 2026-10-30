"""
Draws text lines into a 1-bit pixel array with Pillow's built-in bitmap font.

Layer: drivers
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

#: Pillow's built-in font is 6 px wide per character.
CHAR_WIDTH_PX: Final = 6
#: Its glyphs fit in 8 rows (capitals to descenders) once the blank rows above are skipped.
LINE_PITCH_PX: Final = 8
_FONT_TOP_PX: Final = 2


def text_grid(width_px: int, height_px: int) -> tuple[int, int]:
    """``(columns, rows)`` of characters that fit in ``width_px`` x ``height_px``."""
    return width_px // CHAR_WIDTH_PX, height_px // LINE_PITCH_PX


def render_lines(lines: Sequence[str], width_px: int, height_px: int) -> NDArray[np.bool_]:
    """Draw ``lines`` top to bottom, left-aligned; text beyond the edges is clipped.

    Returns:
        Array of shape ``(height_px, width_px)``; ``True`` is a lit pixel.
    """
    image = Image.new("1", (width_px, height_px), 0)
    draw = ImageDraw.Draw(image)
    # Pillow >= 10.1 returns a scalable font from load_default(); keep the fixed bitmap one.
    font = getattr(ImageFont, "load_default_imagefont", ImageFont.load_default)()
    for row, line in enumerate(lines):
        draw.text((0, row * LINE_PITCH_PX - _FONT_TOP_PX), line, font=font, fill=1)
    return np.array(image, dtype=np.bool_)
