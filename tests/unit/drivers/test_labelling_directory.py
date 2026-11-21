from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from jetank.domain.dataset import DrawnImage, DrawnShape
from jetank.drivers.labelling_directory import LabellingDirectory
from jetank.errors import StorageError

if TYPE_CHECKING:
    from pathlib import Path

ID = "20261121-172308-331070"


def x_anylabeling_file(**overrides: object) -> str:
    """A drawing as X-AnyLabeling saves it (LabelMe format, with fields we ignore)."""
    data = {
        "version": "2.4.4",
        "flags": {},
        "shapes": [
            {
                "label": "floor",
                "score": None,
                "points": [[0.0, 480.0], [0.0, 210.5], [639.0, 200.0]],
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
            },
            {
                "label": "toy_animal ",
                "points": [[40, 250], [100, 300]],
                "shape_type": "rectangle",
            },
        ],
        "imagePath": f"{ID}.jpg",
        "imageData": None,
        "imageHeight": 480,
        "imageWidth": 640,
    }
    data.update(overrides)
    return json.dumps(data)


def test_writes_classes_in_index_order(tmp_path: Path) -> None:
    LabellingDirectory(tmp_path / "labelling").write_classes(["toy_animal", "floor"])
    assert (tmp_path / "labelling" / "classes.txt").read_text() == "toy_animal\nfloor\n"


def test_adds_images_but_never_replaces_one(tmp_path: Path) -> None:
    labelling = LabellingDirectory(tmp_path)
    assert labelling.add_image(ID, b"first")
    assert not labelling.add_image(ID, b"second")
    assert (tmp_path / f"{ID}.jpg").read_bytes() == b"first"
    with pytest.raises(StorageError, match="not a scene ID"):
        labelling.add_image("../x", b"jpeg")


def test_reads_drawings_saved_by_the_tool(tmp_path: Path) -> None:
    (tmp_path / f"{ID}.json").write_text(x_anylabeling_file())
    (tmp_path / "20261121-172310-000000.jpg").write_bytes(b"not labelled yet")
    (drawing,) = LabellingDirectory(tmp_path).load_drawings()
    assert drawing.record_id == ID
    assert (drawing.width_px, drawing.height_px) == (640, 480)
    assert drawing.shapes == (
        DrawnShape(label="floor", kind="polygon", points=((0, 480), (0, 210.5), (639, 200))),
        DrawnShape(label="toy_animal", kind="rectangle", points=((40, 250), (100, 300))),
    )


def test_an_image_saved_with_nothing_drawn_has_no_shapes(tmp_path: Path) -> None:
    (tmp_path / f"{ID}.json").write_text(x_anylabeling_file(shapes=[]))
    assert LabellingDirectory(tmp_path).load_drawings()[0].shapes == ()


def test_a_missing_folder_has_no_drawings(tmp_path: Path) -> None:
    assert LabellingDirectory(tmp_path / "none").load_drawings() == []


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        ("notes.json", "{}", "not named after a scene ID"),
        (f"{ID}.json", "{broken", f"could not read the drawing {ID}"),
        (f"{ID}.json", json.dumps({"shapes": []}), "could not read the drawing"),
    ],
)
def test_bad_drawings_name_the_file(tmp_path: Path, name: str, content: str, expected: str) -> None:
    (tmp_path / name).write_text(content)
    with pytest.raises(StorageError, match=expected):
        LabellingDirectory(tmp_path).load_drawings()


def test_unwritable_folder_raises_storage_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    labelling = LabellingDirectory(blocker / "labelling")
    with pytest.raises(StorageError, match=r"classes\.txt"):
        labelling.write_classes(["floor"])
    with pytest.raises(StorageError, match="could not add"):
        labelling.add_image(ID, b"jpeg")


DRAFT = DrawnImage(
    record_id=ID,
    width_px=640,
    height_px=480,
    shapes=(
        DrawnShape(label="toy_animal", kind="rectangle", points=((40, 250), (100, 300))),
        DrawnShape(label="floor", kind="polygon", points=((0, 480), (0, 240), (640, 240))),
    ),
)


def test_a_draft_reads_back_like_a_drawing_and_is_unreviewed(tmp_path: Path) -> None:
    labelling = LabellingDirectory(tmp_path)
    labelling.add_image(ID, b"jpeg")
    assert labelling.image_ids() == [ID]
    assert not labelling.has_drawing(ID)
    assert labelling.save_draft(DRAFT)
    assert labelling.has_drawing(ID)
    assert labelling.load_drawings() == [DRAFT]  # the draft list itself is not a drawing
    assert labelling.unreviewed_drafts() == {ID}
    saved = json.loads((tmp_path / f"{ID}.json").read_text())
    assert saved["imagePath"] == f"{ID}.jpg"
    assert saved["checked"] is False


def test_a_draft_never_replaces_a_drawing(tmp_path: Path) -> None:
    (tmp_path / f"{ID}.json").write_text(x_anylabeling_file())
    labelling = LabellingDirectory(tmp_path)
    assert not labelling.save_draft(DRAFT)
    assert json.loads((tmp_path / f"{ID}.json").read_text())["version"] == "2.4.4"
    assert labelling.unreviewed_drafts() == set()


def test_saving_or_checking_or_deleting_a_draft_ends_its_draft_status(tmp_path: Path) -> None:
    labelling = LabellingDirectory(tmp_path)
    ids = ["20261121-172308-000001", "20261121-172308-000002", "20261121-172308-000003"]
    for record_id in ids:
        labelling.save_draft(DRAFT.model_copy(update={"record_id": record_id}))
    corrected = json.loads((tmp_path / f"{ids[0]}.json").read_text())
    corrected["shapes"].pop(0)
    (tmp_path / f"{ids[0]}.json").write_text(json.dumps(corrected))
    checked = json.loads((tmp_path / f"{ids[1]}.json").read_text())
    (tmp_path / f"{ids[1]}.json").write_text(json.dumps({**checked, "checked": True}))
    assert labelling.unreviewed_drafts() == {ids[2]}  # corrected and checked ones count
    (tmp_path / f"{ids[2]}.json").unlink()
    assert labelling.unreviewed_drafts() == set()


def test_a_broken_draft_list_is_reported(tmp_path: Path) -> None:
    (tmp_path / ".prelabels.json").write_text("[1, 2]")
    with pytest.raises(StorageError, match="not a list of drafts"):
        LabellingDirectory(tmp_path).unreviewed_drafts()
    (tmp_path / ".prelabels.json").write_text("{broken")
    with pytest.raises(StorageError, match="could not read"):
        LabellingDirectory(tmp_path).unreviewed_drafts()
