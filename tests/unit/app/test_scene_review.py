from __future__ import annotations

import http.client
import json
import threading
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from typing import TYPE_CHECKING

import pytest

from jetank.app import review_server
from jetank.app.cli import main
from jetank.app.review_server import Reply, make_server
from jetank.app.scene_review import SceneReviewApi, load_page
from jetank.domain.records import record_id_for
from jetank.domain.scenes import Lighting, SceneLabels, SceneRecord
from jetank.drivers.fake import FakeSceneStore
from jetank.drivers.scene_directory import SceneDirectory
from jetank.errors import StorageError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

T0 = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)
PAGE = b"<!doctype html><title>scenes</title>"
KITCHEN = SceneLabels(room="kitchen", lighting=Lighting.LAMP, note="home box for the animals")
HALL = SceneLabels(room="hall", lighting=Lighting.DAYLIGHT, has_animals=False)


def make_scene(seconds: int, labels: SceneLabels) -> SceneRecord:
    moment = T0 + timedelta(seconds=seconds)
    return SceneRecord(record_id=record_id_for(moment), recorded_at=moment, labels=labels)


@pytest.fixture
def store() -> FakeSceneStore:
    store = FakeSceneStore()
    store.save(make_scene(1, KITCHEN), b"\xff\xd8kitchen")
    store.save(make_scene(2, HALL), b"\xff\xd8hall")
    return store


@pytest.fixture
def api(store: FakeSceneStore) -> SceneReviewApi:
    return SceneReviewApi(store, PAGE)


def put(api: SceneReviewApi, record_id: str, body: object) -> Reply:
    return api.handle(
        "PUT", f"/api/records/{record_id}", json.dumps(body).encode(), "application/json"
    )


def test_page_is_served(api: SceneReviewApi) -> None:
    reply = api.handle("GET", "/", b"", None)
    assert (reply.status, reply.body) == (HTTPStatus.OK, PAGE)


def test_shipped_page_loads() -> None:
    page = load_page()
    assert b"/api/records" in page
    assert b"scene review" in page


def test_listing_has_scenes_oldest_first_summary_and_choices(api: SceneReviewApi) -> None:
    reply = api.handle("GET", "/api/records", b"", None)
    assert reply.status is HTTPStatus.OK
    listing = json.loads(reply.body)
    assert [r["labels"]["room"] for r in listing["records"]] == ["kitchen", "hall"]
    assert listing["lighting"] == ["daylight", "lamp", "dim"]
    assert listing["rooms"] == ["hall", "kitchen"]
    assert any("With animals: 1" in line for line in listing["summary"])


def test_listing_of_no_scenes_is_empty() -> None:
    listing = json.loads(
        SceneReviewApi(FakeSceneStore(), PAGE).handle("GET", "/api/records", b"", None).body
    )
    assert listing == {
        "records": [],
        "summary": [],
        "lighting": ["daylight", "lamp", "dim"],
        "rooms": [],
    }


def test_image_is_served(api: SceneReviewApi, store: FakeSceneStore) -> None:
    reply = api.handle("GET", f"/api/records/{store.records[0].record_id}/image", b"", None)
    assert (reply.status, reply.body) == (HTTPStatus.OK, b"\xff\xd8kitchen")
    assert reply.content_type == "image/jpeg"


def test_relabel_fixes_the_labels_and_clears_the_note(
    api: SceneReviewApi, store: FakeSceneStore
) -> None:
    kitchen = store.records[0]
    new = {"room": " Living Room", "lighting": "dim", "has_animals": False, "note": ""}
    reply = put(api, kitchen.record_id, new)
    assert reply.status is HTTPStatus.OK
    assert store.records[0].labels == SceneLabels(
        room="living room", lighting=Lighting.DIM, has_animals=False
    )
    assert store.records[0].image_file == kitchen.image_file


@pytest.mark.parametrize(
    "body",
    [
        {"room": "kitchen", "lighting": "neon"},
        {"room": "", "lighting": "lamp"},
        {"room": "kitchen", "lighting": "lamp", "note": "x" * 201},
        {"room": "kitchen", "lighting": "lamp", "animal": "cow"},
    ],
)
def test_relabel_rejects_invalid_labels(
    api: SceneReviewApi, store: FakeSceneStore, body: dict[str, object]
) -> None:
    before = list(store.records)
    reply = put(api, before[0].record_id, body)
    assert reply.status is HTTPStatus.BAD_REQUEST
    assert store.records == before


def test_delete_removes_the_scene(api: SceneReviewApi, store: FakeSceneStore) -> None:
    kitchen, hall = store.records
    reply = api.handle("DELETE", f"/api/records/{kitchen.record_id}", b"", None)
    assert reply.status is HTTPStatus.NO_CONTENT
    assert store.records == [hall]
    assert store.deleted == [kitchen]
    again = api.handle("DELETE", f"/api/records/{kitchen.record_id}", b"", None)
    assert again.status is HTTPStatus.NOT_FOUND
    assert b"no scene" in again.body


def test_storage_failures_are_reported(api: SceneReviewApi, store: FakeSceneStore) -> None:
    store.fail_with = StorageError("card full")
    reply = api.handle("GET", "/api/records", b"", None)
    assert reply.status is HTTPStatus.INTERNAL_SERVER_ERROR
    assert b"card full" in reply.body


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[SceneDirectory, int]]:
    """A real server on a free local port, over the scenes in ``tmp_path``."""
    directory = SceneDirectory(tmp_path)
    server = make_server(SceneReviewApi(directory, PAGE), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield directory, server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_server_reviews_scenes_on_disk(served: tuple[SceneDirectory, int]) -> None:
    directory, port = served
    kitchen = directory.save(make_scene(1, KITCHEN), b"\xff\xd8kitchen")
    assert request(port, "GET", f"/api/records/{kitchen.record_id}/image") == (
        200,
        b"\xff\xd8kitchen",
    )

    body = json.dumps({**KITCHEN.model_dump(mode="json"), "note": ""}).encode()
    status, _ = request(
        port, "PUT", f"/api/records/{kitchen.record_id}", body, {"Content-Type": "application/json"}
    )
    assert status == 200
    assert directory.load_all()[0].labels.note == ""

    assert request(port, "DELETE", f"/api/records/{kitchen.record_id}") == (204, b"")
    assert directory.load_all() == []


def test_cli_review_needs_an_existing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["scenes", "review", "--directory", str(tmp_path / "none")]) == 1
    assert "no scenes directory" in capsys.readouterr().err


def test_cli_review_serves_until_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(self: object, poll_interval: float = 0.5) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(review_server.HTTPServer, "serve_forever", interrupted)
    args = ["scenes", "review", "--directory", str(tmp_path), "--host", "127.0.0.1", "--port", "0"]
    assert main(args) == 0
    assert "scene review at http://" in capsys.readouterr().out


def test_cli_summary_is_still_the_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["scenes", "summary", "--directory", str(tmp_path / "none")]) == 0
    assert "no scenes" in capsys.readouterr().out
