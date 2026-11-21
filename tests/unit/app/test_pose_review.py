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
from jetank.app.pose_review import PoseReviewApi, load_page
from jetank.app.review_server import make_server
from jetank.domain.servo import JointName, ServoState
from jetank.domain.teaching import (
    GraspRecord,
    GraspResult,
    GripPoint,
    TeachLabels,
    TeachStage,
    ToyPosture,
    record_id_for,
)
from jetank.drivers.fake import FakeGraspStore
from jetank.drivers.grasp_directory import GraspDirectory
from jetank.errors import StorageError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

T0 = datetime(2026, 11, 22, 9, 0, tzinfo=timezone.utc)
PAGE = b"<!doctype html><title>review</title>"


def labels(animal: str = "cow", result: GraspResult = GraspResult.GOOD) -> TeachLabels:
    return TeachLabels(
        animal=animal,
        stage=TeachStage.GRASP,
        posture=ToyPosture.STANDING,
        grip_point=GripPoint.BODY,
        result=result,
    )


def make_record(seconds: int = 0, animal: str = "cow") -> GraspRecord:
    moment = T0 + timedelta(seconds=seconds)
    base = ServoState(
        servo_id=1, position=575, voltage_v=6.4, temperature_c=25, moving=False, torque_enabled=True
    )
    return GraspRecord(
        record_id=record_id_for(moment),
        recorded_at=moment,
        labels=labels(animal),
        joints={JointName.BASE: base},
    )


@pytest.fixture
def store() -> FakeGraspStore:
    store = FakeGraspStore()
    store.save(make_record(1, "cow"), b"\xff\xd8cow")
    store.save(make_record(2, "lion"), None)
    return store


@pytest.fixture
def api(store: FakeGraspStore) -> PoseReviewApi:
    return PoseReviewApi(store, PAGE)


def put(api: PoseReviewApi, record_id: str, body: object, content_type: str | None = None) -> int:
    reply = api.handle(
        "PUT",
        f"/api/records/{record_id}",
        json.dumps(body).encode(),
        content_type or "application/json",
    )
    return reply.status


def test_page_is_served(api: PoseReviewApi) -> None:
    for path in ("/", "/index.html", "/?x=1"):
        reply = api.handle("GET", path, b"", None)
        assert reply.status is HTTPStatus.OK
        assert reply.body == PAGE
        assert reply.content_type.startswith("text/html")


def test_shipped_page_loads() -> None:
    assert b"/api/records" in load_page()


def test_listing_has_records_newest_first_summary_and_choices(
    api: PoseReviewApi, store: FakeGraspStore
) -> None:
    reply = api.handle("GET", "/api/records", b"", None)
    assert reply.status is HTTPStatus.OK
    listing = json.loads(reply.body)
    assert [r["labels"]["animal"] for r in listing["records"]] == ["lion", "cow"]
    assert any("cow" in line for line in listing["summary"])
    assert any("poses:" in line for line in listing["summary"])
    assert listing["choices"]["result"] == ["good", "shaky", "does_not_fit"]
    assert listing["choices"]["stage"][0] == "search"


def test_image_is_served_and_may_be_cached(api: PoseReviewApi, store: FakeGraspStore) -> None:
    cow, lion = store.records
    reply = api.handle("GET", f"/api/records/{cow.record_id}/image", b"", None)
    assert reply.status is HTTPStatus.OK
    assert reply.body == b"\xff\xd8cow"
    assert reply.content_type == "image/jpeg"
    assert "max-age" in reply.headers["Cache-Control"]
    missing = api.handle("GET", f"/api/records/{lion.record_id}/image", b"", None)
    assert missing.status is HTTPStatus.NOT_FOUND


def test_relabel_updates_the_record(api: PoseReviewApi, store: FakeGraspStore) -> None:
    cow = store.records[0]
    new = labels("Horse ", GraspResult.SHAKY).model_dump(mode="json")
    reply = api.handle(
        "PUT",
        f"/api/records/{cow.record_id}",
        json.dumps(new).encode(),
        "application/json; charset=utf-8",
    )
    assert reply.status is HTTPStatus.OK
    assert json.loads(reply.body)["labels"]["animal"] == "horse"
    assert store.records[0].labels == labels("horse", GraspResult.SHAKY)
    assert store.records[0].joints == cow.joints


def test_relabel_rejects_invalid_labels(api: PoseReviewApi, store: FakeGraspStore) -> None:
    cow = store.records[0]
    bad = {**labels().model_dump(mode="json"), "result": "great"}
    reply = api.handle(
        "PUT", f"/api/records/{cow.record_id}", json.dumps(bad).encode(), "application/json"
    )
    assert reply.status is HTTPStatus.BAD_REQUEST
    assert b"result" in reply.body
    assert put(api, cow.record_id, {**bad, "result": "good", "extra": 1}) == HTTPStatus.BAD_REQUEST
    assert (
        api.handle("PUT", f"/api/records/{cow.record_id}", b"{not json", "application/json").status
        is HTTPStatus.BAD_REQUEST
    )
    assert store.records[0] == cow


def test_relabel_requires_json(api: PoseReviewApi, store: FakeGraspStore) -> None:
    cow = store.records[0]
    body = labels().model_dump(mode="json")
    assert put(api, cow.record_id, body, "text/plain") == HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    reply = api.handle("PUT", f"/api/records/{cow.record_id}", b"{}", None)
    assert reply.status is HTTPStatus.UNSUPPORTED_MEDIA_TYPE


def test_delete_removes_the_record(api: PoseReviewApi, store: FakeGraspStore) -> None:
    cow, lion = store.records
    reply = api.handle("DELETE", f"/api/records/{cow.record_id}", b"", None)
    assert reply.status is HTTPStatus.NO_CONTENT
    assert store.records == [lion]
    again = api.handle("DELETE", f"/api/records/{cow.record_id}", b"", None)
    assert again.status is HTTPStatus.NOT_FOUND


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("PUT", "/"),
        ("DELETE", "/api/records"),
        ("GET", "/api/records/20261122-090001-000000"),
        ("DELETE", "/api/records/20261122-090001-000000/image"),
    ],
)
def test_wrong_methods_are_refused(api: PoseReviewApi, method: str, path: str) -> None:
    assert api.handle(method, path, b"", None).status is HTTPStatus.METHOD_NOT_ALLOWED


@pytest.mark.parametrize(
    "path",
    ["/teleop", "/api/records/20261122-090009-000000/image", "/api/records/a/b/image", "/api"],
)
def test_unknown_paths_are_not_found(api: PoseReviewApi, path: str) -> None:
    assert api.handle("GET", path, b"", None).status is HTTPStatus.NOT_FOUND


def test_storage_failures_are_reported(api: PoseReviewApi, store: FakeGraspStore) -> None:
    store.fail_with = StorageError("disk on fire")
    reply = api.handle("GET", "/api/records", b"", None)
    assert reply.status is HTTPStatus.INTERNAL_SERVER_ERROR
    assert b"disk on fire" in reply.body


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[GraspDirectory, int]]:
    """A real server on a free local port, over the records in ``tmp_path``."""
    directory = GraspDirectory(tmp_path)
    server = make_server(PoseReviewApi(directory, PAGE), "127.0.0.1", 0)
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
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_server_reviews_records_on_disk(served: tuple[GraspDirectory, int]) -> None:
    directory, port = served
    cow = directory.save(make_record(1, "cow"), b"\xff\xd8cow")
    status, headers, body = request(port, "GET", "/")
    assert (status, body) == (200, PAGE)
    assert headers["X-Content-Type-Options"] == "nosniff"
    status, _, body = request(port, "GET", f"/api/records/{cow.record_id}/image")
    assert (status, body) == (200, b"\xff\xd8cow")

    new = json.dumps(labels("sheep").model_dump(mode="json")).encode()
    status, _, body = request(
        port, "PUT", f"/api/records/{cow.record_id}", new, {"Content-Type": "application/json"}
    )
    assert status == 200
    assert directory.load_all()[0].labels.animal == "sheep"

    status, headers, body = request(port, "DELETE", f"/api/records/{cow.record_id}")
    assert (status, body) == (204, b"")
    assert directory.load_all() == []


def test_server_refuses_oversized_or_bad_bodies(served: tuple[GraspDirectory, int]) -> None:
    _, port = served
    big = b"x" * (review_server.MAX_BODY_BYTES + 1)
    assert request(port, "PUT", "/api/records/x", big)[0] == 400
    assert request(port, "PUT", "/api/records/x", None, {"Content-Length": "lots"})[0] == 400
    assert request(port, "POST", "/api/records")[0] == 501


def test_cli_review_needs_an_existing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["poses", "review", "--directory", str(tmp_path / "none")]) == 1
    assert "no grasp records directory" in capsys.readouterr().err


def test_cli_review_reports_a_busy_port(
    served: tuple[GraspDirectory, int], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, port = served
    args = ["poses", "review", "--directory", str(tmp_path), "--host", "127.0.0.1"]
    assert main([*args, "--port", str(port)]) == 1
    assert "could not listen" in capsys.readouterr().err


def test_cli_review_serves_until_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(self: object, poll_interval: float = 0.5) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(review_server.HTTPServer, "serve_forever", interrupted)
    args = ["poses", "review", "--directory", str(tmp_path), "--host", "127.0.0.1", "--port", "0"]
    assert main(args) == 0
    assert "pose review at http://" in capsys.readouterr().out
