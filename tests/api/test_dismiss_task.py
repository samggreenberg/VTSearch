"""``POST /api/dataset/dismiss/<task_id>`` and the per-user ``/api/events`` stream.

A failed dataset task stays on the ``loading-tasks`` channel until the user who
started it dismisses it, and only that user sees it (#4973).
"""

import json

from vtscore.concurrency.progress import loading_tasks
from vtsearch.auth import LoginProvider, set_login_provider


class _HeaderProvider(LoginProvider):
    """Multi-user stand-in: the ``X-User`` header names the caller."""

    name = "header"

    def get_user(self, request):
        return request.headers.get("X-User") or "anonymous"

    def is_authenticated(self, request):
        return "X-User" in request.headers


def _fail(task_id: str, created_by: str = "alice") -> None:
    tracker = loading_tasks.create_task(task_id, "Doomed import", created_by=created_by, source="server_folder:/x")
    tracker.update("idle", "", 0, 0, error="Import produced no medias.")
    loading_tasks.mark_finished(task_id)


def _as(user: str) -> dict[str, str]:
    return {"X-User": user}


class TestDismissRoute:
    def test_dismisses_the_callers_own_failure(self, client):
        set_login_provider(_HeaderProvider())
        _fail("imp")
        resp = client.post("/api/dataset/dismiss/imp", headers=_as("alice"))
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        assert loading_tasks.list_tasks() == []

    def test_another_users_failure_is_not_found(self, client):
        set_login_provider(_HeaderProvider())
        _fail("imp", created_by="alice")
        resp = client.post("/api/dataset/dismiss/imp", headers=_as("bob"))
        assert resp.status_code == 404
        assert [t["task_id"] for t in loading_tasks.list_tasks()] == ["imp"]

    def test_unknown_task_is_not_found(self, client):
        assert client.post("/api/dataset/dismiss/nope").status_code == 404

    def test_running_task_is_refused(self, client):
        loading_tasks.create_task("running", "Still going")
        resp = client.post("/api/dataset/dismiss/running")
        assert resp.status_code == 409
        assert [t["task_id"] for t in loading_tasks.list_tasks()] == ["running"]

    def test_single_user_mode_dismisses(self, client):
        _fail("imp", created_by="default")
        assert client.post("/api/dataset/dismiss/imp").status_code == 200


def _first_loading_tasks_frame(client, headers: dict[str, str]) -> list[dict]:
    resp = client.get("/api/events", buffered=False, headers=headers)
    try:
        for chunk in resp.response:
            frame = chunk.decode() if isinstance(chunk, bytes) else chunk
            if frame.startswith("event: loading-tasks\n"):
                data_line = next(ln for ln in frame.splitlines() if ln.startswith("data: "))
                return json.loads(data_line.removeprefix("data: "))
        raise AssertionError("stream ended before its loading-tasks snapshot")
    finally:
        resp.close()


class TestEventsRouteIsPerUser:
    def test_a_failure_streams_to_its_owner_only(self, client):
        set_login_provider(_HeaderProvider())
        _fail("imp", created_by="alice")

        mine = _first_loading_tasks_frame(client, _as("alice"))
        assert [t["task_id"] for t in mine] == ["imp"]
        assert mine[0]["source"] == "server_folder:/x"
        assert _first_loading_tasks_frame(client, _as("bob")) == []
