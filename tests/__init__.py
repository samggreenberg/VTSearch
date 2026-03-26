"""Test package for VTSearch."""


def load_model_and_wait(client, model_id, timeout=5.0):
    """Load a model via POST and poll until the background task finishes.

    Returns the initial POST response.
    """
    import time

    res = client.post("/api/models/registry/load", json={"model_id": model_id})
    if model_id is None:
        return res
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        tasks_res = client.get("/api/models/loading-tasks")
        tasks = tasks_res.get_json().get("tasks", [])
        active = [t for t in tasks if t.get("status") != "idle"]
        if not active:
            break
        time.sleep(0.05)
    return res
