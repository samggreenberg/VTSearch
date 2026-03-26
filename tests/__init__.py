"""Test package for VTSearch."""


def load_model_and_wait(client, model_id, timeout=5.0):
    """Load a model via POST and poll until the background task finishes.

    After loading completes, updates the current thread's detector context
    to point to the newly loaded model so that subsequent proxy accesses
    (votes, etc.) resolve to the correct DetectorContext.

    Returns the initial POST response.
    """
    import time

    res = client.post("/api/models/registry/load", json={"model_id": model_id})
    if model_id is None:
        from vtsearch.utils.state_core import set_thread_detector_context
        set_thread_detector_context(None)
        return res
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        tasks_res = client.get("/api/models/loading-tasks")
        tasks = tasks_res.get_json().get("tasks", [])
        active = [t for t in tasks if t.get("status") != "idle"]
        if not active:
            break
        time.sleep(0.05)

    # Update the test thread's context to the newly loaded model.
    from vtsearch.utils.state_core import get_detector_context, set_thread_detector_context

    det_ctx = get_detector_context(model_id)
    if det_ctx is not None:
        set_thread_detector_context(det_ctx)

    return res
