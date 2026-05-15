"""Test package for VTSearch."""


def load_detector_and_wait(client, detector_id, timeout=5.0):
    """Load a detector via POST and poll until the background task finishes.

    After loading completes, updates the current thread's detector context
    to point to the newly loaded detector so that subsequent proxy accesses
    (votes, etc.) resolve to the correct DetectorContext.

    Returns the initial POST response.
    """
    import time

    from vtsearch.concurrency.progress import detector_loading_tasks

    res = client.post("/api/detectors/registry/load", json={"detector_id": detector_id})
    if detector_id is None:
        from vtsearch.state.core import set_thread_detector_context

        set_thread_detector_context(None)
        return res
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        active = [t for t in detector_loading_tasks.list_tasks() if t.get("status") != "idle"]
        if not active:
            break
        time.sleep(0.05)

    # Update the test thread's context to the newly loaded model.
    from vtsearch.state.core import get_detector_context, set_thread_detector_context

    det_ctx = get_detector_context(detector_id)
    if det_ctx is not None:
        set_thread_detector_context(det_ctx)

    return res
