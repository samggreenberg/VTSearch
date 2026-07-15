"""Test package for VTSearch."""


def load_detector_and_wait(client, detector_id, timeout=30.0):
    """Load a detector via POST and poll until the background task finishes.

    After loading completes, updates the current thread's detector context
    to point to the newly loaded detector so that subsequent proxy accesses
    (votes, etc.) resolve to the correct DetectorContext.

    Fails loudly via ``pytest.fail`` if the background load does not finish
    within ``timeout`` seconds, reporting the still-active tasks. Silently
    exhausting the deadline used to leave callers seeing 404s / empty
    ``label_embeddings`` on many-worker xdist runs.

    Returns the initial POST response.
    """
    import time

    import pytest

    from vtscore.concurrency.progress import detector_loading_tasks

    res = client.post("/api/detectors/registry/load", json={"detector_id": detector_id})
    if detector_id is None:
        from vtscore.state.core import set_thread_detector_context

        set_thread_detector_context(None)
        return res
    deadline = time.monotonic() + timeout
    active = None
    while time.monotonic() < deadline:
        active = [t for t in detector_loading_tasks.list_tasks() if t.get("status") != "idle"]
        if not active:
            break
        time.sleep(0.05)
    else:
        pytest.fail(
            f"load_detector_and_wait timed out after {timeout}s waiting for "
            f"detector {detector_id!r} to load; active tasks: {active!r}"
        )

    # Update the test thread's context to the newly loaded model.
    from vtscore.state.core import get_detector_context, set_thread_detector_context

    det_ctx = get_detector_context(detector_id)
    if det_ctx is not None:
        set_thread_detector_context(det_ctx)

    return res
