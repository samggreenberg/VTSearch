"""One cached model per converter instance, so a run loads it once rather than per item."""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from typing import Any


class ModelSlot:
    """Hold one loaded model, keyed by whatever decides which model it is.

    A converter that loads a heavy model (Whisper, PaddleOCR) keeps one of
    these on the instance, so a conversion run loads the model once instead of
    once per item. It holds a single model: a different key frees the old one
    before loading the new, so a finished ``large`` run does not keep 3 GB of
    the wrong checkpoint alive. A failed load leaves the slot empty, so the
    next call tries again.

    Callers hold :attr:`lock` across the load *and* every use of the model.
    The module-level ``CONVERTER`` singleton runs on every thread doing a
    dataset load, and neither engine can run two inferences at once: Whisper's
    decoder installs its kv-cache as forward hooks on the shared attention
    modules, and PaddleOCR's predictors share their input tensors. A fresh
    model per call was never shared, so serialising is the price of sharing.

    Put the loader itself in the key (``whisper.load_model``, the
    ``PaddleOCR`` class). Tests patch fake modules into ``sys.modules``, and a
    key that names the loader keeps one test's fake from serving the next.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._key: Hashable | None = None
        self._model: Any = None

    def get(self, key: Hashable, load: Callable[[], Any]) -> Any:
        """Return the model for *key*, calling *load* when the slot holds another or none.

        The caller must hold :attr:`lock`. An exception from *load* propagates
        and leaves the slot empty.
        """
        if self._model is not None and self._key == key:
            return self._model
        # Drop the old model first, so the two are never in memory together.
        self._key = None
        self._model = None
        model = load()
        self._key = key
        self._model = model
        return model
