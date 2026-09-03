#!/usr/bin/env python
"""Print the two runtime facts that decide which profile cell a run lands in.

The device key is not the GPU's name: ``fit.device_key`` collapses CUDA into
``cuda+cuml`` or ``cuda`` by whether cuML is serving the clustering, because
that split moves the finalize-shaped steps enough to matter. A run that does not
record which side it was on cannot be compared with one that did.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vtscore.timing.fit import device_key  # noqa: E402
from vtscore.timing.profile import cuml_active, resolve_device_name  # noqa: E402

device = resolve_device_name()
cuml = cuml_active()
print(f"device={device}")
print(f"cuml_active={cuml}")
print(f"profile_device_key={device_key(device, cuml)}")
