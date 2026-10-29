"""``vtscore.detectors.labelset_training`` keeps its public surface after the split (#5423).

The module's helpers moved to four siblings - ``labelset_embeddings``,
``labelset_matrix``, ``labelset_structural`` and ``labelset_heads`` - and the two
training entry points stayed.  What an out-of-tree caller imports from
``labelset_training`` must still resolve, to the moved object itself, and the
layering the split rests on must hold: the entry points read the helpers, and
no helper reads back.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import vtscore.detectors.labelset_training as lt

HELPERS = ["labelset_embeddings", "labelset_matrix", "labelset_structural", "labelset_heads"]
DETECTORS = Path(lt.__file__).parent


@pytest.mark.parametrize("name", lt.__all__)
def test_each_public_name_is_its_owners_object(name):
    if name in ("train_from_labelset", "labelset_train_and_score"):
        assert getattr(lt, name).__module__ == lt.__name__
        return
    modules = [importlib.import_module(f"vtscore.detectors.{m}") for m in HELPERS]
    owners = [m for m in modules if name in vars(m)]
    assert len(owners) == 1, f"{name} is in {[m.__name__ for m in owners]}"
    assert getattr(lt, name) is getattr(owners[0], name)


def _imported_detector_modules(module: str) -> set[str]:
    """The ``vtscore.detectors.labelset_*`` modules *module* imports, at top level or inside a function."""
    tree = ast.parse((DETECTORS / f"{module}.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return {n.rsplit(".", 1)[1] for n in names if n.startswith("vtscore.detectors.labelset_")}


@pytest.mark.parametrize("module", HELPERS)
def test_no_helper_reads_back(module):
    imported = _imported_detector_modules(module) & {"labelset_training", *HELPERS}
    # Every helper may read ``_lookups_for_snap`` from the embeddings module; nothing else.
    assert imported <= {"labelset_embeddings"} - {module}, f"{module} imports {sorted(imported)}"
