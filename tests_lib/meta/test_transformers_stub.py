"""The local ``transformers`` stub names only what the installed package exports (#5077).

``typings/transformers/__init__.pyi`` stands in for the real package so pyright
does not bind its ~2,600-file model cascade, and it declares every name as
``Any``. That gives up the one check of the real package our code relied on:
that a name we import from ``transformers`` exists. pyright still holds our
imports to the stub, and this file holds the stub to the installed package, so
the two together make the same claim.
"""

from __future__ import annotations

import ast
from pathlib import Path

STUB = Path(__file__).resolve().parents[2] / "typings" / "transformers" / "__init__.pyi"


def _stub_tree() -> ast.Module:
    return ast.parse(STUB.read_text(encoding="utf-8"))


def _declared_names() -> set[str]:
    return {
        node.target.id
        for node in _stub_tree().body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def test_stub_declares_the_names_we_import() -> None:
    assert {"__version__", "AutoModel", "SiglipModel"} <= _declared_names()


def test_every_stub_name_exists_in_installed_transformers() -> None:
    import transformers

    # ``dir()`` on transformers' lazy module lists every lazy export without
    # importing the model code behind it. It leaves out the extra objects the
    # module serves from ``__getattr__`` (``__version__``), so those get a
    # ``hasattr``, which is cheap for any name ``dir()`` does not list.
    unlisted = _declared_names() - set(dir(transformers))
    missing = sorted(name for name in unlisted if not hasattr(transformers, name))
    assert not missing, (
        f"typings/transformers/__init__.pyi declares names transformers {transformers.__version__} "
        f"does not export: {missing}. Drop them from the stub, or fix the import that needed them."
    )


def test_stub_has_no_catch_all() -> None:
    """A module ``__getattr__`` would let a misspelled ``from transformers import X`` pass pyright."""
    functions = {node.name for node in _stub_tree().body if isinstance(node, ast.FunctionDef)}
    assert "__getattr__" not in functions
