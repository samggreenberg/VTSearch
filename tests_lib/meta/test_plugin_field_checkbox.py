"""Every surface that renders ``PluginField``\\ s must render a *real* checkbox.

The Angular SPA has ten near-duplicate blocks that branch on
``field.field_type`` to pick a widget — one per modal/picker that shows plugin
configuration (results exporters, settings importers and exporters, label
importers, the New Model form, Auto-Find, the autodetect results modal, the
dataset-importer pickers).  When ``"checkbox"`` was added to
:data:`vtscore.plugins.FieldType`, only two of those ten chains grew a branch
for it.  On the other eight, a checkbox field fell through the chain to the
generic ``<input type="text">`` fallback, so a plugin author who wrote

    PluginField(key="enable_email", label="Send via Email", field_type="checkbox")

got a text box the user was invited to *type* ``true`` into (issue #3851).

It stayed invisible for as long as it did because every checkbox field shipped
in this repo belongs to a dataset importer, and dataset importers are rendered
by ``generic-form-picker`` — one of the two chains that did handle it.  So the
bug was unreachable from inside the repo and only a third-party plugin could
find it.

This gate closes that gap from the other side: rather than checking that any
particular plugin renders correctly, it checks the *rendering surfaces*
themselves.  A template that decides what widget to draw from ``field_type``
must delegate the checkbox to ``<vt-plugin-checkbox>``, and the component
behind it must be declared in the owning component's ``imports``.  A new
surface therefore cannot ship a tenth copy of the same omission, and the
shared component keeps the string/bool coercion (``'true'``/``'false'`` on the
wire, ``checked`` in the DOM) in one place instead of ten.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO / "frontend/src/app"

#: The shared widget every field-rendering surface must delegate to.
CHECKBOX_TAG = "vt-plugin-checkbox"
CHECKBOX_CLASS = "PluginCheckboxComponent"

#: The component that *defines* the widget is exempt from having to use it.
SELF = FRONTEND_SRC / "components/plugin-checkbox/plugin-checkbox.component.ts"

#: Matches a template branching on a plugin field's declared type, in either
#: spelling the templates use for the loop variable (``field`` or ``f``).
FIELD_TYPE_BRANCH = re.compile(r"\b\w+\.field_type\s*===")

#: A single ``<input …>`` element, attributes and all.
INPUT_TAG = re.compile(r"<input\b[^>]*>", re.S)

#: An attribute addressing the plugin field the loop is on, by key.
FIELD_KEY_BINDING = re.compile(r"\b\w+\.key\b")


def _templates_that_branch_on_field_type() -> list[Path]:
    """Every ``.html`` template that picks a widget from ``field_type``."""
    found = [
        path
        for path in sorted(FRONTEND_SRC.rglob("*.html"))
        if FIELD_TYPE_BRANCH.search(path.read_text(encoding="utf-8"))
    ]
    assert found, "found no templates branching on field_type — has the SPA moved?"
    return found


def _owning_component(template: Path) -> Path:
    """The ``.ts`` file whose ``templateUrl`` is *template*."""
    sibling = template.with_suffix(".ts")
    assert sibling.exists(), f"{template} has no sibling component file"
    return sibling


def test_checkbox_is_a_declared_field_type() -> None:
    """Guard the premise: the literal this gate is named for still exists.

    If ``"checkbox"`` is ever renamed or dropped from ``FieldType``, every
    assertion below would keep passing while guarding nothing.
    """
    from vtscore.plugins import FieldType, PluginField  # noqa: PLC0415

    assert "checkbox" in FieldType.__args__
    # And the dataclass still accepts it, so the gate tracks a live surface.
    field = PluginField(key="flag", label="Flag", field_type="checkbox")
    assert field.field_type == "checkbox"


@pytest.mark.parametrize("template", _templates_that_branch_on_field_type(), ids=lambda p: p.stem)
def test_field_rendering_template_uses_shared_checkbox(template: Path) -> None:
    """A widget-picking template must draw checkboxes with the shared widget."""
    text = template.read_text(encoding="utf-8")
    assert CHECKBOX_TAG in text, (
        f"{template.relative_to(REPO)} chooses a widget from field_type but never "
        f"renders <{CHECKBOX_TAG}>, so a PluginField(field_type='checkbox') falls "
        f"through to this template's text-input fallback (issue #3851). Render "
        f"<{CHECKBOX_TAG}> inside the field's <label>, and exclude 'checkbox' from "
        f"the fallback branch."
    )


@pytest.mark.parametrize("template", _templates_that_branch_on_field_type(), ids=lambda p: p.stem)
def test_field_rendering_component_declares_shared_checkbox(template: Path) -> None:
    """...and must import it, or Angular renders the tag as an unknown element."""
    component = _owning_component(template)
    text = component.read_text(encoding="utf-8")
    assert CHECKBOX_CLASS in text, (
        f"{component.relative_to(REPO)} renders <{CHECKBOX_TAG}> but does not list "
        f"{CHECKBOX_CLASS} in its standalone `imports`."
    )


def test_no_hand_rolled_plugin_field_checkbox() -> None:
    """No surface may open-code the widget instead of using the shared one.

    A hand-rolled ``<input type="checkbox">`` is how the two chains that *did*
    support checkboxes came to disagree with each other: one compared the value
    against the string ``'true'`` only, the other against the string or a bool,
    and neither matched ``vtscore.plugins.parse_checkbox``'s case-insensitive
    reading of a ``default="False"``.

    Only checkboxes *bound to a plugin field* are in scope.  These templates
    also carry ordinary checkboxes of their own — the Export window's column
    toggles, the source picker's per-media-type switches — which are hand-built
    by design and have nothing to do with ``PluginField``.  An input is taken
    to be a plugin field's when its own attributes address the field by key
    (``field.key`` / ``f.key``), which is exactly what the two replaced
    implementations did and what the unrelated checkboxes never do.
    """
    offenders = []
    for template in _templates_that_branch_on_field_type():
        text = template.read_text(encoding="utf-8")
        for match in INPUT_TAG.finditer(text):
            tag = match.group(0)
            if 'type="checkbox"' not in tag and "type='checkbox'" not in tag:
                continue
            if not FIELD_KEY_BINDING.search(tag):
                continue
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{template.relative_to(REPO)}:{line}")
    assert not offenders, (
        'hand-rolled <input type="checkbox"> bound to a plugin field; use '
        f"<{CHECKBOX_TAG}> so the true/false coercion stays single-sourced: " + ", ".join(offenders)
    )


def test_shared_checkbox_component_exists() -> None:
    """The gate is worthless if the component it points everyone at is gone."""
    assert SELF.exists(), f"{SELF.relative_to(REPO)} is missing"
    text = SELF.read_text(encoding="utf-8")
    assert f"selector: '{CHECKBOX_TAG}'" in text
    assert f"class {CHECKBOX_CLASS}" in text
