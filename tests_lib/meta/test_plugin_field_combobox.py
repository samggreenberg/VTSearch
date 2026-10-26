"""A free-text plugin select must render with the shared combobox, not a datalist.

A ``PluginField(field_type="select", allow_free_text=True)`` is a dropdown the
user may also type into.  Eight surfaces drew it as a native ``<input list>`` +
``<datalist>``, and a datalist cannot keep an option's value and label apart.
A plugin whose ``get_field_options`` returns ``(dataset_uuid, "my-cool-dataset")``
wants the name shown and the UUID submitted; Chrome instead draws the UUID as
each row's headline with the name in small print beneath, and picking a row
writes the UUID into the box (issue #5265).  Swapping the pair would show the
name, and then submit it too, so two datasets of the same name could no longer
be told apart.

``<vt-plugin-combobox>`` keeps the two apart: the box shows the label, the form
stores the value.  This gate holds every surface that renders a free-text select
to that widget, and keeps the native datalist from coming back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO / "frontend/src/app"

#: The shared widget every free-text select must delegate to.
COMBOBOX_TAG = "vt-plugin-combobox"
COMBOBOX_CLASS = "PluginComboboxComponent"

SELF = FRONTEND_SRC / "components/plugin-combobox/plugin-combobox.component.ts"


def _templates_rendering_free_text_selects() -> list[Path]:
    """Every ``.html`` template that has a branch for ``allow_free_text``."""
    found = [
        path
        for path in sorted(FRONTEND_SRC.rglob("*.html"))
        if "allow_free_text" in path.read_text(encoding="utf-8")
    ]
    assert found, "found no templates rendering allow_free_text selects — has the SPA moved?"
    return found


def test_allow_free_text_is_a_plugin_field_attribute() -> None:
    """Guard the premise: the attribute this gate keys on still exists."""
    from vtscore.plugins import PluginField  # noqa: PLC0415

    field = PluginField(key="q", label="Q", field_type="select", allow_free_text=True)
    assert field.allow_free_text is True


@pytest.mark.parametrize("template", _templates_rendering_free_text_selects(), ids=lambda p: p.stem)
def test_free_text_select_uses_shared_combobox(template: Path) -> None:
    """A template that renders a free-text select must use the shared widget..."""
    text = template.read_text(encoding="utf-8")
    assert f"<{COMBOBOX_TAG}" in text, (
        f"{template.relative_to(REPO)} renders allow_free_text selects but never "
        f"<{COMBOBOX_TAG}>; an option's label and value need the shared widget to "
        f"stay apart (issue #5265)."
    )
    assert "<datalist" not in text, (
        f"{template.relative_to(REPO)} renders a <datalist>, which shows a plugin "
        f"option's value as its headline and writes it into the box; use "
        f"<{COMBOBOX_TAG}> (issue #5265)."
    )


@pytest.mark.parametrize("template", _templates_rendering_free_text_selects(), ids=lambda p: p.stem)
def test_free_text_select_component_declares_shared_combobox(template: Path) -> None:
    """...and must import it, or Angular renders the tag as an unknown element."""
    component = template.with_suffix(".ts")
    assert component.exists(), f"{template} has no sibling component file"
    assert COMBOBOX_CLASS in component.read_text(encoding="utf-8"), (
        f"{component.relative_to(REPO)} renders <{COMBOBOX_TAG}> but does not list "
        f"{COMBOBOX_CLASS} in its standalone `imports`."
    )


def test_shared_combobox_component_exists() -> None:
    """The gate is worthless if the component it points everyone at is gone."""
    assert SELF.exists(), f"{SELF.relative_to(REPO)} is missing"
    text = SELF.read_text(encoding="utf-8")
    assert f"selector: '{COMBOBOX_TAG}'" in text
    assert f"class {COMBOBOX_CLASS}" in text
