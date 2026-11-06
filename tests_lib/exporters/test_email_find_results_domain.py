"""The email exporter carries each detector's training-domain check (#5501).

A Find or AutoFind result says, per detector, how much of the scored dataset
looks atypical of the detector's training dataset (``training_domain``, the
check Tune warns with).  An unattended AutoFind's email is where that is first
read, so it prints the figure and the verdict under the detector, and the
default subject flags a likely shift.  A payload without the check (the CLI,
an older caller) renders as it always has.
"""

from __future__ import annotations

from vtscore.exporters.email_smtp import _build_html, _build_plain_text, _default_subject


def _check(frac: float, *, shifted: bool, reference: str = "Caltech-101 (S)") -> dict:
    return {
        "reference_dataset_id": "ds-ref",
        "reference_dataset_name": reference,
        "labels_total": 39,
        "labels_in_reference": 39,
        "labels_in_active": 0,
        "domain_shift": {"frac_atypical": frac, "shifted": shifted},
        "reason": "",
    }


def _results(check: dict | None) -> dict:
    result = {
        "detector_name": "Mammal",
        "threshold": 0.5,
        "total_hits": 1,
        "hits": [{"id": 1, "filename": "a.jpg", "score": 0.9}],
        "negative_hits": [],
    }
    if check is not None:
        result["training_domain"] = check
    return {"media_type": "image", "detectors_run": 1, "results": {"Mammal": result}}


class TestDomainWarning:
    def test_a_likely_shift_is_a_warning_with_its_figure(self):
        results = _results(_check(0.685, shifted=True))
        line = "69% of this dataset looks atypical vs Caltech-101 (S): likely domain shift"
        assert f"Domain warning: {line}" in _build_plain_text(results)
        assert f"Domain warning:</strong> {line}" in _build_html(results)
        assert _default_subject(results).endswith("(domain warning)")

    def test_in_domain_is_reported_without_a_warning(self):
        results = _results(_check(0.04, shifted=False))
        assert "Domain check: 4% of this dataset looks atypical vs Caltech-101 (S): largely in-domain" in (
            _build_plain_text(results)
        )
        assert "Domain warning" not in _build_html(results)
        assert "domain warning" not in _default_subject(results)

    def test_a_check_that_could_not_run_says_nothing(self):
        refused = {**_check(0.0, shifted=False), "domain_shift": None, "reason": "No dataset to compare with"}
        for results in (_results(refused), _results(None)):
            assert "Domain" not in _build_plain_text(results)
            assert "Domain" not in _build_html(results)
            assert "domain warning" not in _default_subject(results)

    def test_the_reference_name_is_escaped_in_html(self):
        results = _results(_check(0.7, shifted=True, reference="<b>photos</b>"))
        assert "&lt;b&gt;photos&lt;/b&gt;" in _build_html(results)
