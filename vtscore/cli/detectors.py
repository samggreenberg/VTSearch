"""The run's detectors: read each one, decide how it re-clips the dataset, and build its head.

:func:`_train_detectors_for_first_chunk` trains every AutoFind (or override)
detector once, against the first chunk, and hands back the routed snapshots
(:class:`_RoutedSnapshots`) their thresholds were calibrated on, so scoring
reads the same population (#3647).  A clipper override (``--override-clipper``,
#5282) is validated here (:func:`_resolve_override_clipper`) and stands in for
every detector's own ``input_spec``.
"""

from __future__ import annotations

from typing import Any

from vtscore.cli import progress as cli_progress


def _describe_clipper_spec(spec: dict[str, Any] | None) -> str:
    """``'sound_tiling' (duration=5.0, min_overlap=0.0)``: an ``input_spec`` in one phrase."""
    if not spec or not spec.get("clipper"):
        return "(none)"
    params = spec.get("clipper_params") or {}
    listed = f" ({', '.join(f'{k}={v}' for k, v in params.items())})" if params else ""
    return f"{spec['clipper']!r}{listed}"


def _resolve_override_clipper(override_clipper: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate an ``override_clipper`` and return it with its effective parameters.

    *override_clipper* is ``input_spec``-shaped - ``{"clipper": name,
    "clipper_params": {...}}`` - and goes through
    :func:`~vtscore.detectors.input_spec.resolve_clipper_spec`, so an unknown
    clipper or parameter fails before any media is loaded.  ``None`` (no
    override) passes through.
    """
    if override_clipper is None:
        return None
    from vtscore.detectors.input_spec import resolve_clipper_spec  # noqa: PLC0415

    name = override_clipper.get("clipper") if isinstance(override_clipper, dict) else None
    if not isinstance(name, str) or not name:
        raise ValueError(f"override_clipper must be a mapping with a 'clipper' name, got {override_clipper!r}.")
    params = override_clipper.get("clipper_params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"override_clipper's 'clipper_params' must be a mapping, got {params!r}.")
    return resolve_clipper_spec(name, params)


def _detector_group_key(
    target_type: str,
    embedder_name: str,
    clipper: str,
    clipper_params: dict[str, Any],
) -> tuple[str, str, str, tuple]:
    """The identity of a scoring snapshot: what it is routed, clipped and embedded to.

    Detectors sharing a key share one prepared snapshot, so a dataset scored by
    two image/siglip detectors at the same granularity is converted, re-clipped
    and embedded once.  Calibration and scoring both key on this, from the one
    definition, so the population the threshold is fitted on cannot drift from
    the population it is applied to (issue #3647).
    """
    return (
        target_type,
        embedder_name,
        clipper,
        tuple(sorted((str(k), str(v)) for k, v in (clipper_params or {}).items())),
    )


class _RoutedSnapshots:
    """One :func:`~vtscore.detectors.converter_routing.route_and_embed` pass per
    detector group, over one chunk of medias.

    Threshold calibration and scoring want the **same** routed snapshot: the
    fold-anchored cut is realized as a quantile of the population it is about
    to cut, so fitting it on the loaded medias while inference reads the
    converted/re-clipped ones lands the quantile on the wrong distribution
    (issue #3647).  Routing twice would also pay twice - a converter re-decodes
    the video, a re-clip re-slices and re-embeds every clip - so the first
    chunk, which is both the chunk detectors are trained on and a chunk that
    gets scored, memoises its snapshots here and hands the same objects to both
    passes.

    Scoped to one chunk deliberately: later chunks route themselves and are
    scored at the threshold the first chunk fixed, which is what keeps every
    chunk of a streaming run judged by one cut.
    """

    def __init__(self, medias: dict[int, dict[str, Any]]) -> None:
        self._medias = medias
        self._cache: dict[tuple[str, str, str, tuple], tuple[dict[int, dict[str, Any]], dict[int, int]]] = {}

    def get(self, key: tuple[str, str, str, tuple]) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
        cached = self._cache.get(key)
        if cached is None:
            from vtscore.detectors.converter_routing import route_and_embed  # noqa: PLC0415

            target_type, embedder_name, clipper, clipper_params_items = key
            cached = route_and_embed(
                self._medias,
                target_type,
                embedder_name,
                clipper=clipper,
                clipper_params=dict(clipper_params_items),
            )
            self._cache[key] = cached
        return cached


def _load_and_train_detectors(
    detector_names: list[str],
    media_type: str,
    snap: dict[int, dict[str, Any]],
    routed: "_RoutedSnapshots | None" = None,
    override_clipper: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve, re-embed, and train a head for each named detector.

    For every name in *detector_names* the on-disk JSON is read, the
    labelset's origins are resolved via their importer's ``resolve_file()``,
    the files are embedded with the dataset's embedder, and the head the
    labels give is built from the resulting vectors: a trained head, or, under
    the label quota (or a typed query's hold, #4736), the Goods' centroid or
    the typed query's head.  Detectors whose ``media_type`` has no direct or
    one-hop converter route from the dataset's types are skipped with a
    ``detector_skipped`` event.  Detectors that declare an ``input_spec`` (a
    clipper the detector was trained on) the loaded dataset wasn't clipped to
    match have the dataset re-clipped to that granularity at scoring time,
    announced with a ``detector_reclip`` event.  An *override_clipper* (an
    ``input_spec`` from :func:`_resolve_override_clipper`) stands in for every
    detector's own ``input_spec``, so each is scored at that granularity
    instead (:func:`_override_reclip`); one whose type the clipper does not
    clip raises :class:`ValueError`.

    When *routed* is given, each detector's threshold is calibrated on the
    snapshot **scoring will read** - the converted, re-clipped, re-embedded one
    that :class:`_RoutedSnapshots` prepares - rather than on the loaded medias.
    The two are the same set on a natively-typed dataset the detector needs no
    re-clip for, which is why this went unnoticed; they are systematically
    different everywhere else, and the cut is realized as a quantile of
    whichever it is handed (issue #3647).  ``None`` keeps the loaded medias as
    the haystack, for callers with no scoring pass to agree with.

    Returns a ``{name: {"mlp": nn.Sequential, "threshold": float, "balance": dict, ...}}``
    map; ``balance`` is what the balance says about ``threshold``
    (:func:`_record_line_state`).  Raises :class:`ValueError` if a detector cannot be trained - for example
    when none of its labels' origin files are resolvable from the CLI
    environment.
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.converter_routing import detector_can_score
    from vtscore.detectors.input_spec import extract_input_spec_from_medias
    from vtscore.detectors.store import _detector_path, _read_detector
    from vtscore.detectors.labelset_training import Haystack, train_from_labelset
    from vtscore.detectors.query_head import detector_text_query, head_score_embedder
    from vtscore.state import seed_detector_beta
    from vtscore.state.core import DetectorContext

    dataset_spec = extract_input_spec_from_medias(snap)
    # The dataset can hold mixed source types (a folder of videos + PDFs);
    # match each detector against the whole set, not just the first media's
    # type, so a converter-reachable detector isn't skipped on iteration order.
    source_types = {m.get("media_type") or "" for m in snap.values()}

    out: dict[str, dict[str, Any]] = {}
    for det_name in detector_names:
        det = _read_detector(_detector_path(det_name))
        if det is None:
            raise ValueError(f"Detector '{det_name}' not found in the detectors dir.")

        det_media_type = det.get("media_type", "") or ""
        # A detector matches when its target type is present directly or is
        # reachable from some source type via a one-hop converter route (so one
        # image detector scores native images, ``video2image``, and
        # ``document2image`` in the same run). Legacy detectors with no
        # media_type match anything, as before.
        if det_media_type and not detector_can_score(det_media_type, source_types):
            cli_progress.emit(
                "detector_skipped",
                text=(
                    f"Skipping detector '{det_name}': media_type "
                    f"{det_media_type!r} has no direct or converter route from "
                    f"dataset types {sorted(source_types)!r}."
                ),
                detector=det_name,
                detector_media_type=det_media_type,
                dataset_media_type=media_type,
            )
            continue

        # When the detector was trained on a clipper granularity the loaded
        # dataset doesn't already match, re-clip the dataset to that granularity
        # at scoring time (auto-clip + re-embed) instead of skipping. The clipper
        # spec rides on the detector's scoring info and is applied by
        # ``route_and_embed``. A matching (or absent) clipper needs no re-clip.
        # A clipper override takes that spec's place for every detector (#5282).
        det_input_spec = det.get("input_spec") if isinstance(det.get("input_spec"), dict) else None
        target_type = det_media_type or media_type
        if override_clipper is not None:
            reclip_clipper, reclip_params = _override_reclip(
                det_name, target_type, override_clipper, det_input_spec, dataset_spec
            )
        else:
            reclip_clipper, reclip_params = _input_spec_reclip(det_name, det_input_spec, dataset_spec)

        labelset = LabelSet.from_dict(det.get("labelset") or {})
        if not labelset.elements and not detector_text_query(det):
            # A typed-query detector scores as its query's set before any label (#4758).
            raise ValueError(f"Detector '{det_name}' has no labels.")

        det_ctx = DetectorContext(det_name, media_type=det_media_type or media_type)
        # The context is keyed by name, not a registry id, so it is handed the
        # balance the detector keeps (#4665) from the JSON already read.
        seed_detector_beta(det_ctx, det)

        def _haystack_for(
            embedder_name: str,
            _target: str = target_type,
            _clipper: str = reclip_clipper,
            _params: dict[str, Any] = reclip_params,
        ) -> "Haystack | None":
            """The population this detector's threshold is realized on.

            Called once the labels are embedded, so ``embedder_name`` is the
            space they landed in - which is exactly what the routing key needs
            and what does not exist before training starts.  An empty routed
            snapshot (nothing converts, nothing embeds) yields ``None``, which
            leaves the loaded medias as the haystack rather than fitting the
            estimator on nothing.
            """
            if routed is None or not _target:
                return None
            hay_medias, to_source = routed.get(_detector_group_key(_target, embedder_name, _clipper, _params))
            return Haystack(hay_medias, to_source) if hay_medias else None

        trained = train_from_labelset(
            det_ctx,
            labelset,
            media_type=target_type,
            snap=snap,
            haystack_for=_haystack_for,
            # Until its labels give a trained head, a typed-query detector is its query's set (#4758).
            text_query=detector_text_query(det),
        )
        if not trained:
            cached = len(det_ctx.label_embeddings)
            total = len(labelset.elements)
            raise ValueError(
                f"Detector '{det_name}': could not build a detector "
                f"(resolved {cached} of {total} label origins, need ≥1 good). "
                "The original media may not be reachable from the CLI - for example, "
                "labels collected through the local_folder importer have no resolve_file() path."
            )
        # Alongside the scoring artifacts (mlp/threshold), carry the metadata a
        # portable-detector export needs so the exporter never re-reads the
        # detector file: the concrete embedder space it trained in, its locked
        # type, and the labelset good/bad tallies.
        from vtscore.detectors.embedder_type import detector_embedder_type_from_data  # noqa: PLC0415

        _report_centroid(det_name, det_ctx, labelset)
        # The balance's state on that threshold - unchecked, headless (#4272,
        # #4413); it rides into every result the detector produces.
        balance = _record_line_state(det_name, det_ctx)
        out[det_name] = {
            "mlp": det_ctx.model,
            "threshold": det_ctx.threshold,
            "balance": balance,
            # The space the head scores in: a typed query's head's is the text slot (#4758).
            "embedder": head_score_embedder(det_ctx.model, det_ctx.embedder or "") or "",
            "media_type": det_media_type or media_type,
            "embedder_type": detector_embedder_type_from_data(det),
            "good_count": sum(1 for el in labelset.elements if el.label == "good"),
            "bad_count": sum(1 for el in labelset.elements if el.label == "bad"),
            # Clipper to re-apply at scoring time (empty when the dataset already
            # matches the detector's granularity, or the detector has no clipper).
            "clipper": reclip_clipper,
            "clipper_params": reclip_params,
        }
    return out


def _input_spec_reclip(
    det_name: str,
    det_input_spec: dict[str, Any] | None,
    dataset_spec: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """The re-clip *det_name*'s own ``input_spec`` asks for: ``(clipper, params)``, or ``("", {})`` for none.

    A detector with no ``input_spec`` clipper, or one the loaded dataset is
    already clipped to match, needs none.
    """
    from vtscore.detectors.input_spec import clipper_matches  # noqa: PLC0415

    if not det_input_spec or clipper_matches(det_input_spec, dataset_spec):
        return "", {}
    reclip_clipper = det_input_spec.get("clipper") or ""
    reclip_params = dict(det_input_spec.get("clipper_params") or {})
    if reclip_clipper:
        dataset_clipper = (dataset_spec or {}).get("clipper") or "(none)"
        cli_progress.emit(
            "detector_reclip",
            text=(
                f"Re-clipping dataset for detector '{det_name}' with "
                f"clipper {reclip_clipper!r} to match its input_spec "
                f"(dataset clipper: {dataset_clipper!r})."
            ),
            detector=det_name,
            detector_input_spec=det_input_spec,
            dataset_input_spec=dataset_spec or {},
        )
    return reclip_clipper, reclip_params


def _override_reclip(
    det_name: str,
    target_type: str,
    override_clipper: dict[str, Any],
    det_input_spec: dict[str, Any] | None,
    dataset_spec: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """The re-clip a clipper override asks of *det_name*: ``(clipper, params)``, or ``("", {})`` for none.

    The override takes the place of the detector's own ``input_spec`` (#5282),
    so it re-clips on the same terms: not when the dataset is already clipped
    to match it, and never for a ``*_default`` clipper, which leaves each media
    as it was loaded.  A clipper for a media type other than the one the
    detector scores (its own, or the dataset's for a legacy detector with
    none) cannot apply, and raises :class:`ValueError` naming both types.
    """
    from vtscore.detectors.input_spec import clipper_matches  # noqa: PLC0415
    from vtscore.media import clippers_for_type, get_clipper  # noqa: PLC0415

    name = override_clipper["clipper"]
    clipper_type = get_clipper(name).media_type
    if clipper_type != target_type:
        fitting = ", ".join(sorted(c.name for c in clippers_for_type(target_type))) or "none"
        hint = f" Clippers for {target_type}: {fitting}." if target_type else ""
        raise ValueError(
            f"Clipper override {name!r} clips {clipper_type} media, but detector {det_name!r} scores "
            f"{target_type or 'untyped'} media.{hint}"
        )
    if name.endswith("_default") or clipper_matches(override_clipper, dataset_spec):
        return "", {}
    cli_progress.emit(
        "detector_reclip",
        text=(
            f"Re-clipping dataset for detector '{det_name}' with clipper "
            f"{_describe_clipper_spec(override_clipper)} from the clipper override "
            f"(its input_spec: {_describe_clipper_spec(det_input_spec)}; "
            f"dataset clipper: {_describe_clipper_spec(dataset_spec)})."
        ),
        detector=det_name,
        detector_input_spec=det_input_spec or {},
        dataset_input_spec=dataset_spec or {},
        override_input_spec=override_clipper,
    )
    return name, dict(override_clipper.get("clipper_params") or {})


def _report_centroid(det_name: str, det_ctx: Any, labelset: Any) -> None:
    """Say so when *det_name*'s labels are under the quota and it scores as the Goods' centroid (#4643).

    The run still scores and exports: the centroid is the detector those labels
    give, here as in Test and AutoFind.  The ``detector_centroid`` event is the
    record that it was not a trained head, and what the labelset still owes.  A
    detector made from a typed query scores as that query's own set there
    instead (#4758), and says so in a ``detector_query`` event; past the quota
    it does until the hold's labels (#4736), and the event then says that is
    why (#4889).
    """
    from vtscore.detectors.centroid_head import is_centroid_head  # noqa: PLC0415
    from vtscore.detectors.label_quota import labelset_quota, served_quota  # noqa: PLC0415
    from vtscore.detectors.query_head import is_query_head, query_head_text  # noqa: PLC0415

    query = is_query_head(det_ctx.model)
    if not (query or is_centroid_head(det_ctx.model)):
        return
    quota = labelset_quota(labelset)
    owed = [
        f"{n} more {kind}{'' if n == 1 else 's'}"
        for n, kind in ((quota.goods_owed, "Good"), (quota.bads_owed, "Bad"))
        if n
    ]
    # The second quota (#4731): with a Good in hand, Bads alone get there too.
    dry_owed = max(0, quota.dry_bad_quota - quota.n_bad) if owed and quota.n_good and quota.dry_bad_quota else 0
    if dry_owed:
        owed = [" and ".join(owed) + f", or {dry_owed} more Bad{'' if dry_owed == 1 else 's'}"]
    serves = f"its typed query's own set ('{query_head_text(det_ctx.model)}')" if query else "the Goods' centroid"
    served = served_quota(det_ctx.model, labelset)
    if served["hold_owed"]:
        # Past the quota, held on the typed query (#4736): the quota is not why (#4889).
        hold_owed = served["hold_owed"]
        text = (
            f"Detector '{det_name}' has {quota.n_good} Good and {quota.n_bad} Bad labels; a detector made from a "
            f"typed query scores as {serves} until it has {served['hold_labels']} labels "
            f"({hold_owed} more label{'' if hold_owed == 1 else 's'}, Good or Bad, for a trained detector)."
        )
    else:
        text = (
            f"Detector '{det_name}' has {quota.n_good} Good and {quota.n_bad} Bad labels, under the quota, so it "
            f"scores as {serves}, not a trained detector" + (f" ({' and '.join(owed)} for one)." if owed else ".")
        )
    cli_progress.emit(
        "detector_query" if query else "detector_centroid",
        text=text,
        detector=det_name,
        **({"query": query_head_text(det_ctx.model)} if query else {}),
        **served,
    )


def _record_line_state(det_name: str, det_ctx: Any) -> dict[str, Any] | None:
    """What the balance says about *det_name*'s trained cut: unchecked, because nobody can vote.

    Read at the balance the training read: the detector's own
    (:func:`vtscore.state.detector_beta`, #4413, #4665).  A headless run cannot
    spot-check its line (#4272), so it exports what the line keeps unchecked -
    every unvoted item at or above the threshold the labels drew (#4452), the
    state's ``count`` - and the ``detector_unchecked`` event is the run's
    record that the set it exports was never checked.  ``None`` with no
    balance (a library caller's ``CoreConfig(beta=None)``): the line is the
    Inclusion 0 cut, and there is no set to report.
    """
    from vtscore.state import detector_beta  # noqa: PLC0415
    from vtscore.state.core import detector_balance_state  # noqa: PLC0415
    from vtscore.training.thresholds import BALANCE_UNCHECKED, aim_words  # noqa: PLC0415

    balance = detector_balance_state(det_ctx, detector_beta(det_ctx))
    if balance is not None and balance["status"] == BALANCE_UNCHECKED:
        cli_progress.emit(
            "detector_unchecked",
            text=(
                f"Detector '{det_name}' exports its top {balance['count']} unchecked ({aim_words(balance)}); "
                "nobody is here to check it."
            ),
            detector=det_name,
            beta=balance["beta"],
            status=balance["status"],
            count=balance["count"],
        )
    return balance


class _NoApplicableDetectorsError(ValueError):
    """No AutoFind (or override) detector applies to the loaded media.

    A ``ValueError`` so every caller that already reports the message keeps
    doing so; the subclass only exists so a saving run (``save_dataset``) can
    tell "nothing to detect with" apart from a real failure and finish with
    the dataset saved instead of exiting non-zero.
    """


def _train_detectors_for_first_chunk(
    chunk_medias: dict[int, dict[str, Any]],
    media_type: str,
    override_detectors: list[str] | None,
    autofind_detectors: list[str],
    override_clipper: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], _RoutedSnapshots]:
    """Train each AutoFind (or override) detector once against the first chunk.

    Returns the trained detectors alongside the routed snapshots their
    thresholds were calibrated on, so the caller can score this same chunk
    against the identical population instead of preparing it a second time
    (issue #3647).

    Raises :class:`ValueError` when no detector applies to *media_type* -
    that's almost always a settings-file misconfiguration the caller wants
    surfaced immediately.
    """
    detector_names = list(override_detectors) if override_detectors is not None else list(autofind_detectors)
    routed = _RoutedSnapshots(chunk_medias)
    detector_mlps: dict[str, dict[str, Any]] = (
        _load_and_train_detectors(detector_names, media_type, chunk_medias, routed, override_clipper)
        if detector_names
        else {}
    )
    if not detector_mlps:
        if override_detectors is not None:
            requested = ", ".join(repr(n) for n in detector_names) or "none"
            raise _NoApplicableDetectorsError(
                f"None of the requested detectors ({requested}) applies to media type: {media_type}."
            )
        raise _NoApplicableDetectorsError(
            f"No AutoFind detectors found for media type: {media_type}. "
            "Add detectors to the settings file's autofind_detectors list."
        )
    return detector_mlps, routed
