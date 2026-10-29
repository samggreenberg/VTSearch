"""Where an autodetect run gets its medias: a dataset pickle or an importer, whole or chunked.

:class:`_SourceSpec` names one source and opens it: :meth:`~_SourceSpec.load`
for the scored stream, :meth:`~_SourceSpec.load_deferred` for a resumed run
that may skip chunks without building them, :meth:`~_SourceSpec.load_outputs`
for a multi-dataset import (#4707).  The loaders behind it renumber chunk ids so
they are unique across the run, honour the importer's ``reference_files``
choice, and run the framework's embed stage over what an importer produced
(:func:`_embed_loaded_medias`), so a dataset is calibrated and scored with every
vector it will ever have.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from vtscore.cli import progress as cli_progress
from vtscore.datasets.chunks import DeferredChunk
from vtscore.datasets.loader import load_dataset_from_pickle

if TYPE_CHECKING:
    from vtscore.datasets.importers.base import OutputSpec


def _list_importer_names() -> list[str]:
    """Return the names of all registered importers."""
    from vtscore.datasets.importers import list_importers

    return [imp.name for imp in list_importers()]


def _load_pickle_whole(dataset_path: str) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield a single medias dict loaded from a pickle file."""
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Thin is safe here: the pickle loader only drops bytes it can re-read
    # (a file on disk, an archive member, a URL) and keeps the payload of a
    # self-contained entry, which nothing outside the pickle could reproduce.
    medias: dict[int, dict[str, Any]] = {}
    load_dataset_from_pickle(dataset_file, medias, thin=True)
    if not medias:
        raise ValueError(f"No medias loaded from dataset: {dataset_path}")
    yield medias


def _renumber_chunks(
    chunks: Iterator[dict[int, dict[str, Any]]],
) -> Iterator[dict[int, dict[str, Any]]]:
    """Re-issue media IDs across a chunk stream so they are globally unique.

    Every chunked importer (and every chunked pickle/folder loader) emits
    chunks whose IDs restart at 1 - the in-process consumer
    :func:`vtscore.datasets.load_pipeline.consume_chunks_into` renumbers
    them as it drains. The CLI pipeline scores each chunk independently
    and merges the per-chunk hit lists, so without renumbering the hits
    in the merged export carry colliding ``id`` values across chunks.
    Wrap the source generator with this helper at the CLI boundary to
    give every media a unique id.
    """
    next_id = 1
    for chunk in chunks:
        yield _renumbered(chunk, next_id)
        next_id += len(chunk)


def _renumbered(chunk: dict[int, dict[str, Any]], first_id: int) -> dict[int, dict[str, Any]]:
    """*chunk*'s medias re-keyed in order from *first_id*, each ``id`` set to its key."""
    renumbered: dict[int, dict[str, Any]] = {}
    for media_id, media in enumerate(chunk.values(), start=first_id):
        media["id"] = media_id
        renumbered[media_id] = media
    return renumbered


def _renumber_deferred_chunks(chunks: Iterator[DeferredChunk], importer_name: str) -> Iterator[DeferredChunk]:
    """:func:`_renumber_chunks` for chunks not built yet: each numbered on from the last, built or not.

    A chunk advances the ids by its declared ``count``, so the chunks after one
    that is never built get the ids an all-built stream gives them.  Building
    a chunk to a different size than it declared is refused, since every id
    after it would then be wrong.
    """

    def build(deferred: DeferredChunk, first_id: int) -> dict[int, dict[str, Any]]:
        chunk = deferred.build()
        if len(chunk) != deferred.count:
            raise ValueError(
                f"Importer '{importer_name}' built {len(chunk)} media for a chunk it declared as {deferred.count}."
            )
        return _renumbered(chunk, first_id)

    next_id = 1
    for deferred in chunks:
        yield DeferredChunk(count=deferred.count, build=functools.partial(build, deferred, next_id), key=deferred.key)
        next_id += deferred.count


def _load_pickle_chunked(dataset_path: str, chunk_size: int) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias loaded from a pickle file."""
    from vtscore.datasets.loader import load_dataset_from_pickle_chunked

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    yield from _renumber_chunks(load_dataset_from_pickle_chunked(dataset_file, chunk_size, thin=True))


def _reference_files_choice(field_values: dict[str, Any]) -> bool:
    """Pop the importer's ``reference_files`` choice and return it as ``thin``.

    The GUI resolves thin mode from the importer's own ``reference_files``
    checkbox (``load_pipeline`` pops it out of the field values and hands it to
    ``run`` as ``thin=``); the field is deliberately not part of the persisted
    origin, so it is a per-load storage choice rather than part of the source's
    identity.  The CLI passes the same field through
    :meth:`~vtscore.plugins.PluginBase.add_cli_arguments`, which turns it into
    ``--reference-files`` / ``--no-reference-files``, but it used to leave the
    value sitting inert in ``field_values`` and force ``thin=True`` regardless.

    Two things went wrong with that.  The flag did nothing, so a CLI user could
    not turn reference mode off; and thin discards ``media_bytes`` in favour of
    a path reference, which strands any media whose bytes cannot be re-read
    from outside the source.  A stranded media cannot be embedded, so it is
    silently skipped at scoring - and because the calibrated threshold is
    fitted on the haystack being scored, the surviving population also moved
    the cut.  Same dataset, same detector, different hits *and* a different
    threshold in the CLI than in the GUI (issue #3556).

    Popping (rather than reading) matches the GUI: ``run`` takes ``thin`` as a
    parameter, not as a field, so the key must not be forwarded into an
    importer's ``field_values``.  Importers that declare no such field get
    ``False`` - non-reference mode, the GUI's default for them too.
    """
    from vtscore.plugins import parse_checkbox  # noqa: PLC0415

    return parse_checkbox(field_values.pop("reference_files", False))


def _embed_loaded_medias(medias: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Run the framework's embed stage over freshly-imported *medias*.

    Importers never call an embedder - that is the contract on
    :class:`~vtscore.datasets.importers.base.core.DataSourceImporter`.  They emit
    media dicts with no vector (``embeddings={}``) and the framework's
    :func:`~vtscore.datasets.stages.embedding.embed_missing` stage embeds
    everything still without a vector once the importer returns.  The GUI's
    ``load_origin`` runs that stage; the CLI ran *none* of the post-import
    stages and let the vectors appear incidentally at scoring time, inside
    ``route_and_embed``'s per-detector-group embed pass.

    That ordering is what issue #3556 cost, because the first thing to read the
    haystack is not scoring but **threshold calibration**: ``train_from_labelset``
    fits the cut on ``scoring_rows_for_snap(snap)``, which drops every media with
    no vector.  An import whose items were still unembedded therefore calibrated
    the detector against a strict subset of the dataset - a lower cut over fewer
    items - and only afterwards did ``route_and_embed`` fill the vectors in.
    Same dataset, same detector, a different threshold and different hits in the
    CLI than in the GUI.

    Media are embedded in groups sharing a ``media_type`` so each group resolves
    its own embedder: a mixed-type import (a folder of videos and PDFs) must not
    have every item pushed through the first item's model, which a single
    whole-dict call would do - ``embed_missing`` resolves one embedder from the
    first media type it finds.

    Nothing is dropped.  The GUI's ``_drop_none_embeddings_stage`` can drop what
    stays at ``None`` because its dataset is scored in the space it was loaded
    in; the CLI's is not.  An item with no native vector - no embedder
    registered for its type, an unreadable file - may still be scoreable through
    the one-hop converter route ``route_and_embed`` applies later, so dropping it
    here would lose hits the CLI finds today.  The count is announced instead, on
    the same event stream as the scoring-time skips.
    """
    from collections import defaultdict  # noqa: PLC0415

    from vtscore.datasets.stages.embedding import _check_named_vector_widths, embed_missing  # noqa: PLC0415
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

    # What the importer filed under an embedder's name must have that
    # embedder's width, as in the GUI's load pipeline.
    _check_named_vector_widths(medias)
    by_type: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for mid, media in medias.items():
        by_type[media.get("media_type") or ""][mid] = media
    for group in by_type.values():
        # Empty name = the framework's own resolution order (the embedder the
        # media already carry, else the media-type default).  The CLI has no
        # embedder pick to forward.  ``--solo-embedder`` plays no part: it
        # preselects the web importer's picker and nothing here reads it.
        embed_missing(group, "", on_progress=cli_progress.progress_callback)

    unembedded = [mid for mid in sorted(medias) if media_embedding(medias[mid]) is None]
    if unembedded:
        cli_progress.emit(
            "medias_unembedded",
            text=(
                f"{len(unembedded)} media carry no embedding after the load embed pass "
                f"and will only score through a converter route: "
                f"{unembedded[:10]}{'…' if len(unembedded) > 10 else ''}"
            ),
            unembedded=len(unembedded),
            unembedded_ids=unembedded[:100],
        )
    return medias


def _load_importer_whole(importer_name: str, field_values: dict[str, Any]) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield a single medias dict loaded via a named importer."""
    for medias in _load_importer_whole_raw(importer_name, field_values):
        yield _embed_loaded_medias(medias)


def _load_importer_whole_raw(importer_name: str, field_values: dict[str, Any]) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield a single medias dict loaded via a named importer, not yet embedded."""
    from vtscore.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")

    importer.validate_cli_field_values(field_values)

    thin = _reference_files_choice(field_values)
    medias: dict[int, dict[str, Any]] = {}
    importer.run_cli(field_values, medias, thin=thin)
    if not medias:
        raise ValueError(f"No medias loaded by importer '{importer_name}'")
    yield medias


def _load_importer_chunked(
    importer_name: str, field_values: dict[str, Any], chunk_size: int
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias loaded via a named importer."""
    for chunk in _load_importer_chunked_raw(importer_name, field_values, chunk_size):
        # Per chunk, not once at the end: the chunked path exists so a dataset
        # too big to hold at once is scored a chunk at a time, and each chunk is
        # calibrated and scored on its own before the next is loaded.
        yield _embed_loaded_medias(chunk)


def _load_importer_chunked_raw(
    importer_name: str, field_values: dict[str, Any], chunk_size: int
) -> Iterator[dict[int, dict[str, Any]]]:
    """Yield chunks of medias loaded via a named importer, not yet embedded."""
    from vtscore.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")

    importer.validate_cli_field_values(field_values)
    thin = _reference_files_choice(field_values)
    yield from _renumber_chunks(importer.run_chunked_cli(field_values, chunk_size, thin=thin))


def _load_importer_chunked_deferred(
    importer_name: str, field_values: dict[str, Any], chunk_size: int
) -> Iterator[DeferredChunk]:
    """Yield chunks of medias via a named importer, not yet built (``run_chunked_cli_deferred``)."""
    from vtscore.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")

    importer.validate_cli_field_values(field_values)
    thin = _reference_files_choice(field_values)
    yield from _renumber_deferred_chunks(
        importer.run_chunked_cli_deferred(field_values, chunk_size, thin=thin), importer_name
    )


def _check_multi_output(importer: Any) -> None:
    """Refuse an ``outputs`` list for an importer that makes one dataset per run.

    The rule the Add Dataset form's *Multi-Dataset* box reads
    (``supports_multi_output`` in the importer's listing): the importer has not
    opted out, and it declares a ``media_type`` field for the outputs to
    multiply.
    """
    from vtscore.datasets.importers.base.core import _supports_multi_output  # noqa: PLC0415

    if not _supports_multi_output(importer):
        raise ValueError(
            f"Importer {importer.name!r} produces one dataset per run; it does not accept an outputs list."
        )


def _output_dataset_names(importer: Any, field_values: dict[str, Any], outputs: "list[OutputSpec]") -> list[str]:
    """The dataset name each entry of *outputs* lands under, in order.

    The same derivation the load pipeline uses
    (:func:`~vtscore.datasets.importers.base.output_dataset_name` over the
    importer's own display name), so a temporary run reports its datasets by
    the names a saving run would have given them.
    """
    from vtscore.datasets.importers.base import output_dataset_name  # noqa: PLC0415

    base_name = importer.resolve_display_name(dict(field_values))
    return [output_dataset_name(base_name, output) for output in outputs]


def _output_index(outputs: "list[OutputSpec]", yielded: "OutputSpec") -> int:
    """Which entry of *outputs* the importer's *yielded* spec is.

    Importers are asked to yield the very objects they were given, so identity
    is the lookup; one that copied them is matched by equality, first match
    wins (as :func:`vtscore.datasets.load_multi._job_for` does).  ``-1`` when
    it is none of them.
    """
    for index, output in enumerate(outputs):
        if output is yielded:
            return index
    for index, output in enumerate(outputs):
        if output == yielded:
            return index
    return -1


def _group_output_chunks(
    stream: "Iterator[tuple[OutputSpec, dict[int, dict[str, Any]]]]",
    outputs: "list[OutputSpec]",
    importer_name: str,
) -> "Iterator[tuple[OutputSpec, Iterator[dict[int, dict[str, Any]]]]]":
    """Regroup an importer's ``(output, chunk)`` stream into one chunk iterator per output.

    The multi-output hooks yield every dataset's chunks through one iterator,
    finishing one output before starting the next (the contract on
    :meth:`~vtscore.datasets.importers.base.core.ImporterBase.run_outputs_chunked`;
    the GUI pipeline tolerates interleaving because it holds every dataset at
    once, which is the one thing a chunked CLI run exists not to do).  Each
    pair yielded here is an output and a lazy iterator over *its* raw chunks,
    in the order the importer produced them; the inner iterator is drained
    (without further work) when the consumer moves on before exhausting it.
    An output the importer never yielded for comes last with an empty
    iterator, so every output is reported.  A chunk for an output already
    finished is an error naming the importer, as is a chunk for an output the
    importer was not asked for.
    """
    it = iter(stream)
    pending = next(it, None)
    finished: set[int] = set()
    while pending is not None:
        index = _output_index(outputs, pending[0])
        if index < 0:
            raise ValueError(
                f"Importer {importer_name!r} yielded media for an output it was not asked for: {pending[0]!r}"
            )
        if index in finished:
            raise ValueError(
                f"Importer {importer_name!r} yielded media for {outputs[index].category_label()!r} after moving on "
                "to another output; a multi-dataset run needs each output's media produced in one go."
            )
        output = outputs[index]

        def chunks(current: int = index) -> Iterator[dict[int, dict[str, Any]]]:
            nonlocal pending
            while pending is not None and _output_index(outputs, pending[0]) == current:
                chunk = pending[1]
                pending = next(it, None)
                if chunk:
                    yield chunk

        yield output, chunks()
        while pending is not None and _output_index(outputs, pending[0]) == index:
            pending = next(it, None)
        finished.add(index)
    for index, output in enumerate(outputs):
        if index not in finished:
            yield output, iter(())


def _load_importer_outputs(
    importer_name: str,
    field_values: dict[str, Any],
    outputs: "list[OutputSpec]",
    chunk_size: int | None,
) -> "Iterator[tuple[OutputSpec, Iterator[dict[int, dict[str, Any]]]]]":
    """Run a named importer once for several datasets; yield ``(output, chunks)`` per dataset.

    The multi-dataset twin of :func:`_load_importer_whole` /
    :func:`_load_importer_chunked`: one call to
    :meth:`~vtscore.datasets.importers.base.core.ImporterBase.run_outputs_chunked`
    (with *chunk_size*) or :meth:`~ImporterBase.run_outputs` (without), so an
    importer that acquires its source once does so once, regrouped by
    :func:`_group_output_chunks`.  Each output's chunks are renumbered from 1
    and embedded as they are pulled, so an output the caller gives up on
    (nothing to detect with) costs no embedding work.

    The shared *field_values* are validated as the single-dataset path
    validates its own (the importer's ``media_type`` field, when it has no
    value, takes its default here and is overridden per output by
    :meth:`~OutputSpec.narrow`); file-typed fields are wrapped for the
    importer as :meth:`~ImporterBase.run_cli` would wrap them.
    """
    from vtscore.datasets.importers import get_importer  # noqa: PLC0415
    from vtscore.plugins.uploads import wrap_cli_file_fields  # noqa: PLC0415

    importer = get_importer(importer_name)
    if importer is None:
        available = _list_importer_names()
        raise ValueError(f"Unknown importer: {importer_name}. Available: {', '.join(available)}")
    _check_multi_output(importer)

    field_values = dict(field_values)
    importer.validate_cli_field_values(field_values)
    thin = _reference_files_choice(field_values)
    field_values = wrap_cli_file_fields(importer.fields, field_values)
    field_values.pop("outputs", None)
    outputs = list(outputs)
    if chunk_size:
        stream = importer.run_outputs_chunked(field_values, outputs, chunk_size, thin=thin)
    else:
        stream = importer.run_outputs(field_values, outputs, thin=thin)
    for output, raw_chunks in _group_output_chunks(stream, outputs, importer_name):
        yield output, (_embed_loaded_medias(chunk) for chunk in _renumber_chunks(raw_chunks))


@dataclass(frozen=True)
class _SourceSpec:
    """Where one autodetect run gets its medias: pickle/importer x whole/chunked.

    The four public ``autodetect_*_main`` entry points used to hand-copy
    three things per cell of that 2x2 - the loader call, the ``--dry-run``
    source description, and the "nothing loaded" message - which is how the
    matrix drifted (the chunked pair grew ``stream_results`` and the whole
    pair did not).  Owning all three here means a new source variant is one
    ``_SourceSpec`` construction, and the three can no longer disagree.
    """

    kind: Literal["pickle", "importer"]
    dataset_path: str = ""
    importer_name: str = ""
    field_values: dict[str, Any] = field(default_factory=dict)
    chunk_size: int | None = None
    #: A multi-dataset import (#4707): the importer runs once and every entry
    #: becomes its own dataset, scored and exported on its own.  Only an
    #: importer source carries one; empty is the ordinary single-dataset run.
    outputs: tuple[OutputSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.outputs and self.kind != "importer":
            raise ValueError("Only an importer source can produce several datasets (outputs).")

    @property
    def empty_error(self) -> str:
        """The error text raised when the source yields no medias at all."""
        if self.kind == "pickle":
            return f"No medias loaded from dataset: {self.dataset_path}"
        return f"No medias loaded by importer '{self.importer_name}'"

    def load(self) -> Iterator[dict[int, dict[str, Any]]]:
        """Open the media source, one dict per chunk (one chunk when whole).

        A multi-dataset source has no single media stream; open it with
        :meth:`load_outputs` instead.
        """
        if self.outputs:
            raise ValueError("A multi-dataset source is opened with load_outputs(), one media stream per dataset.")
        if self.kind == "pickle":
            if self.chunk_size:
                return _load_pickle_chunked(self.dataset_path, self.chunk_size)
            return _load_pickle_whole(self.dataset_path)
        if self.chunk_size:
            return _load_importer_chunked(self.importer_name, self.field_values, self.chunk_size)
        return _load_importer_whole(self.importer_name, self.field_values)

    def load_deferred(self) -> Iterator[DeferredChunk]:
        """Open the media source as :meth:`load` does, but leave each chunk unbuilt and unembedded.

        Each chunk comes out as a :class:`~vtscore.datasets.chunks.DeferredChunk`,
        so a caller decides per chunk whether to build it and then pay for
        :meth:`prepare`; building and preparing every chunk is :meth:`load`.
        A resumed run (``--resume``) passes its completed chunks by this way:
        an importer that walks its chunks without building them
        (``run_chunked_cli_deferred``) never reads their files, and every
        other source's are built but not embedded.
        """
        if self.outputs:
            raise ValueError("A multi-dataset source is opened with load_outputs(), one media stream per dataset.")
        if self.kind == "importer" and self.chunk_size:
            return _load_importer_chunked_deferred(self.importer_name, self.field_values, self.chunk_size)
        if self.kind == "pickle":
            chunks = self.load()
        else:
            chunks = _load_importer_whole_raw(self.importer_name, self.field_values)
        return (DeferredChunk.built(chunk) for chunk in chunks)

    def prepare(self, chunk: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
        """Make one built :meth:`load_deferred` chunk ready to score: embed an importer's media.

        A dataset pickle already carries its vectors, so its chunks pass
        through unchanged.
        """
        if self.kind == "pickle":
            return chunk
        return _embed_loaded_medias(chunk)

    def load_outputs(self) -> Iterator[tuple[OutputSpec, Iterator[dict[int, dict[str, Any]]]]]:
        """Open a multi-dataset source: one ``(output, chunks)`` pair per dataset.

        One importer run feeds every pair (see :func:`_load_importer_outputs`);
        the chunks of each are pulled lazily, so the whole run is never held
        at once.
        """
        if not self.outputs:
            raise ValueError("This source produces one dataset; open it with load().")
        return _load_importer_outputs(self.importer_name, self.field_values, list(self.outputs), self.chunk_size)

    def describe(
        self,
        *,
        stream_results: bool,
        keep_negatives: bool,
        save_dataset: bool = False,
        as_labelset: bool = False,
        resume: bool = False,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """Build the ``source_description`` block reported by ``--dry-run``."""
        common: dict[str, Any] = {
            "kind": self.kind,
            "chunk_size": self.chunk_size,
            "stream_results": stream_results,
            "keep_negatives": keep_negatives,
            "as_labelset": as_labelset,
            "save_dataset": save_dataset,
            "resume": resume,
            "max_results": max_results,
        }
        if self.kind == "pickle":
            return {**common, "dataset": self.dataset_path}
        described = {**common, "importer": self.importer_name, "params": self.field_values}
        if self.outputs:
            described["outputs"] = [output.to_dict() for output in self.outputs]
        return described
