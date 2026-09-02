"""The pieces ``build_pile.py`` is assembled from.

``build_pile.py`` is the CLI and the per-cell build loop; everything it does
*inside* a build lives here, split by the question each part answers:

* :mod:`pilebuild.env` -- which checkout and which harness this run resolves to.
* :mod:`pilebuild.vgsource` / :mod:`pilebuild.boxscan` -- reading the Visual
  Genome source and choosing a band's categories from the box scan.
* :mod:`pilebuild.corrections` -- human verdicts, and the one place their boxes
  cross from normalised into pixel space (#3281).
* :mod:`pilebuild.geometry` -- checks a region box must pass whoever wrote it.
* :mod:`pilebuild.loaders` -- one module per ``DATASETS[ds]["kind"]``, each
  owning both *how the cell is built* and *what a rebuild of it needs*.
* :mod:`pilebuild.provenance` -- what machine produced a cell, and its hash.
* :mod:`pilebuild.audit` / :mod:`pilebuild.manifest` -- the read-only modes.

Nothing here imports ``build_pile``; the dependency runs one way, so a loader
can be exercised without paying for the CLI's import-time environment setup.
"""
