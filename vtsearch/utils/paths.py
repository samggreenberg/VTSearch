"""Server file-path validation utilities.

Prevents path-traversal attacks by ensuring user-supplied file paths
resolve within an allowed base directory.  Used by the web API routes
for server-file importers and exporters.
"""

from __future__ import annotations

from pathlib import Path


def validate_server_filepath(filepath_str: str, base_dir: Path | None = None) -> Path:
    """Validate that *filepath_str* resolves within *base_dir*.

    Parameters
    ----------
    filepath_str:
        User-supplied file path (absolute or relative).
    base_dir:
        The directory the resolved path must reside in.
        Defaults to ``Path.cwd()``.

    Returns
    -------
    Path
        The resolved, canonicalised path.

    Raises
    ------
    ValueError
        If the resolved path is outside *base_dir*.
    """
    if base_dir is None:
        base_dir = Path.cwd()

    path = Path(filepath_str)

    # Resolve relative paths against base_dir, absolute paths as-is
    if not path.is_absolute():
        path = base_dir / path

    resolved = path.resolve()
    base_resolved = base_dir.resolve()

    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise ValueError(
            f"Path must be within '{base_resolved}'. The given path resolves outside the allowed directory."
        )

    return resolved
