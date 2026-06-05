import { ProgressEvent } from '../models/api.models';

/**
 * Format a current/total progress fraction for display.
 *
 * When both values are large enough to be byte counts (>= 1 MB),
 * renders them in compact human-readable units (e.g. "497/1.10GB").
 * Otherwise returns a plain "current/total" string with comma
 * separators for readability.
 */
export function formatProgressFraction(current: number, total: number): string {
  const GB = 1_073_741_824;
  const MB = 1_048_576;
  if (total >= GB) {
    return `${(current / GB).toFixed(2)}/${(total / GB).toFixed(2)}GB`;
  }
  if (total >= MB) {
    return `${Math.round(current / MB)}/${Math.round(total / MB)}MB`;
  }
  return `${current.toLocaleString()}/${total.toLocaleString()}`;
}

/**
 * The fraction of an estimate we allow ourselves to claim as precision. An ETA
 * is snapped to a "nice" step roughly this size relative to its own magnitude,
 * so the bigger the estimate the coarser it reads: a ~5.5 min job rounds to the
 * half-minute, a ~2 hr job to the half-hour. Keeping every chip deliberately
 * approximate is the point — an under-specified ETA can't be held to a
 * precision the backend never actually had.
 */
const ETA_ROUND_RATIO = 0.1;

/** Nice rounding steps, expressed in the chosen display unit (sec/min/hr). */
const ETA_NICE_STEPS = [0.5, 1, 2, 5, 10, 15, 30];

/**
 * Snap ``value`` to the nearest "nice" step whose size is about
 * ``ETA_ROUND_RATIO`` of the value itself (never smaller than ``minStep``).
 */
function snapEta(value: number, minStep: number): number {
  const target = value * ETA_ROUND_RATIO;
  let step = minStep;
  for (const candidate of ETA_NICE_STEPS) {
    if (candidate >= minStep && candidate <= target) step = candidate;
  }
  return Math.round(value / step) * step;
}

/** Render a snapped magnitude with at most one decimal and no trailing ``.0``. */
function etaUnit(value: number, unit: string): string {
  const text = Number.isInteger(value) ? String(value) : value.toFixed(1);
  return `~${text} ${unit} left`;
}

/**
 * Format a remaining-seconds estimate into a deliberately humble, single-unit
 * chip: ``~35 sec left`` / ``~5.5 min left`` / ``~2 hr left``. The value is
 * rounded to a nice step that scales with its magnitude (see
 * ``ETA_ROUND_RATIO``), so we never claim more precision than a noisy estimate
 * deserves and never show an over-precise ``5 min 34 sec``-style breakdown.
 * Returns an empty string for ``null``, non-positive, or non-finite values so
 * the caller can drop it from concatenation unconditionally.
 */
export function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds) || seconds <= 0) return '';
  // Sub-minute: round to a nice multiple of seconds (minimum 5s granularity).
  if (seconds < 60) {
    const s = snapEta(seconds, 5);
    if (s < 60) return etaUnit(s, 'sec');
  }
  // Minutes: round to the nice fraction of a minute (minimum half-minute).
  const minutes = seconds / 60;
  if (minutes < 60) {
    const m = snapEta(minutes, 0.5);
    if (m < 60) return etaUnit(m, 'min');
  }
  // Hours: round to the nice fraction of an hour (minimum half-hour).
  const h = snapEta(seconds / 3600, 0.5);
  return etaUnit(h, 'hr');
}

/**
 * Format a `ProgressEvent` into the canonical
 * ``[Step S/T] (C/T) message · ~ETA left`` string used by every progress consumer.
 *
 * Each piece is optional:
 *   - The ``[Step S/T]`` prefix appears only when ``total_steps > 1``.
 *   - The ``(C/T)`` fraction appears only when ``total > 0``.
 *   - The ``· ~5.5 min left`` tail appears only when ``eta_seconds > 0``; the
 *     backend gates this on at least 5s of elapsed work, so it stays hidden
 *     for short bars.
 *   - When none are present, returns the bare ``message`` (or
 *     ``defaultMessage`` if ``message`` is empty).
 */
export function formatProgressMessage(
  progress: ProgressEvent | null | undefined,
  defaultMessage = '',
  options: { includeEta?: boolean } = {},
): string {
  const { includeEta = true } = options;
  const prog = progress ?? {};
  let msg = prog.message || defaultMessage;
  const step = prog.step;
  const totalSteps = prog.total_steps;
  if (step != null && totalSteps != null && totalSteps > 1) {
    msg = `[Step ${step}/${totalSteps}] ${msg}`;
  }
  const current = prog.current;
  const total = prog.total;
  if (current != null && total != null && total > 0) {
    const fraction = `(${formatProgressFraction(current, total)})`;
    const stepEnd = msg.indexOf('] ');
    if (stepEnd !== -1) {
      msg = msg.slice(0, stepEnd + 2) + fraction + ' ' + msg.slice(stepEnd + 2);
    } else {
      msg = msg ? `${fraction} ${msg}` : fraction;
    }
  }
  if (includeEta) {
    const eta = formatEta(prog.eta_seconds);
    if (eta) {
      msg = msg ? `${msg} · ${eta}` : eta;
    }
  }
  return msg;
}

/**
 * Return ``true`` when a progress event has no known total; callers
 * should show an indeterminate spinner rather than a percent bar.
 */
export function isProgressIndeterminate(
  progress: ProgressEvent | null | undefined,
): boolean {
  const prog = progress ?? {};
  return !(prog.current != null && prog.total != null && prog.total > 0);
}

/**
 * A three-tier breakdown of a progress event for header-style UIs. The bare
 * ``[Step 3/4] Loading embedding model…`` is uninformative to users who have
 * no mental model of the load steps. ``formatProgressHeader`` instead returns:
 *
 *   - ``header``: ``"<what> · <phase>"``, e.g. ``"Loading dataset · embedding model"``.
 *   - ``subtitle``: a plain-English one-liner explaining what the phase actually does.
 *   - ``detail``: the original message + ``(current/total)`` counts, with the
 *     redundant ``[Step S/T]`` prefix stripped (the header conveys the phase).
 *     The ETA tail is omitted here; it is returned separately as ``eta`` so the
 *     UI can pin it to the right of the progress bar where it stays visible
 *     even when a long file path ellipsizes the detail.
 *   - ``eta``: the bare ``~5.5 min left`` chip, or empty when no estimate is available.
 */
export interface ProgressHeader {
  header: string;
  subtitle: string;
  detail: string;
  eta: string;
}

/** Which load flow this progress event belongs to. */
export type ProgressKind = 'dataset' | 'detector' | 'projection';

const EMBEDDER_PRETTY: Record<string, string> = {
  siglip: 'SigLIP',
  siglip2: 'SigLIP 2',
  clip: 'CLIP',
  dinov2: 'DINOv2',
  dinov2_patch: 'DINOv2',
  dinov3: 'DINOv3',
  dinov3_patch: 'DINOv3',
  dinov3_single: 'DINOv3',
  clap: 'LAION-CLAP',
  clap_general: 'LAION-CLAP (general)',
  clap_music: 'LAION-CLAP (music)',
  xclip: 'X-CLIP',
  'x-clip': 'X-CLIP',
  whisper: 'Whisper',
  whisper_encoder: 'Whisper',
  paraspeechclap: 'ParaSpeechCLAP',
  ast: 'AST',
  e5: 'E5',
  bge: 'BGE',
  languagebind: 'LanguageBind',
};

function prettifyEmbedder(name: string | undefined): string {
  if (!name) return '';
  const key = name.toLowerCase().replace(/-/g, '_');
  return EMBEDDER_PRETTY[key] ?? name;
}

function stripStepPrefix(msg: string): string {
  return msg.replace(/^\[Step \d+\/\d+\]\s*/, '');
}

/**
 * Resolve a ``ProgressEvent`` to a header / subtitle / detail triple for UIs
 * that want richer loading-state context than a single line of text. See the
 * ``ProgressHeader`` interface for the meaning of each field. ``kind`` selects
 * between the dataset-load and detector-load phase vocabularies; ``embedder``
 * is woven into the subtitle when the phase mentions one ("Loading SigLIP
 * weights. First-time only, cached on disk afterwards.").
 */
export function formatProgressHeader(
  progress: ProgressEvent | null | undefined,
  kind: ProgressKind,
  embedder?: string,
): ProgressHeader {
  const prog = progress ?? {};
  const status = (prog.status ?? '').toLowerCase();
  const message = prog.message ?? '';
  const what =
    kind === 'detector'
      ? 'Loading detector'
      : kind === 'projection'
        ? 'Building projection'
        : 'Loading dataset';

  let phase = '';
  let subtitle = '';

  if (kind === 'projection') {
    if (/pyramid|tiling|binning/i.test(message)) {
      phase = 'tiling layout';
      subtitle = 'Binning the 2-D layout into the hex/square tile pyramid.';
    } else {
      phase = 'running UMAP';
      subtitle = 'Projecting embeddings to 2-D so the dataset can be browsed as a map.';
    }
  } else if (status === 'downloading') {
    if (/extract/i.test(message)) {
      phase = 'unpacking archive';
      subtitle = 'Extracting the downloaded archive into the dataset cache.';
    } else {
      phase = 'downloading source';
      subtitle = 'Fetching the dataset archive. Cached on disk for next time.';
    }
  } else if (status === 'embedding') {
    phase = 'embedding files';
  } else if (status === 'loading' && /embedding model/i.test(message)) {
    phase = 'embedding model';
    const pretty = prettifyEmbedder(embedder);
    subtitle = pretty
      ? `Loading ${pretty} weights. First-time only; cached on disk afterwards.`
      : 'Loading model weights. First-time only; cached on disk afterwards.';
  } else if (status === 'loading' && /text encoder|warming/i.test(message)) {
    phase = 'warming text encoder';
    subtitle = 'One-time warm-up so the first text search returns instantly.';
  } else if (/duplicates/i.test(message)) {
    phase = 'removing duplicates';
    subtitle = 'Collapsing media that share the same content fingerprint.';
  } else if (/diversity/i.test(message)) {
    phase = 'building diversity index';
    subtitle = 'Indexing for fast diverse browsing and autopilot guidance.';
  } else if (/saving to registry/i.test(message)) {
    phase = 'saving to registry';
    subtitle = 'Persisting the dataset so it survives a restart.';
  } else if (/clipping/i.test(message)) {
    phase = 'slicing clips';
    subtitle = 'Cutting media into clips for finer-grained search.';
  } else if (/embedding clips/i.test(message)) {
    phase = 'embedding clips';
    subtitle = 'Computing a vector for each clip.';
  } else if (/converting/i.test(message)) {
    phase = 'converting media';
    subtitle = 'Running the converter on each input file.';
  } else if (kind === 'detector' && /restoring labels/i.test(message)) {
    phase = 'restoring labels';
    subtitle = 'Reading the saved labelset for this detector.';
  } else if (kind === 'detector' && /seeding examples/i.test(message)) {
    phase = 'seeding examples';
    subtitle = 'Pulling label examples back into the active dataset.';
  } else if (
    kind === 'detector' &&
    /embedding labels|re-?resolving labels|re-?embedding/i.test(message)
  ) {
    phase = 'embedding labels';
    const pretty = prettifyEmbedder(embedder);
    subtitle = pretty
      ? `Re-resolving label media into ${pretty} space so MLP training mixes only same-space vectors.`
      : 'Re-resolving label media so MLP training mixes only same-space vectors.';
  } else if (status === 'loading' && /preparing|scanning|importing/i.test(message)) {
    phase = 'preparing';
    subtitle = /scanning/i.test(message)
      ? 'Walking the source folder to enumerate media files.'
      : 'Setting up the load pipeline.';
  }

  const header = phase ? `${what} · ${phase}` : what;
  const detail = stripStepPrefix(formatProgressMessage(progress, '', { includeEta: false }));
  const eta = formatEta(prog.eta_seconds);
  return { header, subtitle, detail, eta };
}
