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
  return `${text} ${unit} left?`;
}

/**
 * Format a remaining-seconds estimate into a deliberately humble, single-unit
 * chip: ``35 sec left`` / ``5.5 min left`` / ``2 hr left``. The value is
 * rounded to a nice step that scales with its magnitude (see
 * ``ETA_ROUND_RATIO``), so we never claim more precision than a noisy estimate
 * deserves and never show an over-precise ``5 min 34 sec``-style breakdown.
 * Returns an empty string for ``null``, non-positive, or non-finite values so
 * the caller can drop it from concatenation unconditionally.
 */
export function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds) || seconds <= 0) return '';
  // Sub-minute: round to a nice multiple of seconds. We are deliberately
  // imprecise about ETAs, so the granularity floor is 10s — never finer.
  // Showing "< 5 sec" for what can be 10+ seconds of work reads as
  // over-confident; "< 10 sec" is the honest floor.
  if (seconds < 60) {
    const s = snapEta(seconds, 10);
    // A few seconds left snaps down to 0; claiming "~0 sec" reads as "done"
    // even though work remains. Show "< 10 sec left" instead.
    if (s <= 0) return '< 10 sec left?';
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
 * ``[Step S/T] (C/T) message · ETA left`` string used by every progress consumer.
 *
 * Each piece is optional:
 *   - The ``[Step S/T]`` prefix appears only when ``total_steps > 1``.
 *   - The ``(C/T)`` fraction appears only when ``total > 0``.
 *   - The ``· 5.5 min left`` tail appears only when ``eta_seconds > 0``; the
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

/** Resolved inputs for a `<vt-progress-bar>` derived from a progress event. */
export interface ProgressBarState {
  value: number;
  max: number;
  indeterminate: boolean;
  /**
   * True when a determinate whole-job bar is parked on an *indeterminate
   * phase*: the job reports an ``overall`` fraction, but the current phase has
   * no ``total`` to count against (e.g. the model load reports ``0/0`` for its
   * whole duration — issue #2621). The bar should keep its parked fill but
   * animate in place (a shimmer) so the phase reads as alive rather than
   * frozen; the fill value itself must not move, because there is genuinely
   * no progress signal to move it by.
   */
  pulsing?: boolean;
}

/**
 * Resolve the value/max/indeterminate a `<vt-progress-bar>` should render for a
 * progress event, preferring the whole-job ``overall`` fraction so the bar
 * fills once across a multi-step job (download → load → embed → finalize)
 * instead of resetting at each phase.
 *
 * Precedence:
 *   1. ``overall`` (0..1) when the backend reports a multi-step structure →
 *      one continuous bar for the entire job.
 *   2. ``current``/``total`` when a single-phase total is known.
 *   3. Indeterminate spinner when neither is available.
 */
export function progressBarState(
  progress: ProgressEvent | null | undefined,
): ProgressBarState {
  const prog = progress ?? {};
  const overall = prog.overall;
  if (overall != null && isFinite(overall)) {
    const value = Math.min(1, Math.max(0, overall));
    // Mid-job with no within-phase total to count against (and no error): the
    // current phase is indeterminate, so the parked fill should pulse in place.
    const pulsing = value < 1 && !(prog.total != null && prog.total > 0) && !prog.error;
    return { value, max: 1, indeterminate: false, pulsing };
  }
  const current = prog.current;
  const total = prog.total;
  if (current != null && total != null && total > 0) {
    return { value: current, max: total, indeterminate: false };
  }
  return { value: 0, max: 1, indeterminate: true };
}

/**
 * Return ``true`` when a progress event should render as an indeterminate
 * spinner rather than a percent bar — i.e. neither a whole-job ``overall``
 * fraction nor a single-phase ``current``/``total`` is available.
 */
export function isProgressIndeterminate(
  progress: ProgressEvent | null | undefined,
): boolean {
  return progressBarState(progress).indeterminate;
}

/**
 * A three-tier breakdown of a progress event for header-style UIs. The bare
 * ``[Step 3/4] Loading embedding model…`` is uninformative to users who have
 * no mental model of the load steps. ``formatProgressHeader`` instead returns:
 *
 *   - ``header``: ``"<What> · <Phase>"``, e.g. ``"Loading dataset · Embedding model"``.
 *     Each ``·``-separated segment is capitalized so the line reads like a row
 *     of labels ("Loading dataset · Step 3 of 4 · Embedding files") rather than
 *     a run-on sentence.
 *   - ``subtitle``: a plain-English one-liner explaining what the phase actually does.
 *   - ``detail``: the per-item line — ``current/total`` counts (no parentheses)
 *     followed by the item identifier, e.g. ``"012/345 cats/img.png"``. The
 *     leading action verb is stripped because the header already names the phase
 *     ("· Embedding files"); repeating "Embedding" in the narrow, ellipsized
 *     detail slot would just eat the characters the filename needs. The ETA tail
 *     is omitted here; it is returned separately as ``eta`` so the UI can pin it
 *     to the right of the progress bar where it stays visible even when a long
 *     file path ellipsizes the detail.
 *   - ``eta``: the bare ``5.5 min left`` chip, or empty when no estimate is available.
 */
export interface ProgressHeader {
  header: string;
  subtitle: string;
  detail: string;
  eta: string;
}

/** Which load flow this progress event belongs to. */
export type ProgressKind = 'dataset' | 'detector' | 'projection';

/**
 * Strip a leading action verb (a gerund like "Embedding"/"Converting"/"Loading",
 * optionally "Re-…") from a per-item progress message. The structured header
 * already names the phase ("· Embedding files"), so the verb is redundant in the
 * narrow detail slot — dropping it leaves the whole width for the filename, which
 * is the only part that actually varies per item. ``"Embedding cats/img.png"`` →
 * ``"cats/img.png"``. A message with no leading gerund is returned unchanged.
 */
function stripActionVerb(msg: string): string {
  return msg.replace(/^(?:re-?)?[a-z]+ing\s+/i, '');
}

/** Capitalize the first character of a string (leaving the rest untouched). */
function capitalize(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
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
        ? 'Building the map'
        : 'Loading dataset';

  let phase = '';
  let subtitle = '';

  if (kind === 'projection') {
    if (/pyramid|tiling|binning/i.test(message)) {
      phase = 'tiling layout';
      subtitle = 'Grouping the items into map tiles.';
    } else {
      phase = 'arranging';
      subtitle = 'Arranging the items so the dataset can be browsed as a map.';
    }
  } else if (status === 'extracting' || (status === 'downloading' && /extract/i.test(message))) {
    phase = 'unpacking archive';
    subtitle = 'Extracting the downloaded archive into the dataset cache.';
  } else if (status === 'downloading') {
    phase = 'downloading source';
    subtitle = 'Fetching the dataset archive. Cached on disk for next time.';
  } else if (status === 'embedding') {
    phase = 'analyzing files';
  } else if (status === 'loading' && /embedding model/i.test(message)) {
    phase = 'loading embedder';
    subtitle = 'Loading the embedder. First-time only; cached on disk afterwards.';
  } else if (status === 'loading' && /text encoder|warming/i.test(message)) {
    phase = 'warming text encoder';
    subtitle = 'One-time warm-up so the first text search returns instantly.';
  } else if (/failed embedding|dropped /i.test(message)) {
    phase = 'cleaning up';
    subtitle = 'Discarding items that could not be embedded.';
  } else if (/duplicates/i.test(message)) {
    phase = 'removing duplicates';
    subtitle = 'Collapsing media that share the same content fingerprint.';
  } else if (/coverage atlas|diversity/i.test(message)) {
    phase = 'building coverage atlas';
    subtitle = 'Indexing for fast diverse browsing and autopilot guidance.';
  } else if (/projection|tile pyramid/i.test(message)) {
    phase = 'building map';
    subtitle = 'Building the Browse map so it opens instantly.';
  } else if (/saving to registry|serial|packaging|registering/i.test(message)) {
    // The whole serialize → zip → write → register window of step 4. These
    // messages used to match nothing, leaving a bare "Step 4 of 4" with no
    // descriptor — the longest part of the load with the least to show for it.
    phase = 'saving dataset';
    subtitle = 'Writing the dataset to disk so it survives a restart.';
  } else if (/clipping/i.test(message)) {
    phase = 'slicing clips';
    subtitle = 'Cutting media into clips for finer-grained search.';
  } else if (/embedding clips/i.test(message)) {
    phase = 'analyzing clips';
    subtitle = 'Analyzing each clip so it can be searched.';
  } else if (/converting/i.test(message)) {
    phase = 'converting media';
    subtitle = 'Running the converter on each input file.';
  } else if (kind === 'detector' && /missing media|re-?ingesting/i.test(message)) {
    phase = 'fetching media';
    subtitle = "Pulling the imported labels' media in from their original sources.";
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
    phase = 'analyzing labels';
    subtitle = 'Analyzing your voted items with the embedder the labels use, so the detector trains on a consistent set.';
  } else if (status === 'loading' && /preparing|scanning|importing/i.test(message)) {
    phase = 'preparing';
    subtitle = /scanning/i.test(message)
      ? 'Walking the source folder to enumerate media files.'
      : 'Setting up the load pipeline.';
  }

  // Surface the step count in the header ("Step 3 of 4") so the user always
  // knows how many phases the whole job has — per the consolidation brief, the
  // job count matters more than any single phase's fine detail. Each segment is
  // capitalized so the line reads as a row of labels, not a sentence.
  const step = prog.step;
  const totalSteps = prog.total_steps;
  const stepPart =
    step != null && totalSteps != null && totalSteps > 1 ? `Step ${step} of ${totalSteps}` : '';
  const header = [what, stepPart, capitalize(phase)].filter(Boolean).join(' · ');

  // Detail = bare "current/total" (no parentheses — characters are precious in
  // the narrow, ellipsized slot) + the item identifier with its redundant
  // leading verb stripped (the header already names the phase).
  const current = prog.current;
  const total = prog.total;
  const counts =
    current != null && total != null && total > 0 ? formatProgressFraction(current, total) : '';
  const item = stripActionVerb(message);
  const detail = [counts, item].filter(Boolean).join(' ');
  const eta = formatEta(prog.eta_seconds);
  return { header, subtitle, detail, eta };
}
