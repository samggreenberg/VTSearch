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

/** Render a magnitude with at most one decimal and no trailing ``.0``. */
function etaUnit(value: number, unit: string): string {
  const text = Number.isInteger(value) ? String(value) : value.toFixed(1);
  return `About ${text} ${unit} left`;
}

/**
 * Format a remaining-seconds estimate into a deliberately humble, single-unit
 * chip: ``About 45 sec left`` / ``About 10 min left`` / ``About 2 hr left``.
 *
 * The vagueness is not this function's doing — it is the backend's. Every
 * ``eta_seconds`` on the wire has already been snapped to a coarse ladder and
 * held there by hysteresis (see ``ProgressTracker._humble_eta``), so a job whose
 * true estimate wanders between 8 and 11 minutes reports a steady ``600`` the
 * whole time instead of walking the user through every revision. That is
 * deliberate: an estimate reported to the half-minute invites you to check it,
 * and checking it is how you notice it going *up*.
 *
 * All this does is pick the unit and say "About". Ladder rungs are chosen to be
 * round in whichever unit they land in, so nothing is rounded a second time
 * here; a value that somehow arrives off-ladder still renders sanely at one
 * decimal. Returns an empty string for ``null``, non-positive, or non-finite
 * values so the caller can drop it from concatenation unconditionally.
 */
export function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds) || seconds <= 0) return '';
  if (seconds < 60) return etaUnit(Math.round(seconds), 'sec');
  const minutes = seconds / 60;
  if (minutes < 60) return etaUnit(Math.round(minutes * 10) / 10, 'min');
  return etaUnit(Math.round((seconds / 3600) * 10) / 10, 'hr');
}

/**
 * Format a `ProgressEvent` into the canonical
 * ``[Step S/T] (C/T) message · ETA left`` string used by every progress consumer.
 *
 * Each piece is optional:
 *   - The ``[Step S/T]`` prefix appears only when ``total_steps > 1``.
 *   - The ``(C/T)`` fraction appears only when ``total > 0``.
 *   - The ``· About 10 min left`` tail appears only when ``eta_seconds > 0``; the
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
  /**
   * Upper bound of the pulsing zone, on the same scale as `value` (0..1 for
   * `overall` bars): the whole-job fraction at which the current count-less
   * phase's slice ends. When present, the bar sweeps the `value`..`pulseTo`
   * span with the same block the whole-bar spinner uses — "the job is somewhere
   * in here" — instead of shimmering the parked fill. Absent when the backend
   * doesn't report the slice end, the phase is determinate, or the zone covers
   * the whole bar (which collapses to a plain `indeterminate` bar).
   */
  pulseTo?: number;
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
    // When the backend also reports where the count-less phase's slice ends,
    // bound the pulse: the bar sweeps value..pulseTo as the unknown zone rather
    // than shimmering the parked fill alone.
    const stepEnd = prog.overall_step_end;
    if (pulsing && stepEnd != null && isFinite(stepEnd) && stepEnd > value) {
      // A zone covering the *whole* bar says exactly what a plain indeterminate
      // bar says — nothing is known anywhere — so it renders as one, rather than
      // as a band that happens to span everything. Without this, a job that
      // declares a single count-less step (`step 1 of 1`, e.g. "Building
      // coverage atlas…") would animate differently from an identical job that
      // declares no step structure at all.
      if (value <= 0 && stepEnd >= 1) {
        return { value: 0, max: 1, indeterminate: true };
      }
      return { value, max: 1, indeterminate: false, pulsing, pulseTo: Math.min(1, stepEnd) };
    }
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
 *   - ``eta``: the bare ``About 10 min left`` chip, or empty when no estimate is available.
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
