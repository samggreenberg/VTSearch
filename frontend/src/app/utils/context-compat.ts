import { DatasetRegistryEntry, DetectorRegistryEntry } from '../models/api.models';

/**
 * Whether the given dataset/detector pair can be used together.
 *
 * Datasets and detectors each have a single `media_type`; a pair is
 * compatible when both halves are present and their media types match.
 *
 * Centralised so future generalisations (multi-media-type detectors,
 * embedder-family checks) have a single edit point. See
 * `docs/plans/active-context-switcher.md` § Compatibility predicate.
 */
export function isPairCompatible(
  dataset: DatasetRegistryEntry | null | undefined,
  detector: DetectorRegistryEntry | null | undefined,
): boolean {
  if (!dataset || !detector) return false;
  return dataset.media_type === detector.media_type;
}
