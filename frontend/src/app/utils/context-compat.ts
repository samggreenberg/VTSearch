import { DatasetRegistryEntry, DetectorRegistryEntry } from '../models/api.models';

/**
 * Whether the given dataset/detector pair can be used together.
 *
 * A pair is compatible when both halves are present, their `media_type`s
 * match, AND the dataset supplies an embedder of the detector's locked
 * `embedder_type` ("semantic" / "patch_semantic" / "structural"). The detector
 * re-derives its MLP against whichever concrete embedder of that type the
 * dataset binds (SigLIP↔CLIP, DinoV2↔DinoV3), so only the *type* must match,
 * not a specific embedder name.
 *
 * The type gate is applied only when both sides advertise the relevant fields;
 * a legacy detector (no `embedder_type`) or a dataset whose `embedder_types`
 * isn't known degrades gracefully to the media-type-only check, so missing
 * metadata never hides a working pair.
 *
 * Centralised so future generalisations (multi-media-type detectors) have a
 * single edit point.
 */
export function isPairCompatible(
  dataset: DatasetRegistryEntry | null | undefined,
  detector: DetectorRegistryEntry | null | undefined,
): boolean {
  if (!dataset || !detector) return false;
  if (dataset.media_type !== detector.media_type) return false;
  const detType = detector.embedder_type;
  const dsTypes = dataset.embedder_types;
  if (detType && dsTypes && dsTypes.length > 0) {
    return dsTypes.includes(detType);
  }
  return true;
}
