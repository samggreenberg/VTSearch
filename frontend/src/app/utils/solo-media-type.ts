/**
 * Whether a field naming `mediaType` would tell the reader nothing (#5106):
 * the server is locked to one media type (`solo_media_type`) and this is it,
 * so it is the only value the field could hold. Surfaces that show a media
 * type as a field of its own (Find Results, the metadata panes, the stats
 * dialogs, the top-bar pulldowns) leave it out then. A type the lock does not
 * match, such as a dataset kept from before the lock was set, is news, so it
 * still shows.
 */
export function isSoloMediaType(
  solo: string | null | undefined,
  mediaType: string | null | undefined,
): boolean {
  return !!solo && mediaType === solo;
}

/**
 * {@link isSoloMediaType} for a column: true when every row is of the server's
 * one media type, so a Type column would repeat it on each of them. One row of
 * another type brings the column back for all.
 */
export function allSoloMediaType(
  solo: string | null | undefined,
  mediaTypes: Iterable<string | null | undefined>,
): boolean {
  if (!solo) return false;
  for (const mediaType of mediaTypes) {
    if (mediaType !== solo) return false;
  }
  return true;
}
