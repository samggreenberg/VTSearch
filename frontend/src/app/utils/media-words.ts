/** What a dataset of each media type holds, in words (#4680). */
export const MEDIA_PLURALS: Readonly<Record<string, string>> = {
  image: 'images',
  audio: 'audio clips',
  text: 'text paragraphs',
  video: 'videos',
  document: 'documents',
  face: 'faces',
};

/** One item of each media type, in words (#5175). */
export const MEDIA_ITEMS: Readonly<Record<string, string>> = {
  image: 'image',
  audio: 'audio clip',
  text: 'text paragraph',
  video: 'video',
  document: 'document',
  face: 'face',
};

/** What Toasty calls one item of a media type, or "file" when the type is unknown. */
export function mediaItemWord(mediaType: string | null | undefined): string {
  if (!mediaType) return 'file';
  return MEDIA_ITEMS[mediaType] ?? `${mediaType} file`;
}

/**
 * What Toasty says a dataset holds: the server's one media type when it is
 * locked to one (`solo_media_type`), else a mix.
 */
export function datasetMediaWords(solo: string | null | undefined): string {
  if (!solo) return 'images or other media';
  return MEDIA_PLURALS[solo] ?? `${solo} files`;
}
