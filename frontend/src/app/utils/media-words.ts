/** What a dataset of each media type holds, in words (#4680). */
export const MEDIA_PLURALS: Readonly<Record<string, string>> = {
  image: 'images',
  audio: 'audio clips',
  text: 'text paragraphs',
  video: 'videos',
  document: 'documents',
  face: 'faces',
};

/**
 * What Toasty says a dataset holds: the server's one media type when it is
 * locked to one (`solo_media_type`), else a mix.
 */
export function datasetMediaWords(solo: string | null | undefined): string {
  if (!solo) return 'images or other media';
  return MEDIA_PLURALS[solo] ?? `${solo} files`;
}
