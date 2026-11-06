/**
 * The one media type whose datasets have people (``PEOPLE_MEDIA_TYPES`` in
 * ``vtscore/datasets/people.py``): a Face dataset's crops (#5590).
 *
 * Its own module, so the eagerly loaded People route guard can read it without
 * pulling the Dashboard's card-menu icons into the initial bundle.
 */
export const PEOPLE_MEDIA_TYPE = 'face';
