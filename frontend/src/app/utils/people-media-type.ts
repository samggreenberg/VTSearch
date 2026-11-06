/** The one media type whose datasets have people (``PEOPLE_MEDIA_TYPES`` in
 *  ``vtscore/datasets/people.py``): a Face dataset's crops.
 *
 *  A module of its own because the People route's guard reads it, and a guard
 *  is on the eager path: taken from `card-context-menu-items.ts`, it pulled the
 *  whole card menu into the initial bundle. */
export const PEOPLE_MEDIA_TYPE = 'face';
