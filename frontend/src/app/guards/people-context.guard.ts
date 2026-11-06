import { CanActivateFn } from '@angular/router';
import { PEOPLE_MEDIA_TYPE } from '../utils/people-media-type';
import { datasetContextGuard } from './browse-context.guard';

/**
 * The guard of `/people/:datasetId` (#5590): Browse's dataset-only guard, for
 * a Face dataset alone. Any other dataset has no people to show (the routes
 * answer 409 `no_people`), so a link to one bounces to the Dashboard with a
 * toast saying why, before the view asks.
 */
export const peopleContextGuard: CanActivateFn = datasetContextGuard('people', (dataset) =>
  dataset.media_type === PEOPLE_MEDIA_TYPE
    ? null
    : `People is for Face datasets, and '${dataset.name}' is not one.`,
);
