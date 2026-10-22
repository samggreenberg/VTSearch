import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { Router, provideRouter } from '@angular/router';

import { routes } from './app.routes';
import { configureZoneless } from './testing/zoneless-testbed';

@Component({ standalone: true, template: '' })
class TuneStubComponent {}

describe('app routes', () => {
  // The Tune view was Find until #4525 and Test until #5098; a link saved or
  // sent under either name still opens the same pair (#5129). Only the
  // redirect entry is routed for real, so the Tune route's guard stays out.
  it.each(['find', 'test'])('sends /%s/:datasetId/:detectorId to the same pair in Tune', async (old) => {
    const legacy = routes.filter((r) => r.path === `${old}/:datasetId/:detectorId`);
    expect(legacy).toHaveLength(1);
    configureZoneless({
      providers: [
        provideRouter([...legacy, { path: 'tune/:datasetId/:detectorId', component: TuneStubComponent }]),
      ],
    });
    const router = TestBed.inject(Router);

    await router.navigateByUrl(`/${old}/ds1/det1`);

    expect(router.url).toBe('/tune/ds1/det1');
  });
});
