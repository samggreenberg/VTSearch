import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';

import { CollapsePhotosService } from './collapse-photos.service';
import { SourcePhotoService } from './source-photo.service';

/** Collapse to photos (#4751): the user's switch, effective only where faces have photos. */
describe('CollapsePhotosService', () => {
  const offered = signal(false);
  let service: CollapsePhotosService;

  beforeEach(() => {
    offered.set(false);
    TestBed.configureTestingModule({ providers: [{ provide: SourcePhotoService, useValue: { offered } }] });
    service = TestBed.inject(CollapsePhotosService);
  });

  it('starts off', () => {
    expect(service.on()).toBe(false);
    expect(service.active()).toBe(false);
  });

  it('collapses only on a dataset whose faces have photos', () => {
    service.set(true);
    expect(service.active()).toBe(false);
    offered.set(true);
    expect(service.offered()).toBe(true);
    expect(service.active()).toBe(true);
  });

  it('keeps the switch across a dataset that cannot collapse', () => {
    offered.set(true);
    service.toggle();
    offered.set(false);
    expect(service.on()).toBe(true);
    expect(service.active()).toBe(false);
    offered.set(true);
    expect(service.active()).toBe(true);
    service.toggle();
    expect(service.active()).toBe(false);
  });
});
