import { TestBed } from '@angular/core/testing';
import { Observable, of } from 'rxjs';
import { EmbedderCapabilityService } from './embedder-capability.service';
import { DatasetsListingsApiService } from './datasets-listings-api.service';
import type { EmbedderInfo } from '../models/api.models';

describe('EmbedderCapabilityService', () => {
  let service: EmbedderCapabilityService;
  let getEmbedders: ReturnType<typeof vi.fn>;

  const infos: EmbedderInfo[] = [
    { name: 'clip', media_type_id: 'image', supports_text: true },
    { name: 'dino', media_type_id: 'image', supports_text: false, supports_patch_regions: true },
    { name: 'sift', media_type_id: 'image', supports_text: false, supports_geometric_verification: true },
    { name: 'e5', media_type_id: 'text' }, // supports_text undefined ⇒ treated as text-capable
  ];

  beforeEach(() => {
    getEmbedders = vi.fn((): Observable<EmbedderInfo[]> => of(infos));
    TestBed.configureTestingModule({
      providers: [
        EmbedderCapabilityService,
        { provide: DatasetsListingsApiService, useValue: { getEmbedders } },
      ],
    });
    service = TestBed.inject(EmbedderCapabilityService);
  });

  it('is unloaded until the registry resolves', () => {
    expect(service.loaded).toBe(false);
    expect(service.infos()).toBeNull();
  });

  it('defaults to text-supported before the registry loads', () => {
    expect(service.supportsText('anything')).toBe(true);
  });

  it('ensureLoaded loads the registry once (idempotent)', () => {
    service.ensureLoaded();
    expect(service.loaded).toBe(true);
    expect(service.infos()).toEqual(infos);
    service.ensureLoaded();
    expect(getEmbedders).toHaveBeenCalledTimes(1);
  });

  it('ensureLoaded on error still marks the registry loaded (empty)', () => {
    getEmbedders.mockReturnValueOnce(new Observable<EmbedderInfo[]>((o) => o.error(new Error('500'))));
    service.ensureLoaded();
    expect(service.loaded).toBe(true);
    expect(service.infos()).toEqual([]);
  });

  describe('once loaded', () => {
    beforeEach(() => service.ensureLoaded());

    it('supportsText is false only for a known vision-only embedder', () => {
      expect(service.supportsText('dino')).toBe(false);
      expect(service.supportsText('sift')).toBe(false);
      expect(service.supportsText('clip')).toBe(true);
      expect(service.supportsText('e5')).toBe(true);
    });

    it('supportsText defaults to true for empty or unrecognised names', () => {
      expect(service.supportsText('')).toBe(true);
      expect(service.supportsText('mystery')).toBe(true);
    });

    it('supportsTextAny is true if any name is text-capable', () => {
      expect(service.supportsTextAny(['dino', 'clip'])).toBe(true);
      expect(service.supportsTextAny(['dino', 'sift'])).toBe(false);
      expect(service.supportsTextAny([])).toBe(true);
      expect(service.supportsTextAny(null)).toBe(true);
    });

    it('classifies embedders by the backend precedence', () => {
      expect(service.embedderType('sift')).toBe('structural');
      expect(service.embedderType('dino')).toBe('patch_semantic');
      expect(service.embedderType('clip')).toBe('semantic');
      expect(service.embedderType('e5')).toBe('semantic');
      expect(service.embedderType('mystery')).toBe('');
    });

    it('suppliedTypes returns the distinct types in display order', () => {
      expect(service.suppliedTypes(['sift', 'clip', 'dino'])).toEqual([
        'semantic',
        'patch_semantic',
        'structural',
      ]);
      expect(service.suppliedTypes(['clip', 'e5'])).toEqual(['semantic']);
      expect(service.suppliedTypes(null)).toEqual([]);
    });

    it('firstOfType returns the first bound embedder of a type, else empty', () => {
      expect(service.firstOfType(['clip', 'e5'], 'semantic')).toBe('clip');
      expect(service.firstOfType(['clip'], 'structural')).toBe('');
      expect(service.firstOfType(['clip'], '')).toBe('');
      expect(service.firstOfType(null, 'semantic')).toBe('');
    });

    it('supportsRegionOverlayAny covers patch-region and structural embedders', () => {
      expect(service.supportsRegionOverlayAny(['dino'])).toBe(true);
      expect(service.supportsRegionOverlayAny(['sift'])).toBe(true);
      expect(service.supportsRegionOverlayAny(['clip'])).toBe(false);
      expect(service.supportsRegionOverlayAny(null)).toBe(false);
    });

    it('supportsStructuralAny is true only for a structural embedder', () => {
      expect(service.supportsStructuralAny(['sift'])).toBe(true);
      expect(service.supportsStructuralAny(['dino', 'clip'])).toBe(false);
      expect(service.supportsStructuralAny(null)).toBe(false);
    });
  });
});
