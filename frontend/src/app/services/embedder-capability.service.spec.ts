import { signal } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { Observable, of } from 'rxjs';
import { EmbedderCapabilityService } from './embedder-capability.service';
import { DatasetsListingsApiService } from './datasets-listings-api.service';
import { SettingsStateService } from './settings-state.service';
import type { EmbedderInfo } from '../models/api.models';

describe('EmbedderCapabilityService', () => {
  let service: EmbedderCapabilityService;
  let getEmbedders: ReturnType<typeof vi.fn>;
  // Stand-in for SettingsStateService: the service reads only settingsSignal().
  let settingsSignal: ReturnType<typeof signal<{ semantic_only?: boolean } | null>>;

  const infos: EmbedderInfo[] = [
    { name: 'clip', media_type_id: 'image', supports_text: true },
    { name: 'dino', media_type_id: 'image', supports_text: false, supports_patch_regions: true },
    { name: 'sift', media_type_id: 'image', supports_text: false, supports_geometric_verification: true },
    { name: 'e5', media_type_id: 'text' }, // supports_text undefined ⇒ treated as text-capable
  ];

  beforeEach(() => {
    getEmbedders = vi.fn((): Observable<EmbedderInfo[]> => of(infos));
    settingsSignal = signal<{ semantic_only?: boolean } | null>(null);
    TestBed.configureTestingModule({
      providers: [
        EmbedderCapabilityService,
        { provide: DatasetsListingsApiService, useValue: { getEmbedders } },
        { provide: SettingsStateService, useValue: { settingsSignal } },
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

  describe('semantic_only lock', () => {
    it('is off before settings load, so nothing is hidden on missing metadata', () => {
      expect(service.semanticOnly()).toBe(false);
      expect(service.offeredTypes).toEqual(['semantic', 'patch_semantic', 'structural']);
    });

    it('is off when the server reports semantic_only false', () => {
      settingsSignal.set({ semantic_only: false });
      expect(service.semanticOnly()).toBe(false);
      expect(service.offeredTypes).toEqual(['semantic', 'patch_semantic', 'structural']);
    });

    it('narrows the offered types to Semantic when the server reports the lock', () => {
      settingsSignal.set({ semantic_only: true });
      expect(service.semanticOnly()).toBe(true);
      expect(service.offeredTypes).toEqual(['semantic']);
    });

    it('does not change how bound embedders are classified', () => {
      service.ensureLoaded();
      settingsSignal.set({ semantic_only: true });
      // A dataset that still binds a prototype embedder keeps its real type;
      // the lock governs what is *offered*, not how existing data reads.
      expect(service.embedderType('sift')).toBe('structural');
      expect(service.suppliedTypes(['clip', 'dino'])).toEqual(['semantic', 'patch_semantic']);
    });
  });
});
