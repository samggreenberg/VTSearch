import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { ImportDefaultsService } from './import-defaults.service';
import { SettingsStateService } from '../../../../../services/settings-state.service';
import { ToastService } from '../../../../../services/toast.service';
import { MediaTypeInfo } from '../../../../../models/api.models';

describe('ImportDefaultsService', () => {
  let settings: ReturnType<typeof signal<Record<string, unknown> | null>>;
  let service: ImportDefaultsService;
  let toastSuccessSpy: ReturnType<typeof vi.fn>;

  const mediaTypes: MediaTypeInfo[] = [
    { type_id: 'audio', name: 'Audio', folder_import_name: 'audio' } as MediaTypeInfo,
    { type_id: 'image', name: 'Image', folder_import_name: 'images' } as MediaTypeInfo,
  ];

  beforeEach(() => {
    settings = signal<Record<string, unknown> | null>({});
    toastSuccessSpy = vi.fn();
    const settingsStub: Partial<SettingsStateService> = {
      settingsSignal: settings as unknown as SettingsStateService['settingsSignal'],
      update: () => of({}) as ReturnType<SettingsStateService['update']>,
    };
    const toastStub: Partial<ToastService> = { success: toastSuccessSpy as unknown as ToastService['success'] };

    TestBed.configureTestingModule({
      providers: [
        ImportDefaultsService,
        { provide: SettingsStateService, useValue: settingsStub },
        { provide: ToastService, useValue: toastStub },
      ],
    });
    service = TestBed.inject(ImportDefaultsService);
  });

  it('effectiveSoloMediaType is null when no solo lock is set', () => {
    expect(service.effectiveSoloMediaType).toBeNull();
  });

  it('effectiveSoloMediaType/effectiveSoloFolderName resolve the solo lock', () => {
    settings.set({ effective_solo_media_type: 'image' });
    expect(service.effectiveSoloMediaType).toBe('image');
    expect(service.effectiveSoloFolderName(mediaTypes)).toBe('images');
  });

  it('lockedEmbedderFor returns the lock only when the embedder is still registered', () => {
    settings.set({ effective_solo_embedder_per_media_type: { image: 'clip' } });
    expect(service.lockedEmbedderFor('images', mediaTypes, [{ name: 'clip' } as any])).toBe('clip');
    // Stale lock: embedder no longer in the list for this type.
    expect(service.lockedEmbedderFor('images', mediaTypes, [{ name: 'siglip' } as any])).toBe('');
  });

  it('pickInitialEmbedder prefers the solo lock, then the guess, then the saved pick, then the first option', () => {
    const embedders = [{ name: 'clip' } as any, { name: 'siglip' } as any];

    // No settings at all: falls back to the first option.
    expect(service.pickInitialEmbedder(embedders, 'images', mediaTypes, '')).toBe('clip');

    // Guessed embedder wins over the plain fallback.
    expect(service.pickInitialEmbedder(embedders, 'images', mediaTypes, 'siglip')).toBe('siglip');

    // A saved last-pick wins when there's no guess.
    settings.set({ last_embedder_per_media_type: { image: 'siglip' } });
    expect(service.pickInitialEmbedder(embedders, 'images', mediaTypes, '')).toBe('siglip');

    // The solo lock overrides everything else.
    settings.set({
      effective_solo_embedder_per_media_type: { image: 'clip' },
      last_embedder_per_media_type: { image: 'siglip' },
    });
    expect(service.pickInitialEmbedder(embedders, 'images', mediaTypes, 'siglip')).toBe('clip');
  });

  it('chooseClipperForType prefers a saved default, falling back to the registry default', () => {
    const clippers = [{ name: 'image_default' } as any, { name: 'face_crop' } as any];

    expect(service.chooseClipperForType(clippers, 'images', mediaTypes)).toEqual({
      name: 'image_default',
      params: null,
    });

    settings.set({
      import_defaults_by_media_type: { image: { clipper: 'face_crop', clipper_params: { size: 5 } } },
    });
    expect(service.chooseClipperForType(clippers, 'images', mediaTypes)).toEqual({
      name: 'face_crop',
      params: { size: 5 },
    });
  });

  it('specsListWithDefaultsFor falls back to the bare native row with no saved defaults', () => {
    expect(service.specsListWithDefaultsFor(mediaTypes, 'image', [])).toEqual([
      { source_type: 'image', converter: null, params: {} },
    ]);
  });

  it('snapshotImportConfig omits registry defaults and keeps only real overrides', () => {
    const embedders = [{ name: 'clip', is_default: true } as any, { name: 'siglip', is_default: false } as any];
    const clippers = [{ name: 'image_default' } as any];

    // Default embedder + default clipper + no extra specs -> empty snapshot.
    expect(
      service.snapshotImportConfig('image', 'clip', 'image_default', {}, [], embedders, clippers),
    ).toEqual({});

    // Non-default embedder is kept.
    expect(
      service.snapshotImportConfig('image', 'siglip', 'image_default', {}, [], embedders, clippers),
    ).toEqual({ embedder: 'siglip' });
  });

  it('maybeOfferSaveImportDefaults skips a config with no meaningful overrides', () => {
    service.maybeOfferSaveImportDefaults('image', {}, mediaTypes);
    expect(toastSuccessSpy).not.toHaveBeenCalled();
  });

  it('maybeOfferSaveImportDefaults offers to save a real override', () => {
    service.maybeOfferSaveImportDefaults('image', { embedder: 'siglip' }, mediaTypes);
    expect(toastSuccessSpy).toHaveBeenCalledTimes(1);
    expect(toastSuccessSpy.mock.calls[0][0].message).toContain('Image');
  });

  it('maybeOfferSaveImportDefaults skips when the exact config is already saved', () => {
    settings.set({ import_defaults_by_media_type: { image: { embedder: 'siglip' } } });
    service.maybeOfferSaveImportDefaults('image', { embedder: 'siglip' }, mediaTypes);
    expect(toastSuccessSpy).not.toHaveBeenCalled();
  });
});
