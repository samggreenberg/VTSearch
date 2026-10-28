import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { ImportDefaultsService } from './import-defaults.service';
import { SettingsStateService } from '../../../../../services/settings-state.service';
import { ToastService } from '../../../../../services/toast.service';
import { CleanerInfo, MediaTypeInfo } from '../../../../../models/api.models';
import { demoOutputDrafts } from './multi-output.util';

describe('ImportDefaultsService', () => {
  let settings: ReturnType<typeof signal<Record<string, unknown> | null>>;
  let service: ImportDefaultsService;
  let toastSuccessSpy: ReturnType<typeof vi.fn>;
  let updateSpy: ReturnType<typeof vi.fn>;

  const mediaTypes: MediaTypeInfo[] = [
    { type_id: 'audio', name: 'Audio', folder_import_name: 'audio' } as MediaTypeInfo,
    { type_id: 'image', name: 'Image', folder_import_name: 'images' } as MediaTypeInfo,
  ];

  beforeEach(() => {
    settings = signal<Record<string, unknown> | null>({});
    toastSuccessSpy = vi.fn();
    updateSpy = vi.fn(() => of({}));
    const settingsStub: Partial<SettingsStateService> = {
      settingsSignal: settings as unknown as SettingsStateService['settingsSignal'],
      update: updateSpy as unknown as SettingsStateService['update'],
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
    settings.set({ solo_media_type: 'image' });
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
      service.snapshotImportConfig('image', 'clip', 'image_default', {}, [], embedders, clippers, [], []),
    ).toEqual({});

    // Non-default embedder is kept.
    expect(
      service.snapshotImportConfig('image', 'siglip', 'image_default', {}, [], embedders, clippers, [], []),
    ).toEqual({ embedder: 'siglip' });
  });

  describe('cleanup selection', () => {
    // exif_orient is on by default; trim_borders is off by default and takes a param.
    const cleaners: CleanerInfo[] = [
      { name: 'exif_orient', media_type: 'image', default_enabled: true } as CleanerInfo,
      {
        name: 'trim_borders',
        media_type: 'image',
        default_enabled: false,
        parameters: [{ key: 'tolerance', default: 10 }],
      } as CleanerInfo,
    ];
    const embedders = [{ name: 'clip', is_default: true } as any];
    const clippers = [{ name: 'image_default' } as any];
    const snapshot = (selected: { name: string; params: Record<string, number | string> }[]) =>
      service.snapshotImportConfig('image', 'clip', 'image_default', {}, [], embedders, clippers, selected, cleaners);

    it('snapshotImportConfig carries no cleaners entry for the registry-default selection', () => {
      expect(snapshot([{ name: 'exif_orient', params: {} }])).toEqual({});
    });

    it('snapshotImportConfig records a default-enabled gate switched off, and every other listed cleaner', () => {
      expect(snapshot([])).toEqual({
        cleaners: { exif_orient: { enabled: false }, trim_borders: { enabled: false } },
      });
    });

    it('snapshotImportConfig records a non-default gate ticked with its params', () => {
      expect(
        snapshot([
          { name: 'exif_orient', params: {} },
          { name: 'trim_borders', params: { tolerance: 25 } },
        ]),
      ).toEqual({
        cleaners: { exif_orient: { enabled: true }, trim_borders: { enabled: true, params: { tolerance: 25 } } },
      });
    });

    it('defaultCleanerSelection seeds from default_enabled when nothing is saved', () => {
      expect(service.defaultCleanerSelection(cleaners, 'images', mediaTypes)).toEqual([
        { name: 'exif_orient', params: {} },
      ]);
    });

    it('defaultCleanerSelection applies the saved choice per cleaner', () => {
      settings.set({
        import_defaults_by_media_type: {
          image: {
            cleaners: {
              exif_orient: { enabled: false },
              trim_borders: { enabled: true, params: { tolerance: 25, retired_knob: 3 } },
              uninstalled_gate: { enabled: true },
            },
          },
        },
      });
      const withNewcomer: CleanerInfo[] = [
        ...cleaners,
        { name: 'deskew', media_type: 'image', default_enabled: true } as CleanerInfo,
      ];
      // exif_orient stays off; trim_borders keeps its declared param only;
      // the unregistered gate is skipped; deskew, registered since the save,
      // follows its default_enabled.
      expect(service.defaultCleanerSelection(withNewcomer, 'images', mediaTypes)).toEqual([
        { name: 'trim_borders', params: { tolerance: 25 } },
        { name: 'deskew', params: {} },
      ]);
    });

    it('maybeOfferSaveImportDefaults offers to save a cleanup-only change, naming cleanup', () => {
      service.maybeOfferSaveImportDefaults('image', snapshot([]), mediaTypes);
      expect(toastSuccessSpy).toHaveBeenCalledTimes(1);
      expect(toastSuccessSpy.mock.calls[0][0].detail).toContain('cleanup');
    });

    it('maybeOfferSaveImportDefaults stays silent when the saved cleanup choice matches', () => {
      settings.set({
        import_defaults_by_media_type: {
          image: { cleaners: { exif_orient: { enabled: false }, trim_borders: { enabled: false } } },
        },
      });
      service.maybeOfferSaveImportDefaults('image', snapshot([]), mediaTypes);
      expect(toastSuccessSpy).not.toHaveBeenCalled();
    });

    it('maybeOfferSaveImportDefaults offers again when only a cleaner param differs from the saved one', () => {
      settings.set({
        import_defaults_by_media_type: {
          image: {
            cleaners: { exif_orient: { enabled: true }, trim_borders: { enabled: true, params: { tolerance: 25 } } },
          },
        },
      });
      const cfg = snapshot([
        { name: 'exif_orient', params: {} },
        { name: 'trim_borders', params: { tolerance: 40 } },
      ]);
      service.maybeOfferSaveImportDefaults('image', cfg, mediaTypes);
      expect(toastSuccessSpy).toHaveBeenCalledTimes(1);
    });
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

  // #5355: Multi-Dataset (and which kinds it makes) and the two ingest
  // toggles are part of what "Save as default" saves.
  describe('Multi-Dataset and ingest toggles', () => {
    const withFace: MediaTypeInfo[] = [...mediaTypes, { type_id: 'face', name: 'Face' } as MediaTypeInfo];
    const imageDemoRows = () => demoOutputDrafts(withFace, 'image', ['face']);

    it('ingestTogglesFor reads the saved toggles, both off when nothing is saved', () => {
      expect(service.ingestTogglesFor('image', mediaTypes)).toEqual({ buildProjection: false, mergeNearDuplicates: false });
      settings.set({ import_defaults_by_media_type: { image: { build_projection: true } } });
      expect(service.ingestTogglesFor('images', mediaTypes)).toEqual({ buildProjection: true, mergeNearDuplicates: false });
    });

    it('savedMultiDatasetDrafts ticks exactly the saved kinds', () => {
      expect(service.savedMultiDatasetDrafts('image', withFace, imageDemoRows())).toBeNull();
      settings.set({ import_defaults_by_media_type: { image: { multi_dataset: ['face'] } } });
      const drafts = service.savedMultiDatasetDrafts('image', withFace, imageDemoRows())!;
      expect(drafts.map((d) => [d.category, d.checked])).toEqual([
        ['image', false],
        ['face', true],
      ]);
    });

    it('savedMultiDatasetDrafts stays on the one-dataset form when no saved kind is a row here', () => {
      settings.set({ import_defaults_by_media_type: { image: { multi_dataset: ['text'] } } });
      expect(service.savedMultiDatasetDrafts('image', withFace, imageDemoRows())).toBeNull();
    });

    it('snapshotIngestToggles keys only a ticked toggle', () => {
      expect(service.snapshotIngestToggles(false, false)).toEqual({});
      expect(service.snapshotIngestToggles(true, false)).toEqual({ build_projection: true });
      expect(service.snapshotIngestToggles(false, true)).toEqual({ merge_near_duplicates: true });
    });

    it('snapshotMultiDatasetConfig saves the ticked kinds and toggles, carrying the single-dataset settings over', () => {
      settings.set({
        import_defaults_by_media_type: {
          image: { embedder: 'siglip', merge_near_duplicates: true, multi_dataset: ['image'] },
        },
      });
      const drafts = imageDemoRows().map((d) => ({ ...d, checked: true }));
      expect(service.snapshotMultiDatasetConfig('image', withFace, drafts, true, false)).toEqual({
        embedder: 'siglip',
        multi_dataset: ['image', 'face'],
        build_projection: true,
      });
    });

    it('maybeOfferSaveImportDefaults offers a Multi-Dataset import, naming its kinds and toggles', () => {
      service.maybeOfferSaveImportDefaults('image', { multi_dataset: ['image', 'face'], build_projection: true }, withFace);
      expect(toastSuccessSpy).toHaveBeenCalledTimes(1);
      const toast = toastSuccessSpy.mock.calls[0][0];
      expect(toast.message).toContain('Image');
      expect(toast.detail).toBe('Next time it starts with Multi-Dataset (Image, Face) and Build Browse map now.');
    });

    it('maybeOfferSaveImportDefaults stays silent when the same kinds are saved in another order', () => {
      settings.set({ import_defaults_by_media_type: { image: { multi_dataset: ['face', 'image'] } } });
      service.maybeOfferSaveImportDefaults('image', { multi_dataset: ['image', 'face'] }, withFace);
      expect(toastSuccessSpy).not.toHaveBeenCalled();
    });

    it('maybeOfferSaveImportDefaults offers a toggle-only change', () => {
      service.maybeOfferSaveImportDefaults('image', { merge_near_duplicates: true }, mediaTypes);
      expect(toastSuccessSpy).toHaveBeenCalledTimes(1);
      expect(toastSuccessSpy.mock.calls[0][0].detail).toBe('Next time it starts with Merge near-duplicates.');
    });

    it('offers to save a one-dataset import of a type whose default opens Multi-Dataset, and saving forgets it', () => {
      settings.set({ import_defaults_by_media_type: { image: { multi_dataset: ['image', 'face'] }, audio: { embedder: 'clap' } } });
      service.maybeOfferSaveImportDefaults('image', {}, withFace);
      expect(toastSuccessSpy).toHaveBeenCalledTimes(1);
      const toast = toastSuccessSpy.mock.calls[0][0];
      expect(toast.detail).toBe('Next time it starts with one dataset, not Multi-Dataset.');

      toast.action.onClick();
      // Nothing left to save for Image: its entry goes, the other types' stay.
      expect(updateSpy).toHaveBeenCalledWith({ import_defaults_by_media_type: { audio: { embedder: 'clap' } } });
    });
  });
});
