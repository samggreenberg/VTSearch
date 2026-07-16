import { ComponentFixture, TestBed } from '@angular/core/testing';
import { Observable, of, throwError } from 'rxjs';

import { ImportDefaultsSettingsComponent } from './import-defaults-settings.component';
import { DatasetsListingsApiService } from '../../../../services/datasets-listings-api.service';
import {
  ClipperInfo,
  ConverterInfo,
  EmbedderInfo,
  ImportDefaultsByMediaType,
  MediaTypeInfo,
  SourceSpec,
} from '../../../../models/api.models';
import { provideZoneless } from '../../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../../testing/settle-resource';

/**
 * Unit spec for the Import-Defaults settings sub-panel. The panel reads a
 * per-mediaType ``ImportDefaultsByMediaType`` map (embedder / clipper /
 * source-specs) and emits a full replacement map on every edit. These tests
 * drive the listings API through a stub so the lazy per-tab embedder/clipper/
 * converter fetches resolve synchronously, and assert on the component's
 * derived getters and the ``defaultsChange`` payloads.
 */
describe('ImportDefaultsSettingsComponent', () => {
  let fixture: ComponentFixture<ImportDefaultsSettingsComponent>;
  let component: ImportDefaultsSettingsComponent;

  const mediaTypes: MediaTypeInfo[] = [
    { type_id: 'audio', name: 'Sound', icon: 'audio' },
    { type_id: 'image', name: 'Image', icon: 'image' },
  ];

  const audioEmbedders: EmbedderInfo[] = [
    {
      name: 'clap',
      display_name: 'CLAP',
      media_type_id: 'audio',
      is_default: true,
      license_notice: null,
    },
    {
      name: 'wav2vec',
      display_name: 'Wav2Vec',
      media_type_id: 'audio',
      is_default: false,
      license_notice: 'Non-commercial use only',
    },
  ] as unknown as EmbedderInfo[];

  const audioClippers: ClipperInfo[] = [
    { name: 'whole', media_type: 'audio' },
    { name: 'window', media_type: 'audio' },
  ] as unknown as ClipperInfo[];

  const audioConverters: ConverterInfo[] = [
    { name: 'video_to_audio', source_type: 'video', target_type: 'audio' },
  ];

  // Records of which mediaType each listing endpoint was asked for, so tests can
  // assert on lazy-load / caching behaviour.
  let embedderCalls: string[];
  let clipperCalls: string[];
  let converterCalls: string[];

  // Swappable per-test so the error-path test can make a listing throw.
  let embeddersFor: (mt?: string) => Observable<EmbedderInfo[]>;
  let clippersFor: (mt?: string) => Observable<ClipperInfo[]>;
  let convertersFor: (mt?: string) => Observable<ConverterInfo[]>;

  beforeEach(() => {
    embedderCalls = [];
    clipperCalls = [];
    converterCalls = [];
    embeddersFor = (mt) => of(mt === 'audio' ? audioEmbedders : []);
    clippersFor = (mt) => of(mt === 'audio' ? audioClippers : []);
    convertersFor = (mt) => of(mt === 'audio' ? audioConverters : []);

    const listingsStub: Partial<DatasetsListingsApiService> = {
      getEmbedders: (mt?: string) => {
        embedderCalls.push(mt ?? '');
        return embeddersFor(mt);
      },
      getClippers: (mt?: string) => {
        clipperCalls.push(mt ?? '');
        return clippersFor(mt);
      },
      getConverters: (mt?: string) => {
        converterCalls.push(mt ?? '');
        return convertersFor(mt);
      },
    };

    TestBed.configureTestingModule({
      imports: [ImportDefaultsSettingsComponent],
      providers: [
        ...provideZoneless(),
        { provide: DatasetsListingsApiService, useValue: listingsStub },
      ],
    });
  });

  /** Create + init the component with the given inputs. Returns after the
   *  tab-seeding effect (and its synchronous listing subscriptions) has run. */
  async function create(inputs: {
    mediaTypes?: MediaTypeInfo[];
    defaults?: ImportDefaultsByMediaType;
    solo?: string | null;
  } = {}): Promise<void> {
    fixture = TestBed.createComponent(ImportDefaultsSettingsComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('mediaTypes', inputs.mediaTypes ?? mediaTypes);
    fixture.componentRef.setInput('defaults', inputs.defaults ?? {});
    if (inputs.solo !== undefined) {
      fixture.componentRef.setInput('effectiveSoloMediaType', inputs.solo);
    }
    await settleZoneless(fixture);
  }

  /** Subscribe to the component's ``defaultsChange`` output and collect emits. */
  function captureEmits(): ImportDefaultsByMediaType[] {
    const out: ImportDefaultsByMediaType[] = [];
    component.defaultsChange.subscribe((v) => out.push(v));
    return out;
  }

  it('creates and picks the first visible type as the active tab', async () => {
    await create();
    expect(component).toBeTruthy();
    expect(component.activeType).toBe('audio');
    // Lazy-load fired for the initial tab only.
    expect(embedderCalls).toEqual(['audio']);
    expect(clipperCalls).toEqual(['audio']);
    expect(converterCalls).toEqual(['audio']);
    expect(component.activeEmbedders).toEqual(audioEmbedders);
    expect(component.activeClippers).toEqual(audioClippers);
    expect(component.activeConverters).toEqual(audioConverters);
    expect(component.isLoading).toBe(false);
  });

  it('renders no tabs when there are no media types', async () => {
    await create({ mediaTypes: [] });
    expect(component.visibleTypes.length).toBe(0);
    expect(component.activeType).toBe('');
    expect(embedderCalls).toEqual([]);
  });

  it('collapses to the solo media type when effectiveSoloMediaType is set', async () => {
    await create({ solo: 'image' });
    expect(component.visibleTypes.map((mt) => mt.type_id)).toEqual(['image']);
    expect(component.activeType).toBe('image');
    expect(embedderCalls).toEqual(['image']);
  });

  it('lazy-loads a tab on first switch and caches it afterwards', async () => {
    await create();
    expect(embedderCalls).toEqual(['audio']);

    // Re-selecting the current tab does not refetch (already cached).
    component.selectTab('audio');
    expect(embedderCalls).toEqual(['audio']);

    // Switching to a new tab fetches once...
    component.selectTab('image');
    expect(component.activeType).toBe('image');
    expect(embedderCalls).toEqual(['audio', 'image']);

    // ...and switching back does not refetch.
    component.selectTab('audio');
    expect(embedderCalls).toEqual(['audio', 'image']);
  });

  it('falls back to empty lists when a listing errors', async () => {
    embeddersFor = () => throwError(() => new Error('boom'));
    await create();
    expect(component.activeEmbedders).toEqual([]);
    // The other two still loaded, and the loading flag clears once all settle.
    expect(component.activeClippers).toEqual(audioClippers);
    expect(component.isLoading).toBe(false);
  });

  it('splits embedders into recommended vs advanced and derives the default', async () => {
    await create();
    expect(component.recommendedEmbedders.map((e) => e.name)).toEqual(['clap']);
    expect(component.advancedEmbedderOptions.map((e) => e.name)).toEqual(['wav2vec']);
    expect(component.defaultEmbedderName).toBe('clap');
  });

  it('defaultEmbedderName falls back to the first embedder when none is flagged default', async () => {
    embeddersFor = () =>
      of([
        { name: 'a', media_type_id: 'audio', is_default: false },
        { name: 'b', media_type_id: 'audio', is_default: false },
      ] as unknown as EmbedderInfo[]);
    await create();
    expect(component.defaultEmbedderName).toBe('a');
  });

  it('displays the override embedder when set, otherwise the built-in default', async () => {
    await create();
    expect(component.currentEmbedder).toBe('');
    expect(component.displayedEmbedder).toBe('clap');

    await create({ defaults: { audio: { embedder: 'wav2vec' } } });
    expect(component.currentEmbedder).toBe('wav2vec');
    expect(component.displayedEmbedder).toBe('wav2vec');
  });

  it('surfaces the active embedder license notice only when an override is set', async () => {
    await create();
    expect(component.licenseNotice).toBeNull();

    await create({ defaults: { audio: { embedder: 'wav2vec' } } });
    expect(component.licenseNotice).toBe('Non-commercial use only');
  });

  it('onEmbedderChange emits an override for a non-default pick', async () => {
    await create();
    const emits = captureEmits();
    component.onEmbedderChange('wav2vec');
    expect(emits).toEqual([{ audio: { embedder: 'wav2vec' } }]);
  });

  it('onEmbedderChange clears the override when the built-in default is re-picked', async () => {
    await create({ defaults: { audio: { embedder: 'wav2vec' } } });
    const emits = captureEmits();
    component.onEmbedderChange('clap'); // clap is the default → clears
    expect(emits).toEqual([{}]);
  });

  it('reports whether the active type has any saved overrides', async () => {
    await create();
    expect(component.hasOverridesForActiveType).toBe(false);

    await create({ defaults: { audio: { clipper: 'window' } } });
    expect(component.hasOverridesForActiveType).toBe(true);

    await create({ defaults: { audio: { source_specs: [] } } });
    expect(component.hasOverridesForActiveType).toBe(false);
  });

  it('synthesises a native source-spec row when none is saved', async () => {
    await create();
    expect(component.currentSourceSpecsForPicker).toEqual([
      { source_type: 'audio', converter: null, params: {} },
    ]);
  });

  it('returns saved source specs as-is when a native row is already present', async () => {
    const saved: SourceSpec[] = [
      { source_type: 'audio', converter: null, params: {} },
      { source_type: 'video', converter: 'video_to_audio', params: {} },
    ];
    await create({ defaults: { audio: { source_specs: saved } } });
    expect(component.currentSourceSpecsForPicker).toEqual(saved);
  });

  it('onSourceSpecsChange persists the specs, stripping an empty list', async () => {
    await create();
    const emits = captureEmits();

    component.onSourceSpecsChange([
      { source_type: 'video', converter: 'video_to_audio', params: {} },
    ]);
    expect(emits.at(-1)).toEqual({
      audio: { source_specs: [{ source_type: 'video', converter: 'video_to_audio', params: {} }] },
    });

    // An empty array is stripped, and with nothing else set the type key is dropped.
    component.onSourceSpecsChange([]);
    expect(emits.at(-1)).toEqual({});
  });

  it('opens the clipper chooser only when more than one clipper is available', async () => {
    await create();
    component.openClipperChooser();
    expect(component.clipperChooserOpen).toBe(true);

    // Only one clipper → no chooser.
    clippersFor = () => of([{ name: 'whole', media_type: 'audio' }] as unknown as ClipperInfo[]);
    await create();
    component.openClipperChooser();
    expect(component.clipperChooserOpen).toBe(false);
  });

  it('onClipperSelected persists the clipper + params and closes the chooser', async () => {
    await create();
    component.clipperChooserOpen = true;
    const emits = captureEmits();

    component.onClipperSelected({ name: 'window', params: { size: 5 } });
    expect(component.clipperChooserOpen).toBe(false);
    expect(emits.at(-1)).toEqual({
      audio: { clipper: 'window', clipper_params: { size: 5 } },
    });
  });

  it('onClipperChooserCancelled closes the chooser without emitting', async () => {
    await create();
    component.clipperChooserOpen = true;
    const emits = captureEmits();
    component.onClipperChooserCancelled();
    expect(component.clipperChooserOpen).toBe(false);
    expect(emits).toEqual([]);
  });

  it('resetActiveType drops only the active type from the map', async () => {
    await create({
      defaults: { audio: { embedder: 'wav2vec' }, image: { embedder: 'siglip' } },
    });
    const emits = captureEmits();
    component.resetActiveType(); // active is 'audio'
    expect(emits).toEqual([{ image: { embedder: 'siglip' } }]);
  });

  it('merges a patch into existing overrides for the active type', async () => {
    await create({ defaults: { audio: { embedder: 'wav2vec' } } });
    const emits = captureEmits();
    component.onClipperSelected({ name: 'window', params: {} });
    // Existing embedder override is preserved; empty clipper_params is stripped.
    expect(emits.at(-1)).toEqual({ audio: { embedder: 'wav2vec', clipper: 'window' } });
  });
});
