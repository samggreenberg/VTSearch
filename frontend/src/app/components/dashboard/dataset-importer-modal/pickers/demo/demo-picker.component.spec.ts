import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { DemoPickerComponent } from './demo-picker.component';
import { provideZoneless } from '../../../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../../../testing/test-providers';

describe('DemoPickerComponent', () => {
  let component: DemoPickerComponent;
  let fixture: ComponentFixture<DemoPickerComponent>;
  let httpMock: HttpTestingController;

  const demoImporter = { name: 'demo', picker_view: 'demo', fields: [] } as any;
  const mockMediaTypes = [
    { type_id: 'audio', name: 'Audio', icon: 'audio' },
    { type_id: 'image', name: 'Image', icon: 'image' },
  ];
  const mockDemos = [
    { name: 'gtzan', label: 'GTZAN', status: 'needs_download', ready: false, num_files: 1000, media_type: 'audio', pkl_embedder: '' },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DemoPickerComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DemoPickerComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  function openAndFlush(): void {
    component.open(demoImporter);
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && !req.params.get('embedder')).flush({ datasets: mockDemos });
    httpMock.expectOne(req => req.url === '/api/embedders' && req.params.get('media_type') === 'audio').flush({ embedders: [{ name: 'clap' }] });
    httpMock.expectOne(req => req.url === '/api/clippers' && req.params.get('media_type') === 'audio').flush({ clippers: [] });
    httpMock.expectOne(req => req.url === '/api/cleaners' && req.params.get('media_type') === 'audio').flush({ cleaners: [] });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'clap').flush({ datasets: mockDemos });
    httpMock.expectOne(req => req.url === '/api/converters' && req.params.get('source') === 'audio').flush({ converters: [] });
  }

  it('open() fetches media types + demos and auto-selects the audio tab', () => {
    openAndFlush();
    expect(component.demoTabs()).toEqual(['audio']);
    expect(component.activeTab()).toBe('audio');
    expect(component.demos().length).toBe(1);
  });

  it('selecting a row records the selection without submitting', () => {
    openAndFlush();
    let emitted = false;
    component.demoDatasetSelected.subscribe(() => (emitted = true));

    component.selectDemo(mockDemos[0] as any);
    expect(component.selectedDemo()).toBe(mockDemos[0] as any);
    expect(component.demoDatasetName).toBe('GTZAN');
    expect(emitted).toBe(false);
  });

  it('submit() emits demoDatasetSelected with the composed payload', () => {
    openAndFlush();
    component.selectDemo(mockDemos[0] as any);

    let payload: any = null;
    component.demoDatasetSelected.subscribe((d) => (payload = d));
    component.submit();

    expect(payload).toBeTruthy();
    expect(payload.name).toBe('gtzan');
    expect(payload.dataset_name).toBe('GTZAN');
  });

  it('does nothing on submit() with no selection', () => {
    openAndFlush();
    let emitted = false;
    component.demoDatasetSelected.subscribe(() => (emitted = true));
    component.submit();
    expect(emitted).toBe(false);
  });

  it('offers an optional "Convert to" target (e.g. Face) for an embeddable tab with a registered converter', () => {
    openAndFlush();

    component.selectDemoTabWithEmbedder('image');
    httpMock.expectOne(req => req.url === '/api/embedders' && req.params.get('media_type') === 'image').flush({ embedders: [{ name: 'siglip' }] });
    httpMock.expectOne(req => req.url === '/api/clippers' && req.params.get('media_type') === 'image').flush({ clippers: [] });
    httpMock.expectOne(req => req.url === '/api/cleaners' && req.params.get('media_type') === 'image').flush({ cleaners: [] });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'siglip').flush({ datasets: mockDemos });
    httpMock.expectOne(req => req.url === '/api/converters' && req.params.get('source') === 'image')
      .flush({ converters: [{ name: 'image2face', source_type: 'image', target_type: 'face' }] });

    expect(component.activeTabConvertsToOptional()).toBe(true);
    expect(component.activeTabConvertsTo()).toEqual(['face']);
    expect(component.selectedConverterTarget()).toBe('');

    component.onConverterTargetChange('face');
    httpMock.expectOne(req => req.url === '/api/embedders' && req.params.get('media_type') === 'face').flush({ embedders: [{ name: 'face' }] });
    httpMock.expectOne(req => req.url === '/api/clippers' && req.params.get('media_type') === 'face').flush({ clippers: [] });
    httpMock.expectOne(req => req.url === '/api/cleaners' && req.params.get('media_type') === 'face').flush({ cleaners: [] });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'face').flush({ datasets: mockDemos });

    expect(component.selectedConverterTarget()).toBe('face');

    component.selectDemo(mockDemos[0] as any);
    let payload: any = null;
    component.demoDatasetSelected.subscribe((d) => (payload = d));
    component.submit();

    expect(payload.converter).toBe('image2face');
    expect(payload.embedder).toBe('face');
  });

  describe('Multi-Dataset: a demo as linked datasets', () => {
    const celebs = { name: 'celebrity_together_s', label: 'Celebrities in Context (S)', media_type: 'image', status: 'ready', ready: true, num_files: 1200 };

    /** Open the Image tab, whose demos the image2face converter can also load as faces. */
    function openImageTab(): void {
      openAndFlush();
      component.selectDemoTabWithEmbedder('image');
      httpMock.expectOne(req => req.url === '/api/embedders' && req.params.get('media_type') === 'image').flush({ embedders: [{ name: 'siglip' }] });
      httpMock.expectOne(req => req.url === '/api/clippers' && req.params.get('media_type') === 'image').flush({ clippers: [] });
      httpMock.expectOne(req => req.url === '/api/cleaners' && req.params.get('media_type') === 'image').flush({ cleaners: [] });
      httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'siglip').flush({ datasets: mockDemos });
      httpMock.expectOne(req => req.url === '/api/converters' && req.params.get('source') === 'image')
        .flush({ converters: [{ name: 'image2face', source_type: 'image', target_type: 'face' }] });
    }

    const ticked = () => component.outputDrafts().filter((d) => d.checked).map((d) => d.category);

    /** Answer every pending option / status fetch with an empty listing of its own shape. */
    function flushPending(): void {
      const empty: Record<string, object> = {
        '/api/embedders': { embedders: [] },
        '/api/clippers': { clippers: [] },
        '/api/cleaners': { cleaners: [] },
        '/api/converters': { converters: [] },
        '/api/dataset/demo-list': { datasets: mockDemos },
      };
      httpMock.match(() => true).forEach((req) => req.flush(empty[req.request.url] ?? {}));
    }

    it('is offered only on a tab whose demos load as more than one type', () => {
      openAndFlush();
      expect(component.supportsMultiOutput).toBe(false);
      openImageTab();
      expect(component.supportsMultiOutput).toBe(true);
    });

    it('starts from the demo own type and sends one output per ticked row', () => {
      openImageTab();
      component.setMultiDataset(true);
      expect(component.multiDataset()).toBe(true);
      expect(component.outputDrafts().map((d) => d.category)).toEqual(['image', 'face']);
      expect(ticked()).toEqual(['image']);

      component.onOutputDraftsChange(component.outputDrafts().map((d) => ({ ...d, checked: true })));
      component.selectDemo(celebs as any);
      let payload: any = null;
      component.demoDatasetSelected.subscribe((d) => (payload = d));
      component.submit();

      expect(payload.name).toBe('celebrity_together_s');
      expect(payload.outputs).toEqual([
        { media_type: 'image', category: 'image', source_specs: [{ source_type: 'image', converter: null, params: {} }] },
        { media_type: 'face', category: 'face', source_specs: [{ source_type: 'image', converter: 'image2face', params: {} }] },
      ]);
      // The single-dataset choices stay home.
      expect(payload.embedder).toBeUndefined();
      expect(payload.converter).toBeUndefined();
    });

    it('ticks what the single form was converting to beside the demo own type', () => {
      openImageTab();
      component.onConverterTargetChange('face');
      flushPending();
      component.setMultiDataset(true);
      expect(ticked()).toEqual(['image', 'face']);
    });

    it('has nothing to import with every row unticked', () => {
      openImageTab();
      component.setMultiDataset(true);
      component.onOutputDraftsChange(component.outputDrafts().map((d) => ({ ...d, checked: false })));
      expect(component.hasOutputs).toBe(false);
    });

    it('a new tab starts from one dataset again', () => {
      openImageTab();
      component.setMultiDataset(true);
      component.selectDemoTabWithEmbedder('audio');
      flushPending();
      expect(component.multiDataset()).toBe(false);
      expect(component.outputDrafts()).toEqual([]);
    });
  });
});
