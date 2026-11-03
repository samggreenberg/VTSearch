import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { DemoPickerComponent } from './demo-picker.component';
import { provideZoneless } from '../../../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../../../testing/test-providers';
import { settleZoneless } from '../../../../../testing/settle-resource';

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
    let started = false;
    component.importStarted.subscribe(() => (started = true));

    component.selectDemo(mockDemos[0] as any);
    expect(component.selectedDemo()).toBe(mockDemos[0] as any);
    expect(component.demoDatasetName()).toBe('GTZAN');
    httpMock.expectNone('/api/dataset/load-demo');
    expect(started).toBe(false);
  });

  // Zoneless oracle (docs/FRONTEND.md, *Testing*): the parent calls these
  // from listeners on a sibling component, so nothing marks this view dirty
  // but its own signal writes (#5300).
  it('paints the picked row\'s name, and a tab switch clearing it, without a forced render', async () => {
    openAndFlush();
    await settleZoneless(fixture);
    const name = () => fixture.nativeElement.querySelector('#demo-dataset-name') as HTMLInputElement;

    component.selectDemo(mockDemos[0] as any);
    await settleZoneless(fixture);
    expect(name().value).toBe('GTZAN');

    // Asserted before the tab's option lists answer, whose signal writes
    // would repaint the view on their own.
    component.selectDemoTabWithEmbedder('audio');
    await settleZoneless(fixture);
    expect(name().value).toBe('');

    httpMock.expectOne(req => req.url === '/api/embedders' && req.params.get('media_type') === 'audio').flush({ embedders: [{ name: 'clap' }] });
    httpMock.expectOne(req => req.url === '/api/clippers' && req.params.get('media_type') === 'audio').flush({ clippers: [] });
    httpMock.expectOne(req => req.url === '/api/cleaners' && req.params.get('media_type') === 'audio').flush({ cleaners: [] });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'clap').flush({ datasets: mockDemos });
    httpMock.expectOne(req => req.url === '/api/converters' && req.params.get('source') === 'audio').flush({ converters: [] });
  });

  // #5498: the picker starts the import itself and the modal closes only on
  // the server's answer, so the wait shows as "Importing…" rather than as
  // nothing, and a second click cannot start a second import.
  describe('submit()', () => {
    it('POSTs load-demo with the composed body, and announces the start only once the server answers', () => {
      openAndFlush();
      component.selectDemo(mockDemos[0] as any);
      let started = false;
      component.importStarted.subscribe(() => (started = true));

      component.submit();
      const req = httpMock.expectOne('/api/dataset/load-demo');
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({ name: 'gtzan', dataset_name: 'GTZAN', embedder: 'clap' });
      expect(component.submitting()).toBe(true);
      expect(started).toBe(false);

      req.flush({ ok: true, message: 'Loading started', task_id: '_loading_1', task_ids: ['_loading_1'] });
      expect(component.submitting()).toBe(false);
      expect(started).toBe(true);
    });

    it('sends the ingest toggles and the AutoFind choice', () => {
      fixture.componentRef.setInput('buildProjection', true);
      fixture.componentRef.setInput('mergeNearDuplicates', true);
      fixture.componentRef.setInput('autofind', false);
      openAndFlush();
      component.selectDemo(mockDemos[0] as any);

      component.submit();
      const body = httpMock.expectOne('/api/dataset/load-demo').request.body;
      expect(body.build_projection).toBe('true');
      expect(body.merge_near_duplicates).toBe('true');
      expect(body.autofind).toBe('false');
    });

    it('ignores a second click while the first is out', () => {
      openAndFlush();
      component.selectDemo(mockDemos[0] as any);

      component.submit();
      component.submit();
      httpMock.expectOne('/api/dataset/load-demo').flush({ task_id: '_loading_1' });
    });

    it('keeps the modal open on a refusal, saying why, and lets the user try again', () => {
      openAndFlush();
      component.selectDemo(mockDemos[0] as any);
      let started = false;
      component.importStarted.subscribe(() => (started = true));

      component.submit();
      httpMock
        .expectOne('/api/dataset/load-demo')
        .flush({ message: 'Invalid dataset name' }, { status: 400, statusText: 'Bad Request' });
      expect(started).toBe(false);
      expect(component.submitting()).toBe(false);
      expect(component.submitError()).toBe('Invalid dataset name');

      component.submit();
      expect(component.submitError()).toBe('');
      httpMock.expectOne('/api/dataset/load-demo').flush({ task_id: '_loading_2' });
      expect(started).toBe(true);
    });

    it('paints the refusal under the form', async () => {
      openAndFlush();
      component.selectDemo(mockDemos[0] as any);
      component.submit();
      httpMock
        .expectOne('/api/dataset/load-demo')
        .flush({ message: 'Invalid dataset name' }, { status: 400, statusText: 'Bad Request' });
      await settleZoneless(fixture);
      expect(fixture.nativeElement.querySelector('.error-text')?.textContent).toContain('Invalid dataset name');
    });

    it('does nothing with no selection', () => {
      openAndFlush();
      component.submit();
      httpMock.expectNone('/api/dataset/load-demo');
      expect(component.submitting()).toBe(false);
    });
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
    component.submit();

    const body = httpMock.expectOne('/api/dataset/load-demo').request.body;
    expect(body.converter).toBe('image2face');
    expect(body.embedder).toBe('face');
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
      component.submit();

      const payload = httpMock.expectOne('/api/dataset/load-demo').request.body;
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
