import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { Subject } from 'rxjs';
import { NewDetectorModalComponent } from './new-detector-modal.component';
import { ProgressEventsService } from '../../../services/progress-events.service';
import type { LoadingTask } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('NewDetectorModalComponent', () => {
  let component: NewDetectorModalComponent;
  let fixture: ComponentFixture<NewDetectorModalComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [NewDetectorModalComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(NewDetectorModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    TestBed.tick(); // run ngOnInit under zoneless (issues the init GETs)

    // Seed importers back the extra Blank-flow example tabs; the family
    // ships no built-ins, so the default answer is an empty roster.
    httpMock.expectOne('/api/seed-importers').flush({ importers: [] });
    // Flush the media types request from ngOnInit
    httpMock.expectOne('/api/media-types').flush({
      media_types: [
        { type_id: 'audio', name: 'Audio', icon: 'audio' },
        { type_id: 'image', name: 'Image', icon: 'image' },
      ],
    });
    // EmbedderCapabilityService.ensureLoaded() in ngOnInit fetches the
    // embedder registry (drives the no-text warning).
    httpMock.expectOne('/api/embedders').flush({
      embedders: [
        { name: 'clap', supports_text: true },
        { name: 'dinov3', supports_text: false },
      ],
    });
    // settingsState.load() in ngOnInit fetches settings.
    TestBed.tick(); // flush the SettingsStateService rxResource loader (root effect)
    httpMock.expectOne('/api/settings').flush({});
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should populate media types from API', () => {
    expect(component.mediaTypes()).toEqual(['audio', 'image']);
  });

  it('should show error when name is empty', () => {
    component.name.set('');
    component.pendingText.set('test');
    component.submit();
    expect(component.error()).toBe('Name is required');
  });

  it('should show error when no example provided', () => {
    component.name.set('Test Model');
    component.pendingText.set('');
    component.submit();
    expect(component.error()).toBe('An example (text or media) is required');
  });

  it('should accept pending text as text example on submit', () => {
    vi.spyOn(component.created, 'emit');

    component.name.set('Dog Barks');
    component.mediaType.set('audio');
    component.pendingText.set('dog barking sounds');
    component.submit();

    const req = httpMock.expectOne('/api/detectors/registry');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.name).toBe('Dog Barks');
    expect(req.request.body.media_type).toBe('audio');
    expect(req.request.body.text_query).toBe('dog barking sounds');
    req.flush({ ok: true, detector: { id: '123', name: 'Dog Barks' } });

    expect(component.created.emit).toHaveBeenCalledWith('123');
  });

  it('should disable Create button when not ready', () => {
    component.name.set('');
    component.pendingText.set('');
    expect(component.canSubmitBlank).toBe(false);

    component.name.set('Test');
    expect(component.canSubmitBlank).toBe(false);

    component.pendingText.set('query');
    expect(component.canSubmitBlank).toBe(true);
  });

  it('should name the blocker in the Create button title when disabled', () => {
    component.mediaType.set('image');
    component.name.set('');
    component.pendingText.set('');
    component.mediaExamples.set([]);
    // Missing example takes precedence in the hint.
    expect(component.blankSubmitTitle).toContain('Provide a text or image example');

    // With an example but no name, the blocker becomes the name.
    component.pendingText.set('query');
    expect(component.blankSubmitTitle).toContain('Enter a detector name');

    // Fully ready → success-oriented title.
    component.name.set('Test');
    expect(component.blankSubmitTitle).toContain('Create the detector with the example');
  });

  it('should default the example tab to text', () => {
    expect(component.exampleTab()).toBe('text');
  });

  it('should switch between text and media example tabs', () => {
    component.setExampleTab('media');
    expect(component.exampleTab()).toBe('media');
    component.setExampleTab('text');
    expect(component.exampleTab()).toBe('text');
  });

  it('should label the media tab from the detector media type', () => {
    component.mediaType.set('image');
    expect(component.exampleMediaTabLabel).toBe('Image');
    component.mediaType.set('audio');
    expect(component.exampleMediaTabLabel).toBe('Audio');
  });

  it('should drop selected media examples when the user types text', () => {
    component.mediaExamples.set([
      { value: 'file.wav', display: 'file.wav', mediaType: 'audio', thumbFailed: false },
      { value: 'file2.wav', display: 'file2.wav', mediaType: 'audio', thumbFailed: false },
    ]);
    component.exampleTab.set('media');
    expect(component.hasMediaExample).toBe(true);

    // Typing a text example is mutually exclusive with the media examples.
    component.onPendingTextInput('dog barking');

    expect(component.hasMediaExample).toBe(false);
    expect(component.pendingText()).toBe('dog barking');
  });

  it('should clear pending text when media example is set', () => {
    component.pendingText.set('some text');
    component.mediaExamples.set([
      { value: 'file.wav', display: 'file.wav', mediaType: 'audio', thumbFailed: false },
    ]);
    component.pendingText.set('');

    expect(component.hasMediaExample).toBe(true);
    expect(component.pendingText()).toBe('');
  });

  it('removes a single example from the stack', () => {
    component.mediaExamples.set([
      { value: 'a.jpg', display: 'a.jpg', mediaType: 'image', thumbFailed: false },
      { value: 'b.jpg', display: 'b.jpg', mediaType: 'image', thumbFailed: false },
      { value: 'c.jpg', display: 'c.jpg', mediaType: 'image', thumbFailed: false },
    ]);

    component.removeMediaExample(1);

    expect(component.mediaExamples().map((e) => e.value)).toEqual(['a.jpg', 'c.jpg']);
    expect(component.hasMediaExample).toBe(true);
  });

  it('submits every stacked media example in the examples payload', () => {
    vi.spyOn(component.created, 'emit');

    component.name.set('Red Cars');
    component.mediaType.set('image');
    component.mediaExamples.set([
      { value: 'a.jpg', display: 'a.jpg', mediaType: 'image', thumbFailed: false },
      { value: 'b.jpg', display: 'b.jpg', mediaType: 'image', thumbFailed: false },
    ]);
    component.submit();

    const req = httpMock.expectOne('/api/detectors/registry');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.media_example).toBe('a.jpg');
    expect(req.request.body.examples).toEqual([
      { type: 'media', value: 'a.jpg' },
      { type: 'media', value: 'b.jpg' },
    ]);
    req.flush({ ok: true, detector: { id: '456', name: 'Red Cars' } });

    expect(component.created.emit).toHaveBeenCalledWith('456');
  });

  it('marks only the failing thumbnail as failed', () => {
    component.mediaExamples.set([
      { value: 'a.jpg', display: 'a.jpg', mediaType: 'image', thumbFailed: false },
      { value: 'b.jpg', display: 'b.jpg', mediaType: 'image', thumbFailed: false },
    ]);

    component.onExampleThumbError(0);

    expect(component.exampleThumbnailUrl(component.mediaExamples()[0])).toBeNull();
    expect(component.exampleThumbnailUrl(component.mediaExamples()[1])).toBe(
      '/api/server-media-files/b.jpg/thumbnail',
    );
  });

  it('should show server error on failure', () => {
    component.name.set('Test');
    component.pendingText.set('test');
    component.submit();

    httpMock.expectOne('/api/detectors/registry').flush(
      { error: 'Detector already exists' },
      { status: 409, statusText: 'Conflict' },
    );

    expect(component.error()).toBe('Detector already exists');
  });

  it('should return media type icon', () => {
    expect(component.getMediaTypeIcon('audio')).toBe('audio');
    expect(component.getMediaTypeIcon('image')).toBe('image');
    expect(component.getMediaTypeIcon('unknown')).toBe('');
  });

  it('should emit closed on close', () => {
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should not lock media type when no default is provided', () => {
    expect(component.mediaTypeLocked).toBe(false);
  });

  it('should ignore toggleMediaTypeDropdown when locked', () => {
    component.mediaTypeLocked = true;
    component.mediaTypeDropdownOpen = false;
    component.toggleMediaTypeDropdown();
    expect(component.mediaTypeDropdownOpen).toBe(false);
  });

  it('should open dropdown via toggle when unlocked', () => {
    component.mediaTypeLocked = false;
    component.mediaTypeDropdownOpen = false;
    component.toggleMediaTypeDropdown();
    expect(component.mediaTypeDropdownOpen).toBe(true);
  });

  it('should unlock media type on explicit unlock', () => {
    component.mediaTypeLocked = true;
    component.unlockMediaType();
    expect(component.mediaTypeLocked).toBe(false);
  });

  it('pre-fills the name from the text seed while the name is untouched', () => {
    component.onPendingTextInput('dog barking sounds');
    expect(component.pendingText()).toBe('dog barking sounds');
    expect(component.name()).toBe('dog barking sounds');
  });

  it('sanitises pasted whitespace when mirroring the seed into the name', () => {
    component.onPendingTextInput('   dog   barking\n  sounds   ');
    expect(component.pendingText()).toBe('   dog   barking\n  sounds   ');
    expect(component.name()).toBe('dog barking sounds');
  });

  it('warns about no-text datasets only for a text-hint-only detector', () => {
    // No warning before any text is entered.
    fixture.componentRef.setInput('datasetEmbedder', 'dinov3');
    component.pendingText.set('');
    expect(component.showNoTextWarning).toBe(false);

    // Text entered against a no-text embedder → warn.
    component.pendingText.set('a red car');
    expect(component.showNoTextWarning).toBe(true);

    // A media example seed (not text-only) → no warning even on a no-text dataset.
    component.mediaExamples.set([
      { value: 'file.jpg', display: 'file.jpg', mediaType: 'image', thumbFailed: false },
    ]);
    expect(component.showNoTextWarning).toBe(false);
  });

  it('does not warn when the dataset embedder can search by text', () => {
    fixture.componentRef.setInput('datasetEmbedder', 'clap');
    component.pendingText.set('dog barking');
    expect(component.showNoTextWarning).toBe(false);
  });

  it('does not warn when the active dataset embedder is unknown', () => {
    fixture.componentRef.setInput('datasetEmbedder', '');
    component.pendingText.set('dog barking');
    expect(component.showNoTextWarning).toBe(false);
  });

  it('stops mirroring once the user edits the name', () => {
    component.onPendingTextInput('dog');
    expect(component.name()).toBe('dog');

    component.onNameInput('Dog Barks');
    expect(component.name()).toBe('Dog Barks');
    expect(component.nameTouched).toBe(true);

    component.onPendingTextInput('cat meowing');
    expect(component.pendingText()).toBe('cat meowing');
    expect(component.name()).toBe('Dog Barks');
  });

  it('auto-selects the lone importer when a single-source category is opened', () => {
    // "Files" has exactly one importer (the server_file datasource
    // importer), so opening it should skip the redundant one-button
    // sub-tab bar and land on its form.
    component.datasourceImporters.set([
      { name: 'server_file', category: 'server' },
    ]);
    component.mediaImporters.set([
      { name: 'demo', picker_view: 'demo', category: 'demo' },
    ]);

    component.selectImporterTab('server');

    expect(component.selectedImporter?.name).toBe('server_file');
    expect(component.selectedDatasourceImporter?.name).toBe('server_file');
  });

  it('does not auto-select when a category holds more than one importer', () => {
    component.datasourceImporters.set([
      { name: 'server_file', category: 'server' },
      { name: 'server_other', category: 'server' },
    ]);

    component.selectImporterTab('server');

    expect(component.selectedImporter).toBeNull();
  });

  // --- Datasource importers in the media picker (issue #2767) ---

  it('merges datasource importers into the picker after the dataset browse views', () => {
    component.mediaImporters.set([
      { name: 'local_files', picker_view: 'local_files', category: 'local' },
      { name: 'demo', picker_view: 'demo', category: 'demo' },
    ]);
    component.datasourceImporters.set([
      { name: 'server_file', category: 'server' },
      { name: 'url_download', category: 'services' },
      { name: 'my_cloud_thing', category: 'services' },
    ]);

    expect(component.orderedImporters.map((i) => i.name)).toEqual([
      'local_files',
      'server_file',
      'demo',
      'url_download',
      'my_cloud_thing',
    ]);
  });

  it('reports an empty picker view for a selected datasource importer', () => {
    const dsImporter = { name: 'url_download', category: 'services' };
    component.datasourceImporters.set([dsImporter]);

    component.selectImporter(dsImporter);

    // The source picker renders none of its dedicated widgets; the modal
    // renders the importer's dynamic form instead.
    expect(component.activePickerView).toBe('');
    expect(component.selectedDatasourceImporter).toBe(dsImporter);
  });

  it('routes a dataset importer that shares a datasource importer name by identity', () => {
    const datasetDemo = { name: 'demo', picker_view: 'demo', category: 'demo' };
    component.mediaImporters.set([datasetDemo]);
    component.datasourceImporters.set([{ name: 'demo', category: 'services' }]);

    expect(component.isDatasourceImporter(datasetDemo)).toBe(false);
  });

  it('adds the fetched item to the example stack and returns to the form', () => {
    component.view.set('media-picker');
    component.mediaType.set('');

    component.onDatasourceImported({ filename: 'abc123.wav', original_name: 'bark.wav' });

    expect(component.mediaExamples().map((e) => e.value)).toEqual(['abc123.wav']);
    expect(component.mediaExamples()[0].display).toBe('bark.wav');
    // Media type is inferred from the original filename's extension.
    expect(component.mediaExamples()[0].mediaType).toBe('audio');
    // No origin reported → none stored (the example seeds via the sentinel).
    expect(component.mediaExamples()[0].origin).toBeNull();
    expect(component.view()).toBe('main');
  });

  // --- Durable example origins (issue #2774) ---

  it('keeps the datasource item origin and sends it with the examples payload', () => {
    const origin = { importer: 'url_download', params: { url: 'https://x.test/bark.wav' } };
    component.view.set('media-picker');
    component.mediaType.set('audio');

    component.onDatasourceImported({ filename: 'abc123.wav', original_name: 'bark.wav', origin });

    expect(component.mediaExamples()[0].origin).toEqual(origin);

    component.name.set('Barks');
    component.submit();

    const req = httpMock.expectOne('/api/detectors/registry');
    expect(req.request.body.examples).toEqual([{ type: 'media', value: 'abc123.wav', origin }]);
    req.flush({ ok: true, detector: { id: '789', name: 'Barks' } });
  });

  // --- Seed importers: extra Blank-flow example tabs (issue #3140) ---

  it('adds no example tabs when no seed importer is registered', () => {
    expect(component.seedImporters()).toEqual([]);
    expect(component.activeSeedImporter).toBeNull();
  });

  it('appends an imported batch to the stack as unlabeled seeds', () => {
    component.mediaType.set('');

    component.onSeedsImported({
      count: 2,
      truncated: false,
      items: [
        { filename: 'aaa.wav', original_name: 'near-miss-1.wav', origin: null },
        { filename: 'bbb.wav', original_name: 'near-miss-2.wav', origin: null },
      ],
    });

    expect(component.mediaExamples().map((e) => e.value)).toEqual(['aaa.wav', 'bbb.wav']);
    expect(component.mediaExamples().every((e) => e.seed)).toBe(true);
    // Media type is inferred from the original filename's extension.
    expect(component.mediaExamples()[0].mediaType).toBe('audio');
    // The stack is where the user prunes what arrived, so land them on it.
    expect(component.exampleTab()).toBe('media');
    expect(component.seedNotice()).toBe('Added 2 seeds.');
  });

  it('says so when a batch comes back empty or truncated', () => {
    component.onSeedsImported({ count: 0, truncated: false, items: [] });
    expect(component.mediaExamples()).toEqual([]);
    expect(component.seedNotice()).toContain('no seeds');

    component.onSeedsImported({
      count: 1,
      truncated: true,
      items: [{ filename: 'aaa.wav', original_name: 'a.wav' }],
    });
    expect(component.seedNotice()).toContain("importer's limit");
  });

  it('marks seed examples labeled:false in the create payload', () => {
    const origin = { importer: 'holder', params: { cluster: 'c1' } };
    component.name.set('Near misses');
    component.mediaType.set('image');
    component.mediaExamples.set([
      { value: 'hand.jpg', display: 'hand.jpg', mediaType: 'image', thumbFailed: false },
      { value: 'seed.jpg', display: 'seed.jpg', mediaType: 'image', thumbFailed: false, origin, seed: true },
    ]);

    component.submit();

    const req = httpMock.expectOne('/api/detectors/registry');
    expect(req.request.body.examples).toEqual([
      { type: 'media', value: 'hand.jpg' },
      { type: 'media', value: 'seed.jpg', origin, labeled: false },
    ]);
    req.flush({ ok: true, detector: { id: '901', name: 'Near misses' } });
  });

  it('prefers a hand-picked exemplar over a seed for the legacy scalar', () => {
    component.name.set('Mixed');
    component.mediaType.set('image');
    component.mediaExamples.set([
      { value: 'seed.jpg', display: 'seed.jpg', mediaType: 'image', thumbFailed: false, seed: true },
      { value: 'hand.jpg', display: 'hand.jpg', mediaType: 'image', thumbFailed: false },
    ]);

    component.submit();

    const req = httpMock.expectOne('/api/detectors/registry');
    expect(req.request.body.media_example).toBe('hand.jpg');
    req.flush({ ok: true, detector: { id: '902', name: 'Mixed' } });
  });

  it('does not auto-name the detector after a seed', () => {
    component.onSeedsImported({
      count: 1,
      truncated: false,
      items: [{ filename: 'aaa.wav', original_name: 'near-miss.wav' }],
    });
    // A seed is "close but not quite", so naming after it would be wrong.
    expect(component.name()).toBe('');

    // A hand-picked exemplar arriving later still names it.
    component.onDatasourceImported({ filename: 'bbb.wav', original_name: 'bark.wav' });
    expect(component.name()).toBe('bark');
  });

  it('surfaces registered seed importers as example tabs', () => {
    component.seedImporters.set([
      { name: 'holder', display_name: 'Holder', description: 'Near misses' } as never,
    ]);

    component.setExampleTab('holder');

    expect(component.exampleTab()).toBe('holder');
    expect(component.activeSeedImporter?.name).toBe('holder');

    // Switching back to a stock tab drops the plugin panel and its notice.
    component.seedNotice.set('Added 2 seeds.');
    component.setExampleTab('text');
    expect(component.activeSeedImporter).toBeNull();
    expect(component.seedNotice()).toBe('');
  });

  // --- Trained tab: label-importer plugin field parity (issue #2597) ---

  it('fetches dynamic options when a Trained-tab importer is selected', () => {
    const field = { key: 'sheet', field_type: 'select', dynamic_options: true, required: true } as any;
    component.selectLabelImporter({ name: 'gsheets', fields: [field] } as any);

    const req = httpMock.expectOne('/api/label-importers/field-options/gsheets');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.field_key).toBe('sheet');
    req.flush({ options: [{ value: 's1', label: 'Sheet 1' }] });

    expect(component.labelImporterFieldOptions.optionsFor(field)).toEqual([{ value: 's1', label: 'Sheet 1' }]);
    // A required dynamic select auto-selects the first option.
    expect(component.labelImporterValues['sheet']).toBe('s1');
  });

  it('does not auto-select the first option for a required free-text combobox', () => {
    const field = {
      key: 'sheet',
      field_type: 'select',
      dynamic_options: true,
      required: true,
      allow_free_text: true,
    } as any;
    component.selectLabelImporter({ name: 'gsheets', fields: [field] } as any);

    httpMock
      .expectOne('/api/label-importers/field-options/gsheets')
      .flush({ options: [{ value: 's1', label: 'Sheet 1' }] });

    expect(component.labelImporterValues['sheet']).toBeUndefined();
  });

  it('does not auto-select the first static option for a free-text combobox', () => {
    const field = { key: 's', field_type: 'select', options: ['x', 'y'], allow_free_text: true } as any;
    component.selectLabelImporter({ name: 'imp', fields: [field] } as any);

    expect(component.labelImporterValues['s']).toBeUndefined();
  });

  it('coerces static string options into {value,label} pairs on the Trained tab', () => {
    const staticSelect = { key: 's', field_type: 'select', options: ['x', 'y'] } as any;
    expect(component.labelImporterFieldOptions.optionsFor(staticSelect)).toEqual([
      { value: 'x', label: 'x' },
      { value: 'y', label: 'y' },
    ]);
  });

  it('re-fetches a dependent field when its dependency changes', () => {
    const parent = { key: 'doc', field_type: 'select', dynamic_options: true } as any;
    const child = { key: 'tab', field_type: 'select', dynamic_options: true, depends_on: ['doc'] } as any;
    component.selectLabelImporter({ name: 'gsheets', fields: [parent, child] } as any);

    // Both dynamic fields fetch on select.
    httpMock.match('/api/label-importers/field-options/gsheets').forEach((r) => r.flush({ options: [] }));

    component.labelImporterValues['doc'] = 'doc-1';
    component.onLabelImporterFieldChanged('doc');

    // Only the dependent child re-fetches; its stale value is blanked first.
    expect(component.labelImporterValues['tab']).toBe('');
    const req = httpMock.expectOne('/api/label-importers/field-options/gsheets');
    expect(req.request.body.field_key).toBe('tab');
    req.flush({ options: [{ value: 't1', label: 'Tab 1' }] });
  });

  it('keeps a typed free-text value the refreshed options omit on the Trained tab', () => {
    const field = { key: 'q', field_type: 'select', dynamic_options: true, allow_free_text: true } as any;
    component.selectedLabelImporter = { name: 'imp', fields: [field] } as any;
    component.labelImporterValues['q'] = 'hand-typed';
    component.labelImporterFieldOptions.refresh(field, component.labelImporterValues);
    httpMock
      .expectOne('/api/label-importers/field-options/imp')
      .flush({ options: [{ value: 'a', label: 'A' }] });
    expect(component.labelImporterValues['q']).toBe('hand-typed');
  });

  it('clears a strict-select value the refreshed options omit on the Trained tab', () => {
    const field = { key: 'q', field_type: 'select', dynamic_options: true } as any;
    component.selectedLabelImporter = { name: 'imp', fields: [field] } as any;
    component.labelImporterValues['q'] = 'stale';
    component.labelImporterFieldOptions.refresh(field, component.labelImporterValues);
    httpMock
      .expectOne('/api/label-importers/field-options/imp')
      .flush({ options: [{ value: 'a', label: 'A' }] });
    expect(component.labelImporterValues['q']).toBe('');
  });

  it('surfaces a dynamic-option fetch error on the Trained tab', () => {
    const field = { key: 'q', field_type: 'select', dynamic_options: true } as any;
    component.selectedLabelImporter = { name: 'imp', fields: [field] } as any;
    component.labelImporterFieldOptions.refresh(field, component.labelImporterValues);
    httpMock
      .expectOne('/api/label-importers/field-options/imp')
      .flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' });
    expect(component.labelImporterFieldOptions.error()['q']).toBe('boom');
    expect(component.labelImporterFieldOptions.optionsFor(field)).toEqual([]);
  });

  // --- Create & Import: the background labelset-media ingest (#2703) ---

  /** Fill in the Trained tab's form and submit it. */
  function submitTrained(): void {
    component.tab = 'trained';
    component.trainedView = 'form';
    component.selectedLabelImporter = { name: 'server_json_file' } as any;
    component.name.set('Imported');
    component.submit();
  }

  it('waits for the labelset-media ingest before loading the new detector', () => {
    const feed = new Subject<LoadingTask>();
    const spy = vi
      .spyOn(TestBed.inject(ProgressEventsService), 'detectorTaskUntilDone$')
      .mockReturnValue(feed.asObservable());
    vi.spyOn(component.created, 'emit');

    submitTrained();
    httpMock
      .expectOne('/api/detectors/registry/from-labelset/server_json_file')
      .flush({ ok: true, detector: { id: 'd9' }, ingest_task_id: '_detingest_d9' });

    expect(spy).toHaveBeenCalledWith('_detingest_d9');
    // Loading now would restore the labels against media that aren't in the
    // dataset yet (#2690), so the load must not have gone out.
    httpMock.expectNone('/api/detectors/registry/load');
    expect(component.submitting()).toBe(true);

    feed.next({ task_id: '_detingest_d9', status: 'loading', current: 1, total: 2 } as LoadingTask);
    expect(component.ingestTask()?.current).toBe(1);
    expect(component.ingestBar).toEqual({ value: 1, max: 2, indeterminate: false });
    httpMock.expectNone('/api/detectors/registry/load');

    feed.complete();
    const load = httpMock.expectOne('/api/detectors/registry/load');
    expect(load.request.body.detector_id).toBe('d9');
    load.flush({ ok: true });

    expect(component.ingestTask()).toBeNull();
    expect(component.submitting()).toBe(false);
    expect(component.created.emit).toHaveBeenCalledWith('d9');
  });

  it('loads straight away when the import had nothing to ingest', () => {
    const spy = vi.spyOn(TestBed.inject(ProgressEventsService), 'detectorTaskUntilDone$');
    vi.spyOn(component.created, 'emit');

    submitTrained();
    httpMock
      .expectOne('/api/detectors/registry/from-labelset/server_json_file')
      .flush({ ok: true, detector: { id: 'd0' }, ingest_task_id: '' });

    expect(spy).not.toHaveBeenCalled();
    httpMock.expectOne('/api/detectors/registry/load').flush({ ok: true });
    expect(component.created.emit).toHaveBeenCalledWith('d0');
  });

  // --- Double-submit guards (#2941) ---

  it('ignores a second submit while the register POST is in flight', () => {
    component.name.set('Dog Barks');
    component.mediaType.set('audio');
    component.pendingText.set('dog barking sounds');

    // Enter in the text field, then Enter again (or in the name field) before
    // the first POST resolves.
    component.submit();
    component.submit();

    // One detector, not two.
    const req = httpMock.expectOne('/api/detectors/registry');
    req.flush({ ok: true, detector: { id: '123' } });
  });

  it('ignores a second Trained submit while the from-labelset POST is in flight', () => {
    submitTrained();
    component.submit();

    const req = httpMock.expectOne('/api/detectors/registry/from-labelset/server_json_file');
    req.flush({ ok: true, detector: { id: 'd1' }, ingest_task_id: '' });
    httpMock.expectOne('/api/detectors/registry/load').flush({ ok: true });
  });

  it('ignores a second demo-path load while the select POST is in flight', () => {
    component.mediaType.set('audio');
    component.demoFileBrowseSource = 'demo:gtzan';
    component.demoTypedPath = 'blues/blues.00000.wav';

    component.submitDemoTypedPath();
    component.submitDemoTypedPath();

    // One materialisation, so one example row — a duplicate would collide the
    // @for track key on example.value.
    const req = httpMock.expectOne('/api/browse-media-files/select');
    req.flush({ ok: true, filename: 'ex.wav', original_name: 'blues.00000.wav' });
    expect(component.mediaExamples().length).toBe(1);
  });
});

describe('NewDetectorModalComponent with defaultMediaType', () => {
  let component: NewDetectorModalComponent;
  let fixture: ComponentFixture<NewDetectorModalComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [NewDetectorModalComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(NewDetectorModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('defaultMediaType', 'image');
    httpMock = TestBed.inject(HttpTestingController);
    TestBed.tick(); // run ngOnInit under zoneless (issues the init GETs)

    httpMock.expectOne('/api/seed-importers').flush({ importers: [] });
    httpMock.expectOne('/api/media-types').flush({
      media_types: [
        { type_id: 'audio', name: 'Audio', icon: 'audio' },
        { type_id: 'image', name: 'Image', icon: 'image' },
      ],
    });
    httpMock.expectOne('/api/embedders').flush({ embedders: [] });
    // settingsState.load() in ngOnInit fetches settings.
    TestBed.tick(); // flush the SettingsStateService rxResource loader (root effect)
    httpMock.expectOne('/api/settings').flush({});
    // No /api/datasets/registry call when defaultMediaType is provided.
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should lock media type to the active dataset type', () => {
    expect(component.mediaType()).toBe('image');
    expect(component.mediaTypeLocked).toBe(true);
  });
});

describe('NewDetectorModalComponent (semantic_only server)', () => {
  let component: NewDetectorModalComponent;
  let fixture: ComponentFixture<NewDetectorModalComponent>;
  let httpMock: HttpTestingController;

  async function setup(semanticOnly: boolean) {
    await TestBed.configureTestingModule({
      imports: [NewDetectorModalComponent],
      providers: [...provideZoneless(), ...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(NewDetectorModalComponent);
    component = fixture.componentInstance;
    fixture.componentRef.setInput('defaultMediaType', 'image');
    httpMock = TestBed.inject(HttpTestingController);
    TestBed.tick();

    httpMock.expectOne('/api/seed-importers').flush({ importers: [] });
    httpMock.expectOne('/api/media-types').flush({
      media_types: [{ type_id: 'image', name: 'Image', icon: 'image' }],
    });
    httpMock.expectOne('/api/embedders').flush({ embedders: [{ name: 'siglip', supports_text: true }] });
    TestBed.tick();
    httpMock.expectOne('/api/settings').flush({ semantic_only: semanticOnly });
    TestBed.tick();
  }

  afterEach(() => {
    httpMock.verify();
  });

  it('offers all three types and shows the Advanced toggle when unlocked', async () => {
    await setup(false);
    expect(component.embedderTypeOptions).toEqual(['semantic', 'patch_semantic', 'structural']);
    expect(component.showEmbedderTypePicker).toBe(true);
    expect(component.showAdvancedToggle).toBe(true);
  });

  it('drops the one-option type picker (and the Advanced toggle) when locked', async () => {
    await setup(true);
    expect(component.embedderTypeOptions).toEqual(['semantic']);
    expect(component.showEmbedderTypePicker).toBe(false);
    // Nothing else lives under Advanced for this dataset, so the toggle goes too.
    expect(component.primaryLicenseNotice).toBeNull();
    expect(component.showAdvancedToggle).toBe(false);
  });

  it('pins the created detector to semantic when locked', async () => {
    await setup(true);
    // Even a stale explicit pick can't escape the lock.
    component.onEmbedderTypeChange('structural');
    expect(component.effectiveEmbedderType).toBe('semantic');
  });
});
