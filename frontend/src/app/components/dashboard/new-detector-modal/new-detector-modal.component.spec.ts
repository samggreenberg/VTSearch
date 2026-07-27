import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { NewDetectorModalComponent } from './new-detector-modal.component';
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
    // "Files" has exactly one importer (server_folder), so opening it should
    // skip the redundant one-button sub-tab bar and land on the path input.
    component.mediaImporters.set([
      { name: 'server_folder', picker_view: 'server_folder', category: 'server' },
      { name: 'demo', picker_view: 'demo', category: 'demo' },
    ]);

    component.selectImporterTab('server');

    expect(component.selectedImporter?.name).toBe('server_folder');
    expect(component.activePickerView).toBe('server_folder');
  });

  it('does not auto-select when a category holds more than one importer', () => {
    component.mediaImporters.set([
      { name: 'server_folder', picker_view: 'server_folder', category: 'server' },
      { name: 'server_other', picker_view: 'server_folder', category: 'server' },
    ]);

    component.selectImporterTab('server');

    expect(component.selectedImporter).toBeNull();
  });

  // --- Trained tab: label-importer plugin field parity (issue #2597) ---

  it('fetches dynamic options when a Trained-tab importer is selected', () => {
    const field = { key: 'sheet', field_type: 'select', dynamic_options: true, required: true } as any;
    component.selectLabelImporter({ name: 'gsheets', fields: [field] } as any);

    const req = httpMock.expectOne('/api/label-importers/field-options/gsheets');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.field_key).toBe('sheet');
    req.flush({ options: [{ value: 's1', label: 'Sheet 1' }] });

    expect(component.labelImporterOptionsFor(field)).toEqual([{ value: 's1', label: 'Sheet 1' }]);
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
    expect(component.labelImporterOptionsFor(staticSelect)).toEqual([
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
    (component as any).refreshLabelImporterFieldOptions(field);
    httpMock
      .expectOne('/api/label-importers/field-options/imp')
      .flush({ options: [{ value: 'a', label: 'A' }] });
    expect(component.labelImporterValues['q']).toBe('hand-typed');
  });

  it('clears a strict-select value the refreshed options omit on the Trained tab', () => {
    const field = { key: 'q', field_type: 'select', dynamic_options: true } as any;
    component.selectedLabelImporter = { name: 'imp', fields: [field] } as any;
    component.labelImporterValues['q'] = 'stale';
    (component as any).refreshLabelImporterFieldOptions(field);
    httpMock
      .expectOne('/api/label-importers/field-options/imp')
      .flush({ options: [{ value: 'a', label: 'A' }] });
    expect(component.labelImporterValues['q']).toBe('');
  });

  it('surfaces a dynamic-option fetch error on the Trained tab', () => {
    const field = { key: 'q', field_type: 'select', dynamic_options: true } as any;
    component.selectedLabelImporter = { name: 'imp', fields: [field] } as any;
    (component as any).refreshLabelImporterFieldOptions(field);
    httpMock
      .expectOne('/api/label-importers/field-options/imp')
      .flush({ message: 'boom' }, { status: 500, statusText: 'Server Error' });
    expect(component.labelImporterDynamicError()['q']).toBe('boom');
    expect(component.labelImporterOptionsFor(field)).toEqual([]);
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
