import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { DatasetImporterModalComponent } from './dataset-importer-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';
import { getTabLabel } from './pickers/shared/media-type.util';

describe('DatasetImporterModalComponent', () => {
  let component: DatasetImporterModalComponent;
  let fixture: ComponentFixture<DatasetImporterModalComponent>;
  let httpMock: HttpTestingController;

  const mockImporters = [
    {
      name: 'local_folder',
      display_name: 'Folder',
      description: 'Upload from this computer',
      icon: '📁',
      picker_view: 'local_folder',
      category: 'local',
      fields: [],
    },
    {
      name: 'local_files',
      display_name: 'Files',
      description: 'Upload one or more individual files from this computer',
      icon: '📄',
      picker_view: 'local_files',
      category: 'local',
      fields: [],
    },
    {
      name: 'server_folder',
      display_name: 'Folder',
      description: 'Browse the server filesystem',
      icon: '🖥',
      picker_view: 'server_folder',
      category: 'server',
      fields: [
        { key: 'media_type', field_type: 'select', label: 'Media Type', default: 'audio', options: ['audio', 'image'] },
        { key: 'path', field_type: 'text', label: 'Folder Path', required: true },
      ],
    },
    {
      name: 'server_files',
      display_name: 'Files',
      description: 'Read a text file of paths from the server',
      icon: '🗂',
      picker_view: 'form',
      category: 'server',
      fields: [
        { key: 'media_type', field_type: 'select', label: 'Media Type', default: 'audio', options: ['audio', 'image'] },
        { key: 'paths_file', field_type: 'server_path', label: 'Paths File', required: true },
      ],
    },
    {
      name: 'demo',
      display_name: 'Downloaded Media',
      description: 'Pre-configured demo datasets',
      icon: '🗄',
      picker_view: 'demo',
      ui_mode: 'custom',
      category: 'demo',
      fields: [],
    },
    {
      name: 'pickle',
      display_name: 'Upload Saved Dataset',
      description: 'Load a .pkl dataset file',
      picker_view: 'form',
      fields: [{ key: 'file', field_type: 'file', label: 'Dataset File', required: true }],
    },
    {
      name: 'generic_form',
      display_name: 'Generic Form Importer',
      description: 'A test importer that renders the generic form',
      picker_view: 'form',
      fields: [
        { key: 'media_type', field_type: 'select', label: 'Media Type', default: 'audio', options: ['audio', 'image'] },
        { key: 'path', field_type: 'text', label: 'Path', required: true },
      ],
    },
  ];

  /** Convenience accessor for the generic-form mock importer (used by
   *  tests that exercise the default form code path). */
  const genericForm = () => mockImporters.find((i) => i.name === 'generic_form')!;
  const pickleImp = () => mockImporters.find((i) => i.name === 'pickle')!;

  const mockDemos = [
    {
      name: 'gtzan',
      label: 'GTZAN',
      status: 'needs_download',
      ready: false,
      num_files: 1000,
      download_size_mb: 1200,
      description: 'Music genre classification',
      media_type: 'audio',
      num_categories: 10,
      pkl_embedder: '',
      pkl_clipper: '',
      available_converters: [],
    },
    {
      name: 'flowers102',
      label: 'Oxford Flowers 102',
      status: 'ready',
      ready: true,
      num_files: 8189,
      download_size_mb: 330,
      description: '102 flower categories',
      media_type: 'image',
      num_categories: 102,
      pkl_embedder: 'clip',
      pkl_clipper: '',
      available_converters: [],
    },
  ];

  const mockEmbedders = [
    { name: 'clap', media_type_id: 'audio' },
  ];

  const mockImageEmbedders = [
    { name: 'clip', media_type_id: 'image' },
    { name: 'siglip', media_type_id: 'image' },
  ];

  const mockMediaTypes = [
    { type_id: 'audio', name: 'Audio', icon: 'audio' },
    { type_id: 'image', name: 'Image', icon: 'image' },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DatasetImporterModalComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DatasetImporterModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  const mockTabs = [
    { id: 'services', label: 'Services', icon: 'lightning', order: 10 },
    { id: 'server', label: 'Files', icon: 'folder', order: 20 },
    { id: 'local', label: 'Local', icon: 'house', order: 30 },
    { id: 'demo', label: 'Demo', icon: 'flask', order: 40 },
  ];

  /** ngOnInit fires four requests besides the importer list: a bare
   *  ``/api/embedders`` (no media_type), ``/api/media-types``, and the
   *  settings load (``/api/settings``).  Flush all of them so each test
   *  starts from a clean request queue. */
  function flushInitRequests(): void {
    httpMock.expectOne(req => req.url === '/api/embedders' && !req.params.get('media_type'))
      .flush({ embedders: [] });
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    TestBed.tick(); // flush the SettingsStateService rxResource loader (root effect)
    httpMock.expectOne('/api/settings').flush({});
  }

  function flushImporters(): void {
    TestBed.tick(); // run ngOnInit under zoneless (issues the init GETs); also resolves the static picker ViewChilds
    httpMock.expectOne('/api/dataset/all-importers').flush({ importers: mockImporters, tabs: mockTabs });
    flushInitRequests();
  }

  /** Flush the embedder + clipper fetches and the embedder-aware demo-list
   *  refetch that fire when a demo media-type tab is selected (either via
   *  the auto-select inside ``buildDemoTabs`` or an explicit user pick). */
  function flushDemoTabRequests(tab: string, embedders: typeof mockEmbedders): void {
    httpMock.expectOne(req =>
      req.url === '/api/embedders' && req.params.get('media_type') === tab,
    ).flush({ embedders });
    httpMock.expectOne(req =>
      req.url === '/api/clippers' && req.params.get('media_type') === tab,
    ).flush({ clippers: [] });
    if (embedders.length > 0) {
      httpMock.expectOne(req =>
        req.url === '/api/dataset/demo-list' && req.params.get('embedder') === embedders[0].name,
      ).flush({ datasets: mockDemos });
    }
  }

  /** Open the demo picker and flush the always-issued requests.  Opening
   *  the picker now auto-selects a media-type tab (``buildDemoTabs``
   *  prefers the solo type, then the guessed type, then ``audio``), so the
   *  per-tab embedder/clipper/refetch requests for that tab fire here too.
   *  With the default mocks the auto-selected tab is ``audio``. */
  function openAndFlushDemoPicker(): void {
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && !req.params.get('embedder'))
      .flush({ datasets: mockDemos });
    // buildDemoTabs auto-selects the 'audio' tab.
    flushDemoTabRequests('audio', mockEmbedders);
  }

  /** Helper: select a demo media-type tab and flush the embedder + clipper
   *  fetches plus the embedder-aware refetch. */
  function selectDemoTabAndFlush(tab: string, embedders: typeof mockEmbedders): void {
    component.demoPicker.selectDemoTabWithEmbedder(tab);
    flushDemoTabRequests(tab, embedders);
  }

  it('should create', () => {
    flushImporters();
    expect(component).toBeTruthy();
  });

  it('should fetch importers on init', () => {
    flushImporters();
    expect(component.importers().length).toBe(7);
  });

  it('should start with no top-level tab selected and a blank content area', async () => {
    flushImporters();
    await settleZoneless(fixture);
    expect(component.activeImporterTab()).toBe('');
    expect(component.selectedImporter()).toBeNull();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('.importer-subtab').length).toBe(0);
  });

  it('should pre-select the initialTab once importers and tabs arrive', () => {
    component.initialTab = 'server';
    flushImporters();
    expect(component.activeImporterTab()).toBe('server');
  });

  it('should ignore an initialTab id that no declared/used tab matches', () => {
    component.initialTab = 'no-such-tab';
    flushImporters();
    expect(component.activeImporterTab()).toBe('');
  });

  it('should always render the Services tab even when no importers populate it', async () => {
    flushImporters();
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;
    const tabLabels = Array.from(el.querySelectorAll('.tab-bar .tab')).map(
      (b) => (b.textContent || '').trim(),
    );
    expect(tabLabels.some((l) => l.includes('Services'))).toBe(true);
    // No importer is wired to category="services" in the mocks, so the
    // tab renders but importersForActiveTab is empty when selected.
    component.selectImporterTab('services');
    expect(component.importersForActiveTab.length).toBe(0);
  });

  it('should render inner importer sub-tabs for the active category', async () => {
    flushImporters();
    component.selectImporterTab('local');
    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;
    const subtabs = el.querySelectorAll('.importer-subtab');
    expect(subtabs.length).toBe(2);
    expect(subtabs[0].textContent).toContain('Folder');
    expect(subtabs[1].textContent).toContain('Files');

    // Switching to the Server tab swaps in the server importers and
    // clears the prior importer selection so the user must click again.
    component.selectImporterTab('server');
    expect(component.selectedImporter()).toBeNull();
    await settleZoneless(fixture);
    const serverSubtabs = el.querySelectorAll('.importer-subtab');
    expect(serverSubtabs.length).toBe(2);
    expect(serverSubtabs[0].textContent).toContain('Folder');
    expect(serverSubtabs[1].textContent).toContain('Files');
  });

  it('should hide importers marked hidden_from_picker', () => {
    TestBed.tick();
    const importersWithHidden = [
      ...mockImporters,
      { name: 'recaller', display_name: 'ReCaller', hidden_from_picker: true, fields: [] },
    ];
    httpMock.expectOne('/api/dataset/all-importers').flush({ importers: importersWithHidden, tabs: mockTabs });
    flushInitRequests();
    expect(component.importers().find((i) => i.name === 'recaller')).toBeUndefined();
  });

  it('should set activePickerView=local_folder when the Local Folder sub-tab is clicked', () => {
    flushImporters();
    const localFolder = component.importers().find((i) => i.name === 'local_folder')!;
    component.selectImporter(localFolder);
    expect(component.activePickerView).toBe('local_folder');
    expect(component.selectedImporter()?.name).toBe('local_folder');
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  });

  it('should set activePickerView=server_folder when the Server Folder sub-tab is clicked', () => {
    flushImporters();
    const folder = component.importers().find((i) => i.name === 'server_folder')!;
    component.selectImporter(folder);
    expect(component.activePickerView).toBe('server_folder');
    expect(component.selectedImporter()?.name).toBe('server_folder');
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    // The server-filesystem folder browser is now opened lazily by the
    // user (no /api/browse-media-files fires on importer selection).
  });

  it('should set activePickerView=local_files (paths-file picker) when the Local Files sub-tab is clicked', () => {
    flushImporters();
    const localFiles = component.importers().find((i) => i.name === 'local_files')!;
    component.selectImporter(localFiles);
    expect(component.activePickerView).toBe('local_files');
    expect(component.localFolderPicker.pickerKind()).toBe('files');
    expect(component.selectedImporter()?.name).toBe('local_files');
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  });

  it('should set activePickerView=demo when the Demo sub-tab is clicked', () => {
    flushImporters();
    const demo = component.importers().find((i) => i.name === 'demo')!;
    component.selectImporter(demo);
    expect(component.activePickerView).toBe('demo');
    expect(component.selectedImporter()?.name).toBe('demo');
    // Opening the demo picker fetches media-types + demo-list, then
    // auto-selects the 'audio' tab (which fires its embedder/clipper/refetch).
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && !req.params.get('embedder'))
      .flush({ datasets: mockDemos });
    flushDemoTabRequests('audio', mockEmbedders);
  });

  it('should POST uploaded folder via importLocalFolder', () => {
    flushImporters();
    vi.spyOn(component.importStarted, 'emit');

    const localFolder = component.importers().find((i) => i.name === 'local_folder')!;
    component.selectImporter(localFolder);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });

    const file = new File(['contents'], 'a.wav');
    Object.defineProperty(file, 'webkitRelativePath', { value: 'mydir/a.wav' });
    component.localFolderPicker.files.set([file]);
    component.localFolderPicker.mediaType = 'audio';
    component.localFolderPicker.submit();

    const req = httpMock.expectOne('/api/dataset/import-local-folder');
    expect(req.request.method).toBe('POST');
    const body = req.request.body as FormData;
    expect(body.get('media_type')).toBe('audio');
    expect(body.getAll('files').length).toBe(1);
    req.flush({ ok: true });

    expect(component.localFolderPicker.submitting()).toBe(false);
    expect(component.importStarted.emit).toHaveBeenCalled();
  });

  it('should POST uploaded paths file via importLocalFiles', () => {
    flushImporters();
    vi.spyOn(component.importStarted, 'emit');

    const localFiles = component.importers().find((i) => i.name === 'local_files')!;
    component.selectImporter(localFiles);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });

    const pathsFile = new File(['/a.wav\n/b.wav\n'], 'list.txt');
    component.localFolderPicker.files.set([pathsFile]);
    component.localFolderPicker.mediaType = 'audio';
    component.localFolderPicker.submit();

    const req = httpMock.expectOne('/api/dataset/import-local-files');
    expect(req.request.method).toBe('POST');
    const body = req.request.body as FormData;
    expect(body.get('media_type')).toBe('audio');
    const uploaded = body.get('paths_file') as File;
    expect(uploaded.name).toBe('list.txt');
    expect(body.get('files')).toBeNull();
    req.flush({ ok: true });

    expect(component.localFolderPicker.submitting()).toBe(false);
    expect(component.importStarted.emit).toHaveBeenCalled();
  });

  it('should set activePickerView=form on generic importer selection', () => {
    flushImporters();
    const imp = genericForm();
    component.selectImporter(imp);
    expect(component.activePickerView).toBe('form');
    expect(component.selectedImporter()).toBe(imp);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  });

  it('should pre-populate default values', () => {
    flushImporters();
    component.selectImporter(genericForm());
    expect(component.genericFormPicker.formValues['media_type']).toBe('audio');
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  });

  it('should clear the selected importer when the active top-level tab changes', () => {
    flushImporters();
    component.selectImporterTab('local');
    component.selectImporter(genericForm());
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    expect(component.selectedImporter()).not.toBeNull();
    // Switch to a multi-importer tab so the tab change clears the prior
    // selection without auto-selecting a lone importer (the 'demo' tab has
    // a single importer, which selectImporterTab would auto-open).
    component.selectImporterTab('server');
    expect(component.selectedImporter()).toBeNull();
    expect(component.activePickerView).toBe('');
  });

  it('should submit form values via runImporter', () => {
    flushImporters();
    vi.spyOn(component.importStarted, 'emit');

    component.selectImporter(genericForm());
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    component.genericFormPicker.formValues['path'] = '/data/sounds';
    component.genericFormPicker.submit();

    const req = httpMock.expectOne('/api/dataset/import/generic_form');
    expect(req.request.method).toBe('POST');
    expect(req.request.body['path']).toBe('/data/sounds');
    req.flush({});

    expect(component.genericFormPicker.submitting()).toBe(false);
    expect(component.importStarted.emit).toHaveBeenCalled();
  });

  it('should show error on import failure', () => {
    flushImporters();
    component.selectImporter(genericForm());
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    component.genericFormPicker.submit();

    httpMock.expectOne('/api/dataset/import/generic_form').flush(
      { error: 'Not found' },
      { status: 404, statusText: 'Not Found' },
    );

    expect(component.genericFormPicker.submitting()).toBe(false);
    expect(component.genericFormPicker.error()).toBe('Not found');
  });

  it('should emit closed on close', () => {
    flushImporters();
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should use loadFile for file type fields', () => {
    flushImporters();
    vi.spyOn(component.importStarted, 'emit');

    component.selectImporter(pickleImp());
    const mockFile = new File(['data'], 'test.pkl');
    component.genericFormPicker.selectedFile = mockFile;
    component.genericFormPicker.submit();

    const req = httpMock.expectOne('/api/dataset/load-file');
    expect(req.request.method).toBe('POST');
    req.flush({});

    expect(component.importStarted.emit).toHaveBeenCalled();
  });

  // --- Demo picker tests ---

  it('should activate the demo view when openDemoPicker is called', () => {
    flushImporters();

    expect(component.activePickerView).toBe('');
    component.openDemoPicker();
    expect(component.activePickerView).toBe('demo');
    expect(component.demoPicker.demoLoading()).toBe(true);

    // Flush media types and demo list requests
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && !req.params.get('embedder'))
      .flush({ datasets: mockDemos });

    // buildDemoTabs auto-selects the 'audio' tab, firing its
    // embedder/clipper/refetch.
    flushDemoTabRequests('audio', mockEmbedders);

    expect(component.demoPicker.demoLoading()).toBe(false);
    expect(component.demoPicker.demos().length).toBe(2);
    expect(component.demoPicker.activeTab()).toBe('audio');
  });

  it('should build tabs from media types in demo data and auto-select the first', () => {
    flushImporters();
    openAndFlushDemoPicker();

    expect(component.demoPicker.demoTabs()).toEqual(['audio', 'image']);
    // buildDemoTabs auto-selects the 'audio' tab so the demo table shows
    // results immediately instead of sitting blank.
    expect(component.demoPicker.activeTab()).toBe('audio');
    expect(component.demoPicker.filteredDemos.length).toBe(1);
  });

  it('should default the demo media-type tab to the active context type when known', () => {
    flushImporters();
    // The dashboard passes the active context's single media type (e.g. an
    // Image dataset/detector already loaded) as guessedMediaType. It is now
    // consumed via a child @Input; a direct field write on the parent
    // doesn't flow through an Angular input binding under OnPush unless a
    // real change-detection pass runs, so set it on the picker directly
    // (mirroring what the real `[guessedMediaType]` binding would deliver).
    component.guessedMediaType = 'image';
    component.demoPicker.guessedMediaType = 'image';

    component.openDemoPicker();
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && !req.params.get('embedder'))
      .flush({ datasets: mockDemos });

    // buildDemoTabs pre-selects the guessed type instead of falling back to
    // audio, so the image embedder/clipper fetches fire for that tab.
    flushDemoTabRequests('image', mockImageEmbedders);

    expect(component.demoPicker.activeTab()).toBe('image');
  });

  it('should filter demos by the explicitly selected media-type tab', () => {
    flushImporters();
    openAndFlushDemoPicker();

    selectDemoTabAndFlush('audio', mockEmbedders);
    expect(component.demoPicker.filteredDemos.length).toBe(1);
    expect(component.demoPicker.filteredDemos[0].name).toBe('gtzan');

    selectDemoTabAndFlush('image', mockImageEmbedders);
    expect(component.demoPicker.filteredDemos.length).toBe(1);
    expect(component.demoPicker.filteredDemos[0].name).toBe('flowers102');
  });

  it('should sort demos by column', () => {
    flushImporters();
    openAndFlushDemoPicker();

    // Default sort is num_files ascending
    expect(component.demoPicker.demoCols.sortColumn).toBe('num_files');
    expect(component.demoPicker.demoCols.sortAsc).toBe(true);

    // Click same column to toggle direction
    component.demoPicker.demoCols.sortBy('num_files');
    expect(component.demoPicker.demoCols.sortAsc).toBe(false);

    // Click different column to switch
    component.demoPicker.demoCols.sortBy('label');
    expect(component.demoPicker.demoCols.sortColumn).toBe('label');
    expect(component.demoPicker.demoCols.sortAsc).toBe(true);
  });

  it('should record the demo selection without submitting on row click', () => {
    flushImporters();
    vi.spyOn(component.demoSelected, 'emit');
    vi.spyOn(component.closed, 'emit');

    openAndFlushDemoPicker();

    component.demoPicker.selectDemo(mockDemos[0] as any);
    expect(component.demoPicker.selectedDemo()).toBe(mockDemos[0] as any);
    // Auto-populates the Dataset Name from the chosen demo's label.
    expect(component.demoPicker.demoDatasetName).toBe(mockDemos[0].label);
    // No emit/close until the Import footer button is clicked.
    expect(component.demoSelected.emit).not.toHaveBeenCalled();
    expect(component.closed.emit).not.toHaveBeenCalled();
  });

  it('should emit demoSelected and close when the demo picker submits', () => {
    flushImporters();
    vi.spyOn(component.demoSelected, 'emit');
    vi.spyOn(component.closed, 'emit');

    openAndFlushDemoPicker();

    component.demoPicker.selectDemo(mockDemos[0] as any);
    component.demoPicker.submit();
    expect(component.demoSelected.emit).toHaveBeenCalled();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should hide the demo tab bar in favor of a dropdown above the grid', async () => {
    flushImporters();
    openAndFlushDemoPicker();

    await settleZoneless(fixture);
    const el = fixture.nativeElement as HTMLElement;

    // The inner media-type tabs are replaced by a <vt-import-config>
    // dropdown, so no .demo-tab buttons render in this modal anymore.
    expect(el.querySelectorAll('.demo-tab').length).toBe(0);

    // The 'audio' tab is auto-selected on open, so its single demo row
    // (gtzan) is already rendered.
    expect(el.querySelectorAll('.demo-row').length).toBe(1);

    // Switching to the 'image' tab swaps in that tab's single row.
    selectDemoTabAndFlush('image', mockImageEmbedders);
    await settleZoneless(fixture);
    expect(el.querySelectorAll('.demo-row').length).toBe(1);
  });

  it('should clear the row selection when the media type changes', () => {
    flushImporters();
    openAndFlushDemoPicker();
    selectDemoTabAndFlush('audio', mockEmbedders);

    component.demoPicker.selectDemo(mockDemos[0] as any);
    expect(component.demoPicker.selectedDemo()).not.toBeNull();
    expect(component.demoPicker.demoDatasetName).toBe(mockDemos[0].label);

    selectDemoTabAndFlush('image', mockImageEmbedders);
    expect(component.demoPicker.selectedDemo()).toBeNull();
    expect(component.demoPicker.demoDatasetName).toBe('');
  });

  it('should get correct tab label from media types', () => {
    flushImporters();
    component.demoPicker.mediaTypes.set(mockMediaTypes as any);

    expect(getTabLabel(component.demoPicker.mediaTypes(), 'audio')).toContain('Audio');
    expect(getTabLabel(component.demoPicker.mediaTypes(), 'unknown')).toBe('unknown');
  });

  it('should handle demo fetch failure gracefully', () => {
    flushImporters();
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush(
      { error: 'Server error' },
      { status: 500, statusText: 'Internal Server Error' },
    );

    expect(component.demoPicker.demoLoading()).toBe(false);
    expect(component.demoPicker.demos().length).toBe(0);
  });

  // --- Embedder-aware status tests ---

  it('should re-fetch demos from server when embedder changes', () => {
    flushImporters();
    openAndFlushDemoPicker();
    selectDemoTabAndFlush('image', mockImageEmbedders);

    // Changing embedder triggers a re-fetch with the new embedder param
    component.demoPicker.onDemoEmbedderChange('siglip');

    const req = httpMock.expectOne(
      r => r.url === '/api/dataset/demo-list' && r.params.get('embedder') === 'siglip',
    );
    expect(req.request.method).toBe('GET');
    req.flush({ datasets: mockDemos });
  });

  it('should downgrade ready status when pkl_embedder mismatches selected embedder', () => {
    flushImporters();

    // Set up demos with known pkl_embedder for the active tab
    const demosWithEmbedder = [
      {
        name: 'flowers102',
        label: 'Oxford Flowers 102',
        status: 'ready' as const,
        ready: true,
        num_files: 8189,
        download_size_mb: 330,
        description: '102 flower categories',
        media_type: 'image',
        num_categories: 102,
        pkl_embedder: 'clip',
        pkl_clipper: '',
        available_converters: [],
      },
    ];

    component.demoPicker.demos.set(demosWithEmbedder);
    component.demoPicker.activeTab.set('image');
    component.demoPicker.selectedDemoEmbedder.set('siglip');

    // Call updateDemoStatuses via the public embedder change handler
    // (we set up state directly to test the client-side logic)
    (component.demoPicker as any).updateDemoStatuses();

    expect(demosWithEmbedder[0].status).toBe('needs_embedding');
    expect(demosWithEmbedder[0].ready).toBe(false);
  });

  it('should restore ready status when pkl_embedder matches selected embedder', () => {
    flushImporters();

    const demosWithEmbedder = [
      {
        name: 'flowers102',
        label: 'Oxford Flowers 102',
        status: 'needs_embedding' as const,
        ready: false,
        num_files: 8189,
        download_size_mb: 330,
        description: '102 flower categories',
        media_type: 'image',
        num_categories: 102,
        pkl_embedder: 'clip',
        pkl_clipper: '',
        available_converters: [],
      },
    ];

    component.demoPicker.demos.set(demosWithEmbedder);
    component.demoPicker.activeTab.set('image');
    component.demoPicker.selectedDemoEmbedder.set('clip');

    (component.demoPicker as any).updateDemoStatuses();

    expect(demosWithEmbedder[0].status).toBe('ready');
    expect(demosWithEmbedder[0].ready).toBe(true);
  });

  it('should conservatively downgrade ready demos with unknown pkl_embedder', () => {
    flushImporters();

    const demosUnknown = [
      {
        name: 'mystery',
        label: 'Mystery Dataset',
        status: 'ready' as const,
        ready: true,
        num_files: 100,
        download_size_mb: 10,
        description: 'Unknown embedder',
        media_type: 'image',
        num_categories: 5,
        pkl_embedder: '',  // unknown
        pkl_clipper: '',
        available_converters: [],
      },
    ];

    component.demoPicker.demos.set(demosUnknown);
    component.demoPicker.activeTab.set('image');
    component.demoPicker.selectedDemoEmbedder.set('clip');

    (component.demoPicker as any).updateDemoStatuses();

    // Since we don't know the pkl_embedder, we can't confirm it matches
    expect(demosUnknown[0].status).toBe('needs_embedding');
    expect(demosUnknown[0].ready).toBe(false);
  });

  it('should not modify demos from other tabs during updateDemoStatuses', () => {
    flushImporters();

    const crossTabDemos = [
      {
        name: 'gtzan',
        label: 'GTZAN',
        status: 'ready' as const,
        ready: true,
        num_files: 1000,
        download_size_mb: 1200,
        description: 'Music',
        media_type: 'audio',
        num_categories: 10,
        pkl_embedder: 'clap',
        pkl_clipper: '',
        available_converters: [],
      },
      {
        name: 'flowers102',
        label: 'Oxford Flowers 102',
        status: 'ready' as const,
        ready: true,
        num_files: 8189,
        download_size_mb: 330,
        description: '102 flower categories',
        media_type: 'image',
        num_categories: 102,
        pkl_embedder: 'clip',
        pkl_clipper: '',
        available_converters: [],
      },
    ];

    component.demoPicker.demos.set(crossTabDemos);
    component.demoPicker.activeTab.set('image');
    component.demoPicker.selectedDemoEmbedder.set('siglip');  // Doesn't match 'clip'

    (component.demoPicker as any).updateDemoStatuses();

    // Image demo should be downgraded (active tab, embedder mismatch)
    expect(crossTabDemos[1].status).toBe('needs_embedding');
    // Audio demo should NOT be touched (different tab)
    expect(crossTabDemos[0].status).toBe('ready');
  });

  it('should not touch needs_download demos during updateDemoStatuses', () => {
    flushImporters();

    const downloadDemos = [
      {
        name: 'gtzan',
        label: 'GTZAN',
        status: 'needs_download' as const,
        ready: false,
        num_files: 1000,
        download_size_mb: 1200,
        description: 'Music',
        media_type: 'audio',
        num_categories: 10,
        pkl_embedder: '',
        pkl_clipper: '',
        available_converters: [],
      },
    ];

    component.demoPicker.demos.set(downloadDemos);
    component.demoPicker.activeTab.set('audio');
    component.demoPicker.selectedDemoEmbedder.set('clap');

    (component.demoPicker as any).updateDemoStatuses();

    expect(downloadDemos[0].status).toBe('needs_download');
  });

  it('should re-fetch demos with the tab embedder once a media-type tab is selected', () => {
    flushImporters();
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && !req.params.get('embedder'))
      .flush({ datasets: mockDemos });

    // buildDemoTabs auto-selects the 'audio' tab, which loads its
    // embedders + clippers and re-fetches the demo list with the now-known
    // default embedder ('clap').
    httpMock.expectOne(req =>
      req.url === '/api/embedders' && req.params.get('media_type') === 'audio',
    ).flush({ embedders: mockEmbedders });
    httpMock.expectOne(req =>
      req.url === '/api/clippers' && req.params.get('media_type') === 'audio',
    ).flush({ clippers: [] });

    const refetchReq = httpMock.expectOne(req =>
      req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'clap',
    );
    expect(refetchReq.request.method).toBe('GET');
    refetchReq.flush({ datasets: mockDemos });
  });
});
