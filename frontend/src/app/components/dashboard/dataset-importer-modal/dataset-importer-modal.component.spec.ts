import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { DatasetImporterModalComponent } from './dataset-importer-modal.component';

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
      display_name: 'Downloaded Demo Media',
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
      providers: [provideHttpClient(), provideHttpClientTesting()],
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
    { id: 'server', label: 'Server', icon: 'server', order: 20 },
    { id: 'local', label: 'Local', icon: 'house', order: 30 },
    { id: 'demo', label: 'Demo', icon: 'flask', order: 40 },
  ];

  function flushImporters(): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/dataset/all-importers').flush({ importers: mockImporters, tabs: mockTabs });
  }

  /** Open the demo picker and flush the always-issued requests.  Unlike
   *  the previous behavior, opening the picker no longer auto-selects a
   *  media-type tab, so no per-tab embedder/clipper requests fire here.
   *  Tests that need a tab selected should call ``selectDemoTab`` (or
   *  ``selectDemoTabWithEmbedder``) explicitly and flush the resulting
   *  requests themselves. */
  function openAndFlushDemoPicker(): void {
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });
  }

  /** Helper: select a demo media-type tab and flush the embedder + clipper
   *  fetches plus the embedder-aware refetch.  Returns the flushed
   *  embedders for convenience. */
  function selectDemoTabAndFlush(tab: string, embedders: typeof mockEmbedders): void {
    component.selectDemoTabWithEmbedder(tab);
    httpMock.expectOne(req =>
      req.url === '/api/embedders' && req.params.get('media_type') === tab,
    ).flush({ embedders });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    if (embedders.length > 0) {
      httpMock.expectOne(req =>
        req.url === '/api/dataset/demo-list' && req.params.get('embedder') === embedders[0].name,
      ).flush({ datasets: mockDemos });
    }
  }

  it('should create', () => {
    flushImporters();
    expect(component).toBeTruthy();
  });

  it('should fetch importers on init', () => {
    flushImporters();
    expect(component.importers.length).toBe(7);
  });

  it('should start with no top-level tab selected and a blank content area', () => {
    flushImporters();
    fixture.detectChanges();
    expect(component.activeImporterTab).toBe('');
    expect(component.selectedImporter).toBeNull();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('.importer-subtab').length).toBe(0);
  });

  it('should pre-select the initialTab once importers and tabs arrive', () => {
    component.initialTab = 'server';
    flushImporters();
    expect(component.activeImporterTab).toBe('server');
  });

  it('should ignore an initialTab id that no declared/used tab matches', () => {
    component.initialTab = 'no-such-tab';
    flushImporters();
    expect(component.activeImporterTab).toBe('');
  });

  it('should always render the Services tab even when no importers populate it', () => {
    flushImporters();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const tabLabels = Array.from(el.querySelectorAll('.importer-tab')).map(
      (b) => (b.textContent || '').trim(),
    );
    expect(tabLabels.some((l) => l.includes('Services'))).toBeTrue();
    // No importer is wired to category="services" in the mocks, so the
    // tab renders but importersForActiveTab is empty when selected.
    component.selectImporterTab('services');
    expect(component.importersForActiveTab.length).toBe(0);
  });

  it('should render inner importer sub-tabs for the active category', () => {
    flushImporters();
    component.selectImporterTab('local');
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const subtabs = el.querySelectorAll('.importer-subtab');
    expect(subtabs.length).toBe(2);
    expect(subtabs[0].textContent).toContain('Folder');
    expect(subtabs[1].textContent).toContain('Files');

    // Switching to the Server tab swaps in the server importers and
    // clears the prior importer selection so the user must click again.
    component.selectImporterTab('server');
    expect(component.selectedImporter).toBeNull();
    fixture.detectChanges();
    const serverSubtabs = el.querySelectorAll('.importer-subtab');
    expect(serverSubtabs.length).toBe(2);
    expect(serverSubtabs[0].textContent).toContain('Folder');
    expect(serverSubtabs[1].textContent).toContain('Files');
  });

  it('should hide importers marked hidden_from_picker', () => {
    fixture.detectChanges();
    const importersWithHidden = [
      ...mockImporters,
      { name: 'recaller', display_name: 'ReCaller', hidden_from_picker: true, fields: [] },
    ];
    httpMock.expectOne('/api/dataset/all-importers').flush({ importers: importersWithHidden, tabs: mockTabs });
    expect(component.importers.find((i) => i.name === 'recaller')).toBeUndefined();
  });

  it('should set activePickerView=local_folder when the Local Folder sub-tab is clicked', () => {
    flushImporters();
    const localFolder = component.importers.find((i) => i.name === 'local_folder')!;
    component.selectImporter(localFolder);
    expect(component.activePickerView).toBe('local_folder');
    expect(component.selectedImporter?.name).toBe('local_folder');
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  });

  it('should set activePickerView=server_folder when the Server Folder sub-tab is clicked', () => {
    flushImporters();
    const folder = component.importers.find((i) => i.name === 'server_folder')!;
    component.selectImporter(folder);
    expect(component.activePickerView).toBe('server_folder');
    expect(component.selectedImporter?.name).toBe('server_folder');
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    httpMock.expectOne(req => req.url === '/api/browse-media-files').flush({ directories: [], files: [], root_path: '' });
  });

  it('should set activePickerView=local_files (paths-file picker) when the Local Files sub-tab is clicked', () => {
    flushImporters();
    const localFiles = component.importers.find((i) => i.name === 'local_files')!;
    component.selectImporter(localFiles);
    expect(component.activePickerView).toBe('local_files');
    expect(component.lfPickerKind).toBe('files');
    expect(component.selectedImporter?.name).toBe('local_files');
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  });

  it('should set activePickerView=demo when the Demo sub-tab is clicked', () => {
    flushImporters();
    const demo = component.importers.find((i) => i.name === 'demo')!;
    component.selectImporter(demo);
    expect(component.activePickerView).toBe('demo');
    expect(component.selectedImporter?.name).toBe('demo');
    // Opening the demo picker no longer auto-selects a media-type tab,
    // so only the always-issued media-types + demo-list calls fire.
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });
  });

  it('should POST uploaded folder via importLocalFolder', () => {
    flushImporters();
    spyOn(component.importStarted, 'emit');

    const localFolder = component.importers.find((i) => i.name === 'local_folder')!;
    component.selectImporter(localFolder);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });

    const file = new File(['contents'], 'a.wav');
    Object.defineProperty(file, 'webkitRelativePath', { value: 'mydir/a.wav' });
    component.lfFiles = [file];
    component.lfMediaType = 'audio';
    component.lfSubmit();

    const req = httpMock.expectOne('/api/dataset/import-local-folder');
    expect(req.request.method).toBe('POST');
    const body = req.request.body as FormData;
    expect(body.get('media_type')).toBe('audio');
    expect(body.getAll('files').length).toBe(1);
    req.flush({ ok: true });

    expect(component.lfSubmitting).toBeFalse();
    expect(component.importStarted.emit).toHaveBeenCalled();
  });

  it('should POST uploaded paths file via importLocalFiles', () => {
    flushImporters();
    spyOn(component.importStarted, 'emit');

    const localFiles = component.importers.find((i) => i.name === 'local_files')!;
    component.selectImporter(localFiles);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });

    const pathsFile = new File(['/a.wav\n/b.wav\n'], 'list.txt');
    component.lfFiles = [pathsFile];
    component.lfMediaType = 'audio';
    component.lfSubmit();

    const req = httpMock.expectOne('/api/dataset/import-local-files');
    expect(req.request.method).toBe('POST');
    const body = req.request.body as FormData;
    expect(body.get('media_type')).toBe('audio');
    const uploaded = body.get('paths_file') as File;
    expect(uploaded.name).toBe('list.txt');
    expect(body.get('files')).toBeNull();
    req.flush({ ok: true });

    expect(component.lfSubmitting).toBeFalse();
    expect(component.importStarted.emit).toHaveBeenCalled();
  });

  it('should set activePickerView=form on generic importer selection', () => {
    flushImporters();
    const imp = genericForm();
    component.selectImporter(imp);
    expect(component.activePickerView).toBe('form');
    expect(component.selectedImporter).toBe(imp);
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  });

  it('should pre-populate default values', () => {
    flushImporters();
    component.selectImporter(genericForm());
    expect(component.formValues['media_type']).toBe('audio');
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  });

  it('should clear the selected importer when the active top-level tab changes', () => {
    flushImporters();
    component.selectImporterTab('local');
    component.selectImporter(genericForm());
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    expect(component.selectedImporter).not.toBeNull();
    component.selectImporterTab('demo');
    expect(component.selectedImporter).toBeNull();
    expect(component.activePickerView).toBe('');
  });

  it('should submit form values via runImporter', () => {
    flushImporters();
    spyOn(component.importStarted, 'emit');

    component.selectImporter(genericForm());
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    component.formValues['path'] = '/data/sounds';
    component.submit();

    const req = httpMock.expectOne('/api/dataset/import/generic_form');
    expect(req.request.method).toBe('POST');
    expect(req.request.body['path']).toBe('/data/sounds');
    req.flush({});

    expect(component.submitting).toBeFalse();
    expect(component.importStarted.emit).toHaveBeenCalled();
  });

  it('should show error on import failure', () => {
    flushImporters();
    component.selectImporter(genericForm());
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
    component.submit();

    httpMock.expectOne('/api/dataset/import/generic_form').flush(
      { error: 'Not found' },
      { status: 404, statusText: 'Not Found' },
    );

    expect(component.submitting).toBeFalse();
    expect(component.error).toBe('Not found');
  });

  it('should emit closed on close', () => {
    flushImporters();
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should use loadFile for file type fields', () => {
    flushImporters();
    spyOn(component.importStarted, 'emit');

    component.selectImporter(pickleImp());
    const mockFile = new File(['data'], 'test.pkl');
    component.selectedFile = mockFile;
    component.submit();

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
    expect(component.demoLoading).toBeTrue();

    // Flush media types and demo list requests
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

    // No media-type tab is auto-selected anymore, so no embedder/clipper
    // fetches fire automatically.
    expect(component.demoLoading).toBeFalse();
    expect(component.demos.length).toBe(2);
    expect(component.activeTab).toBe('');
  });

  it('should build tabs from media types in demo data without auto-selecting one', () => {
    flushImporters();
    openAndFlushDemoPicker();

    expect(component.demoTabs).toEqual(['audio', 'image']);
    // No media-type tab is auto-selected — the demo table area stays
    // blank until the user clicks one.
    expect(component.activeTab).toBe('');
    expect(component.filteredDemos.length).toBe(0);
  });

  it('should filter demos by the explicitly selected media-type tab', () => {
    flushImporters();
    openAndFlushDemoPicker();

    selectDemoTabAndFlush('audio', mockEmbedders);
    expect(component.filteredDemos.length).toBe(1);
    expect(component.filteredDemos[0].name).toBe('gtzan');

    selectDemoTabAndFlush('image', mockImageEmbedders);
    expect(component.filteredDemos.length).toBe(1);
    expect(component.filteredDemos[0].name).toBe('flowers102');
  });

  it('should sort demos by column', () => {
    flushImporters();
    openAndFlushDemoPicker();

    // Default sort is num_files ascending
    expect(component.demoCols.sortColumn).toBe('num_files');
    expect(component.demoCols.sortAsc).toBeTrue();

    // Click same column to toggle direction
    component.demoCols.sortBy('num_files');
    expect(component.demoCols.sortAsc).toBeFalse();

    // Click different column to switch
    component.demoCols.sortBy('label');
    expect(component.demoCols.sortColumn).toBe('label');
    expect(component.demoCols.sortAsc).toBeTrue();
  });

  it('should record the demo selection without submitting on row click', () => {
    flushImporters();
    spyOn(component.demoSelected, 'emit');
    spyOn(component.closed, 'emit');

    openAndFlushDemoPicker();

    component.selectDemo(mockDemos[0] as any);
    expect(component.selectedDemo).toBe(mockDemos[0] as any);
    // Auto-populates the Dataset Name from the chosen demo's label.
    expect(component.demoDatasetName).toBe(mockDemos[0].label);
    // No emit/close until the Import footer button is clicked.
    expect(component.demoSelected.emit).not.toHaveBeenCalled();
    expect(component.closed.emit).not.toHaveBeenCalled();
  });

  it('should emit demoSelected and close when submitDemo is called', () => {
    flushImporters();
    spyOn(component.demoSelected, 'emit');
    spyOn(component.closed, 'emit');

    openAndFlushDemoPicker();

    component.selectDemo(mockDemos[0] as any);
    component.submitDemo();
    expect(component.demoSelected.emit).toHaveBeenCalled();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should hide the demo tab bar in favor of a dropdown above the grid', () => {
    flushImporters();
    openAndFlushDemoPicker();

    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    // The inner media-type tabs are replaced by a <vt-import-config>
    // dropdown — so no .demo-tab buttons render in this modal anymore.
    expect(el.querySelectorAll('.demo-tab').length).toBe(0);

    // No media type picked yet → no rows.
    expect(el.querySelectorAll('.demo-row').length).toBe(0);

    selectDemoTabAndFlush('audio', mockEmbedders);
    fixture.detectChanges();
    expect(el.querySelectorAll('.demo-row').length).toBe(1);
  });

  it('should clear the row selection when the media type changes', () => {
    flushImporters();
    openAndFlushDemoPicker();
    selectDemoTabAndFlush('audio', mockEmbedders);

    component.selectDemo(mockDemos[0] as any);
    expect(component.selectedDemo).not.toBeNull();
    expect(component.demoDatasetName).toBe(mockDemos[0].label);

    selectDemoTabAndFlush('image', mockImageEmbedders);
    expect(component.selectedDemo).toBeNull();
    expect(component.demoDatasetName).toBe('');
  });

  it('should get correct tab label from media types', () => {
    flushImporters();
    component.mediaTypes = mockMediaTypes as any;

    expect(component.getTabLabel('audio')).toContain('Audio');
    expect(component.getTabLabel('unknown')).toBe('unknown');
  });

  it('should handle demo fetch failure gracefully', () => {
    flushImporters();
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush(
      { error: 'Server error' },
      { status: 500, statusText: 'Internal Server Error' },
    );

    expect(component.demoLoading).toBeFalse();
    expect(component.demos.length).toBe(0);
  });

  // --- Embedder-aware status tests ---

  it('should re-fetch demos from server when embedder changes', () => {
    flushImporters();
    openAndFlushDemoPicker();
    selectDemoTabAndFlush('image', mockImageEmbedders);

    // Changing embedder triggers a re-fetch with the new embedder param
    component.onDemoEmbedderChange('siglip');

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
      },
    ];

    component.demos = demosWithEmbedder;
    component.activeTab = 'image';
    component.selectedDemoEmbedder = 'siglip';

    // Call updateDemoStatuses via the public embedder change handler
    // (we set up state directly to test the client-side logic)
    (component as any).updateDemoStatuses();

    expect(demosWithEmbedder[0].status).toBe('needs_embedding');
    expect(demosWithEmbedder[0].ready).toBeFalse();
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
      },
    ];

    component.demos = demosWithEmbedder;
    component.activeTab = 'image';
    component.selectedDemoEmbedder = 'clip';

    (component as any).updateDemoStatuses();

    expect(demosWithEmbedder[0].status).toBe('ready');
    expect(demosWithEmbedder[0].ready).toBeTrue();
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
      },
    ];

    component.demos = demosUnknown;
    component.activeTab = 'image';
    component.selectedDemoEmbedder = 'clip';

    (component as any).updateDemoStatuses();

    // Since we don't know the pkl_embedder, we can't confirm it matches
    expect(demosUnknown[0].status).toBe('needs_embedding');
    expect(demosUnknown[0].ready).toBeFalse();
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
      },
    ];

    component.demos = crossTabDemos;
    component.activeTab = 'image';
    component.selectedDemoEmbedder = 'siglip';  // Doesn't match 'clip'

    (component as any).updateDemoStatuses();

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
      },
    ];

    component.demos = downloadDemos;
    component.activeTab = 'audio';
    component.selectedDemoEmbedder = 'clap';

    (component as any).updateDemoStatuses();

    expect(downloadDemos[0].status).toBe('needs_download');
  });

  it('should re-fetch demos with the tab embedder once the user picks a media-type tab', () => {
    flushImporters();
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

    // No requests fire yet — the user hasn't picked a media-type tab.
    httpMock.expectNone(req => req.url === '/api/embedders');

    // Picking a media-type tab loads its embedders + clippers and then
    // re-fetches the demo list with the now-known default embedder.
    component.selectDemoTabWithEmbedder('audio');

    httpMock.expectOne(req =>
      req.url === '/api/embedders' && req.params.get('media_type') === 'audio',
    ).flush({ embedders: mockEmbedders });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });

    const refetchReq = httpMock.expectOne(req =>
      req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'clap',
    );
    expect(refetchReq.request.method).toBe('GET');
    refetchReq.flush({ datasets: mockDemos });
  });
});
