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
      name: 'folder',
      label: 'Load from Folder',
      description: 'Import media files from a folder',
      fields: [
        { key: 'path', field_type: 'text', label: 'Folder Path', required: true },
        { key: 'media_type', field_type: 'select', label: 'Media Type', default: 'audio' },
      ],
    },
    {
      name: 'pickle',
      label: 'Load from File',
      description: 'Load a .pkl dataset file',
      fields: [{ key: 'file', field_type: 'file', label: 'Dataset File', required: true }],
    },
  ];

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
    { type_id: 'audio', name: 'Audio', icon: 'audio', tab_title: 'Audio' },
    { type_id: 'image', name: 'Images', icon: 'image', tab_title: 'Images' },
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

  function flushImporters(): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/dataset/all-importers').flush({ importers: mockImporters });
  }

  /** Open the demo picker and flush all resulting HTTP requests. */
  function openAndFlushDemoPicker(embedders = mockEmbedders): void {
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    // Initial demo list fetch (no embedder param)
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });
    // loadDemoEmbedders fires for the first tab
    httpMock.expectOne(req =>
      req.url === '/api/embedders' && req.params.get('media_type') === 'audio',
    ).flush({ embedders });
    // refetchDemoStatuses fires after embedders are loaded (if embedder is set)
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
    expect(component.importers.length).toBe(2);
  });

  it('should start in picker view', () => {
    flushImporters();
    expect(component.view).toBe('picker');
  });

  it('should render hardcoded local/server folder buttons plus importer cards plus demo card', () => {
    flushImporters();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const cards = el.querySelectorAll('.importer-card');
    // 2 hardcoded buttons (Local Folder, Server Folder) + 1 demo card
    // + 2 mock importers (folder, pickle)
    expect(cards.length).toBe(5);
    // Hardcoded Local Folder button always comes first so users see the
    // browser-side option before the server-side one.
    expect(cards[0].textContent).toContain('Import from Local Folder');
    expect(cards[1].textContent).toContain('Import from Server Folder');
    expect(cards[2].textContent).toContain('Load Demo Dataset');
  });

  it('should switch to local_folder view when Import from Local Folder is clicked', () => {
    flushImporters();
    component.openLocalFolderUploader();
    expect(component.view).toBe('local_folder');
    // The first available media type from the (mock) folder importer is used
    // as the default; for our mock that's whatever the field's default key is.
    httpMock.expectOne(req => req.url === '/api/embedders').flush({ embedders: [] });
    httpMock.expectOne(req => req.url === '/api/clippers').flush({ clippers: [] });
  });

  it('should POST uploaded folder via importLocalFolder', () => {
    flushImporters();
    spyOn(component.importStarted, 'emit');

    component.openLocalFolderUploader();
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

  it('should switch to form view on importer selection', () => {
    flushImporters();
    component.selectImporter(mockImporters[0]);
    expect(component.view).toBe('form');
    expect(component.selectedImporter).toBe(mockImporters[0]);
  });

  it('should pre-populate default values', () => {
    flushImporters();
    component.selectImporter(mockImporters[0]);
    expect(component.formValues['media_type']).toBe('audio');
  });

  it('should go back to picker view', () => {
    flushImporters();
    component.selectImporter(mockImporters[0]);
    component.back();
    expect(component.view).toBe('picker');
    expect(component.selectedImporter).toBeNull();
  });

  it('should submit form values via runImporter', () => {
    flushImporters();
    spyOn(component.importStarted, 'emit');

    component.selectImporter(mockImporters[0]);
    component.formValues['path'] = '/data/sounds';
    component.submit();

    const req = httpMock.expectOne('/api/dataset/import/folder');
    expect(req.request.method).toBe('POST');
    expect(req.request.body['path']).toBe('/data/sounds');
    req.flush({});

    expect(component.submitting).toBeFalse();
    expect(component.importStarted.emit).toHaveBeenCalled();
  });

  it('should show error on import failure', () => {
    flushImporters();
    component.selectImporter(mockImporters[0]);
    component.submit();

    httpMock.expectOne('/api/dataset/import/folder').flush(
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

    component.selectImporter(mockImporters[1]);
    const mockFile = new File(['data'], 'test.pkl');
    component.selectedFile = mockFile;
    component.submit();

    const req = httpMock.expectOne('/api/dataset/load-file');
    expect(req.request.method).toBe('POST');
    req.flush({});

    expect(component.importStarted.emit).toHaveBeenCalled();
  });

  // --- Demo picker tests ---

  it('should switch to demo view when openDemoPicker is called', () => {
    flushImporters();

    expect(component.view).toBe('picker');
    component.openDemoPicker();
    expect(component.view).toBe('demo');
    expect(component.demoLoading).toBeTrue();

    // Flush media types and demo list requests
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

    // Flush embedder fetch + refetch triggered by loadDemoEmbedders
    httpMock.expectOne(req =>
      req.url === '/api/embedders' && req.params.get('media_type') === 'audio',
    ).flush({ embedders: mockEmbedders });
    httpMock.expectOne(req =>
      req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'clap',
    ).flush({ datasets: mockDemos });

    expect(component.demoLoading).toBeFalse();
    expect(component.demos.length).toBe(2);
  });

  it('should build tabs from media types in demo data', () => {
    flushImporters();
    openAndFlushDemoPicker();

    expect(component.demoTabs).toEqual(['audio', 'image']);
    expect(component.activeTab).toBe('audio');
  });

  it('should filter demos by active tab', () => {
    flushImporters();
    openAndFlushDemoPicker();

    expect(component.filteredDemos.length).toBe(1);
    expect(component.filteredDemos[0].name).toBe('gtzan');

    component.selectDemoTab('image');

    // selectDemoTab triggers loadDemoEmbedders for the new tab
    httpMock.expectOne(req =>
      req.url === '/api/embedders' && req.params.get('media_type') === 'image',
    ).flush({ embedders: mockImageEmbedders });
    // refetchDemoStatuses fires
    httpMock.expectOne(req =>
      req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'clip',
    ).flush({ datasets: mockDemos });

    expect(component.filteredDemos.length).toBe(1);
    expect(component.filteredDemos[0].name).toBe('flowers102');
  });

  it('should sort demos by column', () => {
    flushImporters();
    openAndFlushDemoPicker();

    // Default sort is num_files ascending
    expect(component.demoSortKey).toBe('num_files');
    expect(component.demoSortAsc).toBeTrue();

    // Click same column to toggle direction
    component.sortDemoBy('num_files');
    expect(component.demoSortAsc).toBeFalse();

    // Click different column to switch
    component.sortDemoBy('label');
    expect(component.demoSortKey).toBe('label');
    expect(component.demoSortAsc).toBeTrue();
  });

  it('should emit demoSelected and close on demo selection', () => {
    flushImporters();
    spyOn(component.demoSelected, 'emit');
    spyOn(component.closed, 'emit');

    openAndFlushDemoPicker();

    component.selectDemo(mockDemos[0] as any);
    expect(component.demoSelected.emit).toHaveBeenCalled();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should go back from demo to picker view', () => {
    flushImporters();
    openAndFlushDemoPicker();

    expect(component.view).toBe('demo');
    component.back();
    expect(component.view).toBe('picker');
  });

  it('should render demo tab bar and table in template', () => {
    flushImporters();
    openAndFlushDemoPicker();

    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;

    const tabs = el.querySelectorAll('.demo-tab');
    expect(tabs.length).toBe(2);

    const rows = el.querySelectorAll('.demo-row');
    expect(rows.length).toBe(1); // Only audio tab active, one audio demo
  });

  it('should get correct tab label from media types', () => {
    flushImporters();
    component.mediaTypes = mockMediaTypes as any;

    expect(component.getTabLabel('audio')).toContain('Audio');
    expect(component.getTabLabel('unknown')).toBe('unknown');
  });

  it('should return correct status badge class', () => {
    flushImporters();
    expect(component.statusBadgeClass('ready')).toBe('badge-ready');
    expect(component.statusBadgeClass('needs_embedding')).toBe('badge-embedding');
    expect(component.statusBadgeClass('needs_download')).toBe('badge-download');
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
    openAndFlushDemoPicker(mockImageEmbedders);

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

  it('should re-fetch demos with default embedder on initial load', () => {
    flushImporters();
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

    // loadDemoEmbedders fires for the first tab
    httpMock.expectOne(req =>
      req.url === '/api/embedders' && req.params.get('media_type') === 'audio',
    ).flush({ embedders: mockEmbedders });

    // After embedders load, refetchDemoStatuses fires with the default embedder
    const refetchReq = httpMock.expectOne(req =>
      req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'clap',
    );
    expect(refetchReq.request.method).toBe('GET');
    refetchReq.flush({ datasets: mockDemos });
  });
});
