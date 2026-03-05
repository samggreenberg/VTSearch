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
        { name: 'path', type: 'text', label: 'Folder Path', required: true },
        { name: 'media_type', type: 'select', label: 'Media Type', default: 'audio' },
      ],
    },
    {
      name: 'pickle',
      label: 'Load from File',
      description: 'Load a .pkl dataset file',
      fields: [{ name: 'file', type: 'file', label: 'Dataset File', required: true }],
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
    },
    {
      name: 'flowers102',
      label: 'Oxford Flowers 102',
      status: 'ready',
      ready: true,
      num_files: 8189,
      download_size_mb: 330,
      description: '102 flower categories',
      media_type: 'images',
      num_categories: 102,
    },
  ];

  const mockMediaTypes = [
    { type_id: 'audio', name: 'Audio', icon: '\uD83C\uDFB5', tab_title: 'Audio' },
    { type_id: 'images', name: 'Images', icon: '\uD83D\uDDBC\uFE0F', tab_title: 'Images' },
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

  it('should render importer cards plus demo card', () => {
    flushImporters();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const cards = el.querySelectorAll('.importer-card');
    // 2 importers + 1 demo card
    expect(cards.length).toBe(3);
    expect(cards[0].textContent).toContain('Load from Folder');
    expect(cards[2].textContent).toContain('Load Demo Dataset');
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
    component.openDemoPicker();

    expect(component.view).toBe('demo');
    expect(component.demoLoading).toBeTrue();

    // Flush media types and demo list requests
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

    expect(component.demoLoading).toBeFalse();
    expect(component.demos.length).toBe(2);
  });

  it('should build tabs from media types in demo data', () => {
    flushImporters();
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

    expect(component.demoTabs).toEqual(['audio', 'images']);
    expect(component.activeTab).toBe('audio');
  });

  it('should filter demos by active tab', () => {
    flushImporters();
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

    expect(component.filteredDemos.length).toBe(1);
    expect(component.filteredDemos[0].name).toBe('gtzan');

    component.selectDemoTab('images');
    expect(component.filteredDemos.length).toBe(1);
    expect(component.filteredDemos[0].name).toBe('flowers102');
  });

  it('should sort demos by column', () => {
    flushImporters();
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

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

    component.openDemoPicker();
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

    component.selectDemo(mockDemos[0] as any);
    expect(component.demoSelected.emit).toHaveBeenCalledWith(mockDemos[0] as any);
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should go back from demo to picker view', () => {
    flushImporters();
    component.openDemoPicker();
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

    expect(component.view).toBe('demo');
    component.back();
    expect(component.view).toBe('picker');
  });

  it('should render demo tab bar and table in template', () => {
    flushImporters();
    component.openDemoPicker();

    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne('/api/dataset/demo-list').flush({ datasets: mockDemos });

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
});
