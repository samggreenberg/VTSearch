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

  it('should render importer cards', () => {
    flushImporters();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const cards = el.querySelectorAll('.importer-card');
    expect(cards.length).toBe(2);
    expect(cards[0].textContent).toContain('Load from Folder');
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
});
