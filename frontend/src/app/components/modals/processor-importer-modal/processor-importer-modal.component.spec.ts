import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ProcessorImporterModalComponent } from './processor-importer-modal.component';

describe('ProcessorImporterModalComponent', () => {
  let component: ProcessorImporterModalComponent;
  let fixture: ComponentFixture<ProcessorImporterModalComponent>;
  let httpMock: HttpTestingController;

  const mockImporters = [
    {
      name: 'server_detector_file',
      display_name: 'Load Detector File',
      description: 'Import from a saved detector file',
      fields: [{ key: 'filepath', field_type: 'text', label: 'File Path', required: true }],
    },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ProcessorImporterModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ProcessorImporterModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushInit(): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/processor-importers').flush(mockImporters);
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should fetch importers on init', () => {
    flushInit();
    expect(component.importers.length).toBe(1);
  });

  it('should start in picker view', () => {
    flushInit();
    expect(component.view).toBe('picker');
  });

  it('should switch to form view on selection', () => {
    flushInit();
    component.selectImporter(mockImporters[0] as any);
    expect(component.view).toBe('form');
  });

  it('should go back to picker', () => {
    flushInit();
    component.selectImporter(mockImporters[0] as any);
    component.back();
    expect(component.view).toBe('picker');
  });

  it('should submit and emit imported', () => {
    flushInit();
    spyOn(component.imported, 'emit');
    component.selectImporter(mockImporters[0] as any);
    component.formValues['filepath'] = '/data/detector.json';
    component.submit();

    httpMock
      .expectOne('/api/processor-importers/import/server_detector_file')
      .flush({ message: 'Imported' });

    expect(component.imported.emit).toHaveBeenCalled();
  });

  it('should show error on failure', () => {
    flushInit();
    component.selectImporter(mockImporters[0] as any);
    component.submit();

    httpMock
      .expectOne('/api/processor-importers/import/server_detector_file')
      .flush({ error: 'Not found' }, { status: 404, statusText: 'Not Found' });

    expect(component.error).toBe('Not found');
  });

  it('should emit closed on close', () => {
    flushInit();
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
