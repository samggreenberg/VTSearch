import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { LabelImporterModalComponent } from './label-importer-modal.component';

describe('LabelImporterModalComponent', () => {
  let component: LabelImporterModalComponent;
  let fixture: ComponentFixture<LabelImporterModalComponent>;
  let httpMock: HttpTestingController;

  const mockImporters = [
    {
      name: 'server_json_file',
      display_name: 'JSON File',
      description: 'Import labels from a JSON file',
      fields: [{ key: 'filepath', field_type: 'text', label: 'File Path', required: true }],
    },
    {
      name: 'server_csv_file',
      display_name: 'CSV File',
      description: 'Import labels from a CSV file',
      fields: [{ key: 'filepath', field_type: 'text', label: 'File Path' }],
    },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LabelImporterModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(LabelImporterModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushInit(): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/label-importers').flush(mockImporters);
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should fetch importers on init', () => {
    flushInit();
    expect(component.importers.length).toBe(2);
  });

  it('should start in picker view', () => {
    flushInit();
    expect(component.view).toBe('picker');
  });

  it('should switch to form view on selection', () => {
    flushInit();
    component.selectImporter(mockImporters[0] as any);
    expect(component.view).toBe('form');
    expect(component.selectedImporter!.name).toBe('server_json_file');
  });

  it('should go back to picker', () => {
    flushInit();
    component.selectImporter(mockImporters[0] as any);
    component.back();
    expect(component.view).toBe('picker');
    expect(component.selectedImporter).toBeNull();
  });

  it('should submit form and emit imported', () => {
    flushInit();
    spyOn(component.imported, 'emit');
    component.selectImporter(mockImporters[0] as any);
    component.formValues['filepath'] = '/data/labels.json';
    component.submit();

    const req = httpMock.expectOne('/api/label-importers/import/server_json_file');
    expect(req.request.method).toBe('POST');
    req.flush({ applied: 5, message: 'Applied 5 labels' });

    expect(component.submitting).toBeFalse();
    expect(component.imported.emit).toHaveBeenCalled();
  });

  it('should show error on import failure', () => {
    flushInit();
    component.selectImporter(mockImporters[0] as any);
    component.submit();

    httpMock.expectOne('/api/label-importers/import/server_json_file').flush(
      { error: 'File not found' },
      { status: 404, statusText: 'Not Found' },
    );

    expect(component.error).toBe('File not found');
  });

  it('should render importer cards', () => {
    flushInit();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const cards = el.querySelectorAll('.importer-card');
    expect(cards.length).toBe(2);
  });

  it('should emit closed on close', () => {
    flushInit();
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should render Add media to Good and Add media to Bad buttons in picker view', () => {
    flushInit();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const addGoodBtn = el.querySelector('.btn-add-good');
    const addBadBtn = el.querySelector('.btn-add-bad');
    expect(addGoodBtn).toBeTruthy();
    expect(addBadBtn).toBeTruthy();
    expect(addGoodBtn!.textContent!.trim()).toBe('Add media to Good');
    expect(addBadBtn!.textContent!.trim()).toBe('Add media to Bad');
  });

  it('should have hidden file inputs for add-to-pile', () => {
    flushInit();
    fixture.detectChanges();
    const hiddenInputs = fixture.nativeElement.querySelectorAll('.hidden-file-input');
    expect(hiddenInputs.length).toBe(2);
  });
});
