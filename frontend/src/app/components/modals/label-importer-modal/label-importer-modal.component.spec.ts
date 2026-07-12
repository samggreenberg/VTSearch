import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { LabelImporterModalComponent } from './label-importer-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../../testing/settle-resource';

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
    await configureZoneless({
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

  // The importer list rides an eager `rxResource` whose loader runs in a root
  // effect. `TestBed.tick()` runs that effect to issue the GET — `whenStable()`
  // can't be used here: a loading `rxResource` holds the app unstable, so it
  // would deadlock waiting for the flush. After flushing, `settleResource()`
  // commits the resolved value (microtask + tick) with no manual `detectChanges`.
  async function flushInit(): Promise<void> {
    TestBed.tick();
    httpMock.expectOne('/api/label-importers').flush(mockImporters);
    await settleResource();
  }

  it('should create', async () => {
    await flushInit();
    expect(component).toBeTruthy();
  });

  it('should fetch importers on init', async () => {
    await flushInit();
    expect(component.importers().length).toBe(2);
  });

  it('should start in picker view', async () => {
    await flushInit();
    expect(component.view).toBe('picker');
  });

  it('should switch to form view on selection', async () => {
    await flushInit();
    component.selectImporter(mockImporters[0] as any);
    expect(component.view).toBe('form');
    expect(component.selectedImporter!.name).toBe('server_json_file');
  });

  it('should go back to picker', async () => {
    await flushInit();
    component.selectImporter(mockImporters[0] as any);
    component.back();
    expect(component.view).toBe('picker');
    expect(component.selectedImporter).toBeNull();
  });

  it('should submit form and emit imported', async () => {
    await flushInit();
    vi.spyOn(component.imported, 'emit');
    component.selectImporter(mockImporters[0] as any);
    component.formValues['filepath'] = '/data/labels.json';
    component.submit();

    const req = httpMock.expectOne('/api/label-importers/import/server_json_file');
    expect(req.request.method).toBe('POST');
    req.flush({ applied: 5, message: 'Applied 5 labels' });

    expect(component.submitting()).toBe(false);
    expect(component.imported.emit).toHaveBeenCalled();
    // The post-success auto-close timer must not leak past the test.
    component.close();
  });

  it('should show error on import failure', async () => {
    await flushInit();
    component.selectImporter(mockImporters[0] as any);
    component.submit();

    httpMock.expectOne('/api/label-importers/import/server_json_file').flush(
      { error: 'File not found' },
      { status: 404, statusText: 'Not Found' },
    );

    expect(component.error()).toBe('File not found');
  });

  it('should render importer cards', async () => {
    await flushInit();
    const el = fixture.nativeElement as HTMLElement;
    const cards = el.querySelectorAll('.picker-card');
    expect(cards.length).toBe(2);
  });

  it('should emit closed on close', async () => {
    await flushInit();
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should render Add media to Good and Add media to Bad buttons in picker view', async () => {
    await flushInit();
    const el = fixture.nativeElement as HTMLElement;
    const addGoodBtn = el.querySelector('.btn-add-good');
    const addBadBtn = el.querySelector('.btn-add-bad');
    expect(addGoodBtn).toBeTruthy();
    expect(addBadBtn).toBeTruthy();
    expect(addGoodBtn!.textContent!.trim()).toBe('Add media to Good');
    expect(addBadBtn!.textContent!.trim()).toBe('Add media to Bad');
  });

  it('should have hidden file inputs for add-to-pile', async () => {
    await flushInit();
    const hiddenInputs = fixture.nativeElement.querySelectorAll('.hidden-file-input');
    expect(hiddenInputs.length).toBe(2);
  });

  // Zoneless staleness canary: the submit error lands in an HTTP subscribe — an
  // unpatched callback under zoneless. It repaints only because `importError`
  // (merged into the `error` computed) is a signal read in the template. Drive a
  // failed import and assert the error text renders with no manual
  // `detectChanges`.
  it('repaints the error text after a failed import (zoneless canary)', async () => {
    await flushInit();
    component.selectImporter(mockImporters[0] as any);
    await settleZoneless(fixture);

    component.submit();
    httpMock.expectOne('/api/label-importers/import/server_json_file').flush(
      { error: 'Import blew up' },
      { status: 500, statusText: 'Server Error' },
    );
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('Import blew up');
  });
});
