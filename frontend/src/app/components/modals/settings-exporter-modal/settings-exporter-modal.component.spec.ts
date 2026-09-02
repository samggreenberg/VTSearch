import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { SettingsExporterModalComponent } from './settings-exporter-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('SettingsExporterModalComponent', () => {
  let component: SettingsExporterModalComponent;
  let fixture: ComponentFixture<SettingsExporterModalComponent>;
  let httpMock: HttpTestingController;

  // A fields-bearing exporter so selecting it does NOT immediately submit (the
  // field-less branch auto-submits); keeps the picker→form transition explicit.
  const mockExporters = [
    {
      name: 'server_json_file',
      display_name: 'JSON File',
      description: 'Export settings to JSON',
      fields: [{ key: 'filepath', field_type: 'text', label: 'File Path', required: true }],
    },
  ];

  beforeEach(async () => {
    await configureZoneless({
      imports: [SettingsExporterModalComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(SettingsExporterModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  // The exporter list rides an eager `rxResource` whose loader runs in a root
  // effect. `TestBed.tick()` runs that effect to issue the GET — `whenStable()`
  // can't be used here: a loading `rxResource` holds the app unstable, so it
  // would deadlock waiting for the flush. After flushing, `settleResource()`
  // commits the resolved value (microtask + tick) with no manual `detectChanges`.
  async function flushInit(): Promise<void> {
    TestBed.tick();
    httpMock.expectOne('/api/settings-exporters').flush(mockExporters);
    await settleResource();
  }

  it('should create and render exporter cards', async () => {
    await flushInit();
    expect(component).toBeTruthy();
    expect(fixture.nativeElement.querySelectorAll('.picker-card').length).toBe(1);
  });

  it('does not auto-select the first option for a free-text combobox', async () => {
    await flushInit();
    const freeTextExporter = {
      name: 'free_text_exporter',
      fields: [{ key: 'q', field_type: 'select', options: ['a', 'b'], allow_free_text: true }],
    } as any;
    component.selectExporter(freeTextExporter);
    expect(component.formValues['q']).toBeUndefined();
  });

  // Zoneless staleness canary: the success message lands in an HTTP subscribe (an
  // unpatched callback) and repaints only because `successMessage` is a signal
  // read in the template. Drive a successful export and assert it renders with no
  // manual `detectChanges`.
  it('repaints the success message after a successful export (zoneless canary)', async () => {
    await flushInit();
    component.selectExporter(mockExporters[0] as any);
    await settleZoneless(fixture);

    component.submit();
    httpMock
      .expectOne('/api/settings-exporters/export')
      .flush({ message: 'Exported settings' });
    await settleZoneless(fixture);

    const ok = fixture.nativeElement.querySelector('.success-text') as HTMLElement;
    expect(ok).toBeTruthy();
    expect(ok.textContent).toContain('Exported settings');

    // Cancel the queued auto-close timer so it cannot leak past the test.
    component.close();
  });

  it('repaints the error text after a failed export (zoneless canary)', async () => {
    await flushInit();
    component.selectExporter(mockExporters[0] as any);
    await settleZoneless(fixture);

    component.submit();
    httpMock.expectOne('/api/settings-exporters/export').flush(
      { message: 'cannot write' },
      { status: 500, statusText: 'Server Error' },
    );
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('cannot write');
  });
});
