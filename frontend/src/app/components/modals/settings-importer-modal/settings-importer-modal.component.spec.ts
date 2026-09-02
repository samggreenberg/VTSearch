import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { SettingsImporterModalComponent } from './settings-importer-modal.component';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../../testing/settle-resource';
import { provideHttpTesting } from '../../../testing/test-providers';

describe('SettingsImporterModalComponent', () => {
  let component: SettingsImporterModalComponent;
  let fixture: ComponentFixture<SettingsImporterModalComponent>;
  let httpMock: HttpTestingController;

  const mockImporters = [
    {
      name: 'server_json_file',
      display_name: 'JSON File',
      description: 'Import settings from JSON',
      fields: [{ key: 'filepath', field_type: 'text', label: 'File Path', required: true }],
    },
  ];

  beforeEach(async () => {
    await configureZoneless({
      imports: [SettingsImporterModalComponent],
      providers: [...provideHttpTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(SettingsImporterModalComponent);
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
    httpMock.expectOne('/api/settings-importers').flush(mockImporters);
    await settleResource();
  }

  it('should create and render importer cards', async () => {
    await flushInit();
    expect(component).toBeTruthy();
    expect(fixture.nativeElement.querySelectorAll('.picker-card').length).toBe(1);
  });

  it('does not auto-select the first option for a free-text combobox', async () => {
    await flushInit();
    const freeTextImporter = {
      name: 'free_text_importer',
      fields: [{ key: 'q', field_type: 'select', options: ['a', 'b'], allow_free_text: true }],
    } as any;
    component.selectImporter(freeTextImporter);
    expect(component.formValues['q']).toBeUndefined();
  });

  // Zoneless staleness canary: the success message lands in an HTTP subscribe (an
  // unpatched callback) and repaints only because `successMessage` is a signal
  // read in the template. Drive a successful import and assert it renders with no
  // manual `detectChanges`.
  it('repaints the success message after a successful import (zoneless canary)', async () => {
    await flushInit();
    component.selectImporter(mockImporters[0] as any);
    await settleZoneless(fixture);

    component.submit();
    httpMock
      .expectOne('/api/settings-importers/import/server_json_file')
      .flush({ message: 'Imported 7 settings' });
    await settleZoneless(fixture);

    const ok = fixture.nativeElement.querySelector('.success-text') as HTMLElement;
    expect(ok).toBeTruthy();
    expect(ok.textContent).toContain('Imported 7 settings');

    // Cancel the queued auto-close timer so it cannot leak past the test.
    component.close();
  });

  it('repaints the error text after a failed import (zoneless canary)', async () => {
    await flushInit();
    component.selectImporter(mockImporters[0] as any);
    await settleZoneless(fixture);

    component.submit();
    httpMock.expectOne('/api/settings-importers/import/server_json_file').flush(
      { message: 'bad file' },
      { status: 400, statusText: 'Bad Request' },
    );
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('bad file');
  });
});
