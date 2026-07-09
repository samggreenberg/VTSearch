import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { LabelExporterModalComponent } from './label-exporter-modal.component';
import { ToastService } from '../../../services/toast.service';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleResource, settleZoneless } from '../../../testing/settle-resource';

describe('LabelExporterModalComponent', () => {
  let component: LabelExporterModalComponent;
  let fixture: ComponentFixture<LabelExporterModalComponent>;
  let httpMock: HttpTestingController;

  const mockExporters = [
    { name: 'gui', label: 'Browser Download', description: 'Download as JSON' },
    { name: 'server_json_file', label: 'Server JSON', description: 'Save to server' },
  ];

  beforeEach(async () => {
    await configureZoneless({
      imports: [LabelExporterModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(LabelExporterModalComponent);
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
    httpMock.expectOne('/api/exporters').flush(mockExporters);
    await settleResource();
  }

  it('should create', async () => {
    await flushInit();
    expect(component).toBeTruthy();
  });

  it('should load exporters on init', async () => {
    await flushInit();
    expect(component.exporters().length).toBe(2);
    expect(component.loading()).toBe(false);
  });

  it('should show correct title for goods only', async () => {
    component.goodsOnly = true;
    await flushInit();
    expect(component.title).toContain('Goods');
  });

  it('should show default title', async () => {
    await flushInit();
    expect(component.title).toBe('Export Labels');
  });

  it('should export labels when exporter selected', async () => {
    await flushInit();
    vi.spyOn(component.closed, 'emit');
    vi.spyOn(component.exportComplete, 'emit');

    component.selectExporter(mockExporters[0] as any);

    // Expect labels export request
    const labelsReq = httpMock.expectOne('/api/labels/export');
    labelsReq.flush({ labels: [] });

    // Expect export run request
    const exportReq = httpMock.expectOne('/api/exporters/export');
    exportReq.flush({ success: true });

    expect(component.exportComplete.emit).toHaveBeenCalled();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('fires a success toast so feedback survives the modal closing', async () => {
    await flushInit();
    const toast = TestBed.inject(ToastService);
    const successSpy = vi.spyOn(toast, 'success');

    component.selectExporter(mockExporters[1] as any);
    httpMock.expectOne('/api/labels/export').flush({
      labels: [
        { md5: 'a', label: 'good' },
        { md5: 'b', label: 'good' },
        { md5: 'c', label: 'good' },
      ],
    });
    httpMock.expectOne('/api/exporters/export').flush({ success: true });

    expect(successSpy).toHaveBeenCalledTimes(1);
    expect(successSpy.mock.calls[0][0].message).toBe('Exported 3 labels to server_json_file');
  });

  it('should render exporter cards', async () => {
    await flushInit();
    const el = fixture.nativeElement as HTMLElement;
    const cards = el.querySelectorAll('.exporter-card');
    expect(cards.length).toBe(2);
  });

  it('should emit closed on close', async () => {
    await flushInit();
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  // Zoneless staleness canary: a failed labels fetch lands in an HTTP subscribe
  // (an unpatched callback). It repaints only because `exportError` (merged into
  // the `error` computed) is a signal read in the template. Drive the failure and
  // assert the error text renders with no manual `detectChanges`.
  it('repaints the error text after a failed export (zoneless canary)', async () => {
    await flushInit();

    component.selectExporter(mockExporters[0] as any);
    httpMock.expectOne('/api/labels/export').flush(
      { error: 'nope' },
      { status: 500, statusText: 'Server Error' },
    );
    await settleZoneless(fixture);

    const err = fixture.nativeElement.querySelector('.error-text') as HTMLElement;
    expect(err).toBeTruthy();
    expect(err.textContent).toContain('Failed to fetch labels');
  });
});
