import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ExamplesEditorModalComponent } from './examples-editor-modal.component';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('ExamplesEditorModalComponent', () => {
  let component: ExamplesEditorModalComponent;
  let fixture: ComponentFixture<ExamplesEditorModalComponent>;
  let httpMock: HttpTestingController;

  const mockExamples = [
    { type: 'good', label: 'Song A' },
    { type: 'good', label: 'Song B' },
    { type: 'bad', label: 'Noise C' },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ExamplesEditorModalComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ExamplesEditorModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    fixture.componentRef.setInput('modelName', 'test-model');
  });

  afterEach(() => {
    httpMock.verify();
  });

  // Settle runs ngOnInit (issues the GET), flush, settle again so the loaded
  // examples repaint. No manual detectChanges.
  async function flushInit(): Promise<void> {
    await settleZoneless(fixture);
    httpMock.expectOne('/api/detectors/test-model').flush({ examples: mockExamples });
    await settleZoneless(fixture);
  }

  it('should create', async () => {
    await flushInit();
    expect(component).toBeTruthy();
  });

  // Zoneless canary: `examples`/`loading` are written from the load subscribe (an
  // unpatched callback). As signals they repaint the grid, so the loaded cards
  // render with no manual detectChanges.
  it('should load examples on init and render the cards', async () => {
    await flushInit();
    expect(component.examples().length).toBe(3);
    expect(component.loading()).toBe(false);
    expect(fixture.nativeElement.querySelectorAll('.example-card').length).toBe(3);
  });

  it('should count good and bad examples', async () => {
    await flushInit();
    expect(component.goodExamples.length).toBe(2);
    expect(component.badExamples.length).toBe(1);
  });

  it('should remove an example', async () => {
    await flushInit();
    component.removeExample(0);
    expect(component.examples().length).toBe(2);
  });

  it('should save examples', async () => {
    await flushInit();
    vi.spyOn(component.saved, 'emit');
    component.save();

    const req = httpMock.expectOne('/api/detectors/test-model/examples');
    expect(req.request.method).toBe('PUT');
    req.flush({});

    expect(component.saving()).toBe(false);
    expect(component.saved.emit).toHaveBeenCalled();
    // Cancel the queued auto-close timer so it cannot leak past the test.
    component.close();
  });

  it('should show error on save failure', async () => {
    await flushInit();
    component.save();
    httpMock
      .expectOne('/api/detectors/test-model/examples')
      .flush({ error: 'Save failed' }, { status: 500, statusText: 'Error' });
    expect(component.error()).toBe('Save failed');
  });

  it('should handle empty model name', async () => {
    fixture.componentRef.setInput('modelName', '');
    await settleZoneless(fixture);
    expect(component.loading()).toBe(false);
  });

  it('should handle load error', async () => {
    await settleZoneless(fixture);
    httpMock
      .expectOne('/api/detectors/test-model')
      .flush({}, { status: 500, statusText: 'Error' });
    await settleZoneless(fixture);
    expect(component.loading()).toBe(false);
    expect(component.error()).toBe('Failed to load examples');
  });

  it('should emit closed on close', async () => {
    await flushInit();
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
