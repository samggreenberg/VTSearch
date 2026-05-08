import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ExamplesEditorModalComponent } from './examples-editor-modal.component';

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
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ExamplesEditorModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    component.modelName = 'test-model';
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushInit(): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/detectors/test-model').flush({ examples: mockExamples });
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should load examples on init', () => {
    flushInit();
    expect(component.examples.length).toBe(3);
    expect(component.loading).toBeFalse();
  });

  it('should count good and bad examples', () => {
    flushInit();
    expect(component.goodExamples.length).toBe(2);
    expect(component.badExamples.length).toBe(1);
  });

  it('should remove an example', () => {
    flushInit();
    component.removeExample(0);
    expect(component.examples.length).toBe(2);
  });

  it('should save examples', () => {
    flushInit();
    spyOn(component.saved, 'emit');
    component.save();

    const req = httpMock.expectOne('/api/detectors/test-model/examples');
    expect(req.request.method).toBe('PUT');
    req.flush({});

    expect(component.saving).toBeFalse();
    expect(component.saved.emit).toHaveBeenCalled();
  });

  it('should show error on save failure', () => {
    flushInit();
    component.save();
    httpMock
      .expectOne('/api/detectors/test-model/examples')
      .flush({ error: 'Save failed' }, { status: 500, statusText: 'Error' });
    expect(component.error).toBe('Save failed');
  });

  it('should handle empty model name', () => {
    component.modelName = '';
    fixture.detectChanges();
    expect(component.loading).toBeFalse();
  });

  it('should handle load error', () => {
    fixture.detectChanges();
    httpMock
      .expectOne('/api/detectors/test-model')
      .flush({}, { status: 500, statusText: 'Error' });
    expect(component.loading).toBeFalse();
    expect(component.error).toBe('Failed to load examples');
  });

  it('should emit closed on close', () => {
    flushInit();
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
