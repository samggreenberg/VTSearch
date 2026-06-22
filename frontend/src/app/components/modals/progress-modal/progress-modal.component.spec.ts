import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ProgressModalComponent } from './progress-modal.component';

describe('ProgressModalComponent', () => {
  let component: ProgressModalComponent;
  let fixture: ComponentFixture<ProgressModalComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ProgressModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(ProgressModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    component.ngOnDestroy();
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should set title based on metric', () => {
    component.metric = 'smart';
    expect(component.title).toContain('Error Cost');

    component.metric = 'stable';
    expect(component.title).toContain('Prediction Flip');

    component.metric = 'diverse';
    expect(component.title).toContain('Diversity Coverage');
  });

  it('should start in analyzing state', async () => {
    vi.useFakeTimers();
    try {
      component.metric = 'smart';
      fixture.detectChanges();
      expect(component.analyzing).toBe(true);

      // Progress now arrives over the `eval` SSE channel, not via HTTP polling.
      // The only HTTP call is the train-and-score POST, which returns a job
      // envelope; on a cache hit (status=done) the component applies the data
      // inline without polling.
      const trainReq = httpMock.expectOne('/api/eval/train-and-score');
      trainReq.flush({
        job_id: 'abc',
        status: 'done',
        metric: 'smart',
        error_cost: [{ num_labels: 5, error_cost: 0.5 }],
      });

      await vi.advanceTimersByTimeAsync(50);
      expect(component.analyzing).toBe(false);
      expect(component.chartData.length).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('should emit closed on close', () => {
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
