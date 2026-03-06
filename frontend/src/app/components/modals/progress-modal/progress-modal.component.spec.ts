import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
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

  it('should start in analyzing state', fakeAsync(() => {
    component.metric = 'smart';
    fixture.detectChanges();
    expect(component.analyzing).toBeTrue();

    // Flush initial polling request
    const iterReq = httpMock.expectOne('/api/eval/voting-iterations');
    iterReq.flush({ progress: 0, total: 10, done: false });

    // Flush train-and-score
    const trainReq = httpMock.expectOne('/api/eval/train-and-score');
    trainReq.flush({ error_cost: [{ num_labels: 5, error_cost: 0.5 }] });

    tick(50);
    expect(component.analyzing).toBeFalse();
    expect(component.chartData.length).toBe(1);

    // Cleanup timer
    component.ngOnDestroy();
    httpMock.match('/api/eval/voting-iterations'); // flush remaining polls
  }));

  it('should emit closed on close', () => {
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
