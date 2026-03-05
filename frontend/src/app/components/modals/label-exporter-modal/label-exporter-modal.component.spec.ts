import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { LabelExporterModalComponent } from './label-exporter-modal.component';

describe('LabelExporterModalComponent', () => {
  let component: LabelExporterModalComponent;
  let fixture: ComponentFixture<LabelExporterModalComponent>;
  let httpMock: HttpTestingController;

  const mockExporters = [
    { name: 'gui', label: 'Browser Download', description: 'Download as JSON' },
    { name: 'server_json_file', label: 'Server JSON', description: 'Save to server' },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
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

  function flushInit(): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/exporters').flush(mockExporters);
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should load exporters on init', () => {
    flushInit();
    expect(component.exporters.length).toBe(2);
    expect(component.loading).toBeFalse();
  });

  it('should show correct title for goods only', () => {
    component.goodsOnly = true;
    flushInit();
    expect(component.title).toContain('Goods');
  });

  it('should show default title', () => {
    flushInit();
    expect(component.title).toBe('Export Labels');
  });

  it('should export labels when exporter selected', () => {
    flushInit();
    spyOn(component.closed, 'emit');
    spyOn(component.exportComplete, 'emit');

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

  it('should render exporter cards', () => {
    flushInit();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const cards = el.querySelectorAll('.exporter-card');
    expect(cards.length).toBe(2);
  });

  it('should emit closed on close', () => {
    flushInit();
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
