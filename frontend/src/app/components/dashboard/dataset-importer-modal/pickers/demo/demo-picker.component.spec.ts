import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { DemoPickerComponent } from './demo-picker.component';
import { provideZoneless } from '../../../../../testing/zoneless-testbed';

describe('DemoPickerComponent', () => {
  let component: DemoPickerComponent;
  let fixture: ComponentFixture<DemoPickerComponent>;
  let httpMock: HttpTestingController;

  const demoImporter = { name: 'demo', picker_view: 'demo', fields: [] } as any;
  const mockMediaTypes = [
    { type_id: 'audio', name: 'Audio', icon: 'audio' },
    { type_id: 'image', name: 'Image', icon: 'image' },
  ];
  const mockDemos = [
    { name: 'gtzan', label: 'GTZAN', status: 'needs_download', ready: false, num_files: 1000, media_type: 'audio', pkl_embedder: '' },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DemoPickerComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(DemoPickerComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  function openAndFlush(): void {
    component.open(demoImporter);
    httpMock.expectOne('/api/media-types').flush({ media_types: mockMediaTypes });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && !req.params.get('embedder')).flush({ datasets: mockDemos });
    httpMock.expectOne(req => req.url === '/api/embedders' && req.params.get('media_type') === 'audio').flush({ embedders: [{ name: 'clap' }] });
    httpMock.expectOne(req => req.url === '/api/clippers' && req.params.get('media_type') === 'audio').flush({ clippers: [] });
    httpMock.expectOne(req => req.url === '/api/dataset/demo-list' && req.params.get('embedder') === 'clap').flush({ datasets: mockDemos });
  }

  it('open() fetches media types + demos and auto-selects the audio tab', () => {
    openAndFlush();
    expect(component.demoTabs()).toEqual(['audio']);
    expect(component.activeTab()).toBe('audio');
    expect(component.demos().length).toBe(1);
  });

  it('selecting a row records the selection without submitting', () => {
    openAndFlush();
    let emitted = false;
    component.demoDatasetSelected.subscribe(() => (emitted = true));

    component.selectDemo(mockDemos[0] as any);
    expect(component.selectedDemo()).toBe(mockDemos[0] as any);
    expect(component.demoDatasetName).toBe('GTZAN');
    expect(emitted).toBe(false);
  });

  it('submit() emits demoDatasetSelected with the composed payload', () => {
    openAndFlush();
    component.selectDemo(mockDemos[0] as any);

    let payload: any = null;
    component.demoDatasetSelected.subscribe((d) => (payload = d));
    component.submit();

    expect(payload).toBeTruthy();
    expect(payload.name).toBe('gtzan');
    expect(payload.dataset_name).toBe('GTZAN');
  });

  it('does nothing on submit() with no selection', () => {
    openAndFlush();
    let emitted = false;
    component.demoDatasetSelected.subscribe(() => (emitted = true));
    component.submit();
    expect(emitted).toBe(false);
  });
});
