import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { LoadSortModalComponent } from './load-sort-modal.component';

describe('LoadSortModalComponent', () => {
  let component: LoadSortModalComponent;
  let fixture: ComponentFixture<LoadSortModalComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LoadSortModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(LoadSortModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushInit(): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/server-media-files').flush({
      files: [
        { name: 'example', filename: 'example.wav', path: '/data/example.wav', size_bytes: 1000 },
        { name: 'sample', filename: 'sample.wav', path: '/data/sample.wav', size_bytes: 2000 },
      ],
    });
    httpMock.expectOne('/api/models/registry').flush({
      models: [
        { id: 'm1', name: 'My Model', media_type: 'audio', num_training: 12 },
        { id: 'm2', name: 'Untrained', media_type: 'audio', num_training: 0 },
      ],
    });
  }

  it('should create', () => {
    flushInit();
    expect(component).toBeTruthy();
  });

  it('should load server files and registry models on init', () => {
    flushInit();
    expect(component.serverMediaFiles.length).toBe(2);
    expect(component.serverMediaFiles[0].filename).toBe('example.wav');
    expect(component.registryModels.length).toBe(1);
    expect(component.registryModels[0].name).toBe('My Model');
    expect(component.loading).toBeFalse();
  });

  it('should emit modelSelected and close when a registry model is loaded', () => {
    flushInit();
    spyOn(component.modelSelected, 'emit');
    spyOn(component.closed, 'emit');

    component.loadRegistryModel({ id: 'm1', name: 'My Model', media_type: 'audio', num_training: 12 });

    expect(component.modelSelected.emit).toHaveBeenCalledWith('m1');
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should load server media and emit', () => {
    flushInit();
    spyOn(component.exampleSortStarted, 'emit');
    spyOn(component.closed, 'emit');

    component.loadServerMedia('example.wav');
    httpMock.expectOne('/api/example-sort-server').flush({ results: [], threshold: 0.5 });

    expect(component.exampleSortStarted.emit).toHaveBeenCalled();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should render file lists', () => {
    flushInit();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const items = el.querySelectorAll('.file-item');
    expect(items.length).toBe(3); // 1 trained model + 2 media files
  });

  it('should emit closed on close', () => {
    flushInit();
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
