import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { NewDetectorModalComponent } from './new-model-modal.component';

describe('NewDetectorModalComponent', () => {
  let component: NewDetectorModalComponent;
  let fixture: ComponentFixture<NewDetectorModalComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [NewDetectorModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(NewDetectorModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    fixture.detectChanges();

    // Flush the media types request from ngOnInit
    httpMock.expectOne('/api/media-types').flush({
      media_types: [
        { type_id: 'audio', name: 'Audio', icon: 'audio', tab_title: 'Sounds' },
        { type_id: 'image', name: 'Image', icon: 'image', tab_title: 'Images' },
      ],
    });
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should populate media types from API', () => {
    expect(component.mediaTypes).toEqual(['audio', 'image']);
  });

  it('should show error when name is empty', () => {
    component.name = '';
    component.pendingText = 'test';
    component.submit();
    expect(component.error).toBe('Name is required');
  });

  it('should show error when no example provided', () => {
    component.name = 'Test Model';
    component.pendingText = '';
    component.submit();
    expect(component.error).toBe('An example (text or media) is required');
  });

  it('should accept pending text as text example on submit', () => {
    spyOn(component.created, 'emit');

    component.name = 'Dog Barks';
    component.mediaType = 'audio';
    component.pendingText = 'dog barking sounds';
    component.submit();

    const req = httpMock.expectOne('/api/detectors/registry');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.name).toBe('Dog Barks');
    expect(req.request.body.media_type).toBe('audio');
    expect(req.request.body.text_query).toBe('dog barking sounds');
    req.flush({ id: '123', name: 'Dog Barks' });

    expect(component.created.emit).toHaveBeenCalled();
  });

  it('should disable Create button when not ready', () => {
    component.name = '';
    component.pendingText = '';
    expect(component.canSubmitBlank).toBeFalse();

    component.name = 'Test';
    expect(component.canSubmitBlank).toBeFalse();

    component.pendingText = 'query';
    expect(component.canSubmitBlank).toBeTrue();
  });

  it('should disable media buttons when text is entered', () => {
    component.pendingText = '';
    expect(component.hasPendingText).toBeFalse();

    component.pendingText = 'some text';
    expect(component.hasPendingText).toBeTrue();
  });

  it('should clear pending text when media example is set', () => {
    component.pendingText = 'some text';
    component.exampleType = 'media';
    component.exampleValue = 'file.wav';
    component.exampleDisplay = 'file.wav';
    component.pendingText = '';

    expect(component.hasMediaExample).toBeTrue();
    expect(component.pendingText).toBe('');
  });

  it('should show server error on failure', () => {
    component.name = 'Test';
    component.pendingText = 'test';
    component.submit();

    httpMock.expectOne('/api/detectors/registry').flush(
      { error: 'Model already exists' },
      { status: 409, statusText: 'Conflict' },
    );

    expect(component.error).toBe('Model already exists');
  });

  it('should return media type icon', () => {
    expect(component.getMediaTypeIcon('audio')).toBe('audio');
    expect(component.getMediaTypeIcon('image')).toBe('image');
    expect(component.getMediaTypeIcon('unknown')).toBe('');
  });

  it('should emit closed on close', () => {
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
