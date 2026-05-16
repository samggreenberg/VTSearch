import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { NewDetectorModalComponent } from './new-detector-modal.component';

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
        { type_id: 'audio', name: 'Audio', icon: 'audio' },
        { type_id: 'image', name: 'Image', icon: 'image' },
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
    req.flush({ ok: true, detector: { id: '123', name: 'Dog Barks' } });

    expect(component.created.emit).toHaveBeenCalledWith('123');
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
      { error: 'Detector already exists' },
      { status: 409, statusText: 'Conflict' },
    );

    expect(component.error).toBe('Detector already exists');
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

  it('should not lock media type when no default is provided', () => {
    expect(component.mediaTypeLocked).toBeFalse();
  });

  it('should ignore toggleMediaTypeDropdown when locked', () => {
    component.mediaTypeLocked = true;
    component.mediaTypeDropdownOpen = false;
    component.toggleMediaTypeDropdown();
    expect(component.mediaTypeDropdownOpen).toBeFalse();
  });

  it('should open dropdown via toggle when unlocked', () => {
    component.mediaTypeLocked = false;
    component.mediaTypeDropdownOpen = false;
    component.toggleMediaTypeDropdown();
    expect(component.mediaTypeDropdownOpen).toBeTrue();
  });

  it('should unlock media type on explicit unlock', () => {
    component.mediaTypeLocked = true;
    component.unlockMediaType();
    expect(component.mediaTypeLocked).toBeFalse();
  });
});

describe('NewDetectorModalComponent with defaultMediaType', () => {
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
    component.defaultMediaType = 'image';
    httpMock = TestBed.inject(HttpTestingController);
    fixture.detectChanges();

    httpMock.expectOne('/api/media-types').flush({
      media_types: [
        { type_id: 'audio', name: 'Audio', icon: 'audio' },
        { type_id: 'image', name: 'Image', icon: 'image' },
      ],
    });
    // No /api/datasets/registry call when defaultMediaType is provided.
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should lock media type to the active dataset type', () => {
    expect(component.mediaType).toBe('image');
    expect(component.mediaTypeLocked).toBeTrue();
  });
});
