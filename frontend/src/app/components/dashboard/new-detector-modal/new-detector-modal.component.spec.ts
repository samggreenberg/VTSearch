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
    // EmbedderCapabilityService.ensureLoaded() in ngOnInit fetches the
    // embedder registry (drives the no-text warning).
    httpMock.expectOne('/api/embedders').flush([
      { name: 'clap', supports_text: true },
      { name: 'dinov3', supports_text: false },
    ]);
    // settingsState.load() in ngOnInit fetches settings.
    TestBed.tick(); // flush the SettingsStateService rxResource loader (root effect)
    httpMock.expectOne('/api/settings').flush({});
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
    vi.spyOn(component.created, 'emit');

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
    expect(component.canSubmitBlank).toBe(false);

    component.name = 'Test';
    expect(component.canSubmitBlank).toBe(false);

    component.pendingText = 'query';
    expect(component.canSubmitBlank).toBe(true);
  });

  it('should disable media buttons when text is entered', () => {
    component.pendingText = '';
    expect(component.hasPendingText).toBe(false);

    component.pendingText = 'some text';
    expect(component.hasPendingText).toBe(true);
  });

  it('should clear pending text when media example is set', () => {
    component.pendingText = 'some text';
    component.exampleType = 'media';
    component.exampleValue = 'file.wav';
    component.exampleDisplay = 'file.wav';
    component.pendingText = '';

    expect(component.hasMediaExample).toBe(true);
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
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('should not lock media type when no default is provided', () => {
    expect(component.mediaTypeLocked).toBe(false);
  });

  it('should ignore toggleMediaTypeDropdown when locked', () => {
    component.mediaTypeLocked = true;
    component.mediaTypeDropdownOpen = false;
    component.toggleMediaTypeDropdown();
    expect(component.mediaTypeDropdownOpen).toBe(false);
  });

  it('should open dropdown via toggle when unlocked', () => {
    component.mediaTypeLocked = false;
    component.mediaTypeDropdownOpen = false;
    component.toggleMediaTypeDropdown();
    expect(component.mediaTypeDropdownOpen).toBe(true);
  });

  it('should unlock media type on explicit unlock', () => {
    component.mediaTypeLocked = true;
    component.unlockMediaType();
    expect(component.mediaTypeLocked).toBe(false);
  });

  it('pre-fills the name from the text seed while the name is untouched', () => {
    component.onPendingTextInput('dog barking sounds');
    expect(component.pendingText).toBe('dog barking sounds');
    expect(component.name).toBe('dog barking sounds');
  });

  it('sanitises pasted whitespace when mirroring the seed into the name', () => {
    component.onPendingTextInput('   dog   barking\n  sounds   ');
    expect(component.pendingText).toBe('   dog   barking\n  sounds   ');
    expect(component.name).toBe('dog barking sounds');
  });

  it('warns about no-text datasets only for a text-hint-only detector', () => {
    // No warning before any text is entered.
    component.datasetEmbedder = 'dinov3';
    component.pendingText = '';
    expect(component.showNoTextWarning).toBe(false);

    // Text entered against a no-text embedder → warn.
    component.pendingText = 'a red car';
    expect(component.showNoTextWarning).toBe(true);

    // A media example seed (not text-only) → no warning even on a no-text dataset.
    component.exampleType = 'media';
    component.exampleValue = 'file.jpg';
    component.exampleDisplay = 'file.jpg';
    expect(component.showNoTextWarning).toBe(false);
  });

  it('does not warn when the dataset embedder can search by text', () => {
    component.datasetEmbedder = 'clap';
    component.pendingText = 'dog barking';
    expect(component.showNoTextWarning).toBe(false);
  });

  it('does not warn when the active dataset embedder is unknown', () => {
    component.datasetEmbedder = '';
    component.pendingText = 'dog barking';
    expect(component.showNoTextWarning).toBe(false);
  });

  it('stops mirroring once the user edits the name', () => {
    component.onPendingTextInput('dog');
    expect(component.name).toBe('dog');

    component.onNameInput('Dog Barks');
    expect(component.name).toBe('Dog Barks');
    expect(component.nameTouched).toBe(true);

    component.onPendingTextInput('cat meowing');
    expect(component.pendingText).toBe('cat meowing');
    expect(component.name).toBe('Dog Barks');
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
    httpMock.expectOne('/api/embedders').flush([]);
    // settingsState.load() in ngOnInit fetches settings.
    TestBed.tick(); // flush the SettingsStateService rxResource loader (root effect)
    httpMock.expectOne('/api/settings').flush({});
    // No /api/datasets/registry call when defaultMediaType is provided.
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should lock media type to the active dataset type', () => {
    expect(component.mediaType).toBe('image');
    expect(component.mediaTypeLocked).toBe(true);
  });
});
