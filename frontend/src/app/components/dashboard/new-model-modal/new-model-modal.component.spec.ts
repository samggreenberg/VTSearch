import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { NewModelModalComponent } from './new-model-modal.component';

describe('NewModelModalComponent', () => {
  let component: NewModelModalComponent;
  let fixture: ComponentFixture<NewModelModalComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [NewModelModalComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();

    fixture = TestBed.createComponent(NewModelModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
    fixture.detectChanges();

    // Flush the media types request from ngOnInit
    httpMock.expectOne('/api/media-types').flush({
      media_types: [
        { type_id: 'audio', name: 'Audio' },
        { type_id: 'image', name: 'Image' },
      ],
    });
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
    component.textQuery = 'test';
    component.submit();
    expect(component.error).toBe('Name is required');
  });

  it('should show error when query is empty', () => {
    component.name = 'Test Model';
    component.textQuery = '';
    component.submit();
    expect(component.error).toBe('At least one text query or example is required');
  });

  it('should submit to models registry API', () => {
    spyOn(component.created, 'emit');

    component.name = 'Dog Barks';
    component.mediaType = 'audio';
    component.textQuery = 'dog barking sounds';
    component.submit();

    const req = httpMock.expectOne('/api/models/registry');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.name).toBe('Dog Barks');
    expect(req.request.body.media_type).toBe('audio');
    expect(req.request.body.text_query).toBe('dog barking sounds');
    expect(req.request.body.trainable).toBe(true);
    req.flush({ id: '123', name: 'Dog Barks' });

    expect(component.created.emit).toHaveBeenCalled();
  });

  it('should show server error on failure', () => {
    component.name = 'Test';
    component.textQuery = 'test';
    component.submit();

    httpMock.expectOne('/api/models/registry').flush(
      { error: 'Model already exists' },
      { status: 409, statusText: 'Conflict' },
    );

    expect(component.error).toBe('Model already exists');
  });

  it('should emit closed on close', () => {
    spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });
});
