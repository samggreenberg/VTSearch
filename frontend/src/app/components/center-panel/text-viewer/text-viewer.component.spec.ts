import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TextViewerComponent } from './text-viewer.component';
import { MediaItem } from '../../../models/api.models';

describe('TextViewerComponent', () => {
  let component: TextViewerComponent;
  let fixture: ComponentFixture<TextViewerComponent>;
  let httpMock: HttpTestingController;

  const mockMedia: MediaItem = {
    id: 4,
    type: 'text',
    filename: 'test.txt',
    md5: 'jkl012',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TextViewerComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
    fixture = TestBed.createComponent(TextViewerComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should display loading initially', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Loading...');
  });

  it('should load and display text', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    fixture.detectChanges();

    const req = httpMock.expectOne('/api/medias/4/text');
    req.flush({ content: 'Hello world', word_count: 2, character_count: 11 });
    fixture.detectChanges();

    expect(component.text).toBe('Hello world');
    expect(fixture.nativeElement.textContent).toContain('Hello world');
  });

  it('should show error on failure', () => {
    component.media = mockMedia;
    component.ngOnChanges({
      media: { currentValue: mockMedia, previousValue: null, firstChange: true, isFirstChange: () => true },
    });
    fixture.detectChanges();

    const req = httpMock.expectOne('/api/medias/4/text');
    req.error(new ProgressEvent('error'));
    fixture.detectChanges();

    expect(component.text).toContain('Error');
  });
});
