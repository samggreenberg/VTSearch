import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TextViewerComponent } from './text-viewer.component';
import { Media } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('TextViewerComponent', () => {
  let component: TextViewerComponent;
  let fixture: ComponentFixture<TextViewerComponent>;
  let httpMock: HttpTestingController;

  const mockMedia: Media = {
    id: 4,
    media_type: 'text',
    filename: 'test.txt',
    md5: 'jkl012',
    custom_metadata: {},
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TextViewerComponent],
      providers: [...provideZoneless(), provideHttpClient(), provideHttpClientTesting()],
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

  it('should display loading initially', async () => {
    // setInput drives ngOnChanges (the real channel), which kicks off the fetch.
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);
    expect(fixture.nativeElement.textContent).toContain('Loading...');
    // Flush the in-flight request so afterEach httpMock.verify() sees none
    // dangling. The loading-state assertion above runs before the response.
    httpMock.expectOne('/api/medias/4/text').flush({ content: 'done' });
  });

  // Zoneless staleness path: `text` is written from an HTTP `.subscribe()`
  // callback (not a CD trigger). It is a signal, so the response repaints the
  // DOM with no manual detectChanges.
  it('should load and display text', async () => {
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);

    const req = httpMock.expectOne('/api/medias/4/text');
    req.flush({ content: 'Hello world', word_count: 2, character_count: 11 });
    await settleZoneless(fixture);

    expect(component.text()).toBe('Hello world');
    expect(fixture.nativeElement.textContent).toContain('Hello world');
  });

  it('should show error on failure', async () => {
    fixture.componentRef.setInput('media', mockMedia);
    await settleZoneless(fixture);

    const req = httpMock.expectOne('/api/medias/4/text');
    req.error(new ProgressEvent('error'));
    await settleZoneless(fixture);

    expect(component.text()).toContain('Error');
    expect(fixture.nativeElement.textContent).toContain('Error');
  });
});
