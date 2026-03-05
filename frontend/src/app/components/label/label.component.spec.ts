import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { LabelComponent } from './label.component';
import { MediaItem } from '../../models/api.models';

describe('LabelComponent', () => {
  let component: LabelComponent;
  let fixture: ComponentFixture<LabelComponent>;
  let httpMock: HttpTestingController;

  const mockMedias: MediaItem[] = [
    { id: 1, type: 'audio', duration: 5, file_size: 1024, filename: 'a.wav', category: 'test', md5: 'aaa' },
    { id: 2, type: 'image', duration: 0, file_size: 2048, filename: 'b.png', category: 'test', md5: 'bbb' },
  ];

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LabelComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
    fixture = TestBed.createComponent(LabelComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should load medias on init', () => {
    fixture.detectChanges();
    const req = httpMock.expectOne('/api/medias');
    req.flush(mockMedias);
    expect(component.medias.length).toBe(2);
    expect(component.selectedMedia?.id).toBe(1);
  });

  it('should select media when clicked', () => {
    component.medias = mockMedias;
    component.selectMedia(mockMedias[1]);
    expect(component.selectedMedia?.id).toBe(2);
  });

  it('should render media list', () => {
    fixture.detectChanges();
    const req = httpMock.expectOne('/api/medias');
    req.flush(mockMedias);
    fixture.detectChanges();

    const items = fixture.nativeElement.querySelectorAll('.media-list-item');
    expect(items.length).toBe(2);
    expect(items[0].textContent.trim()).toContain('a.wav');
    expect(items[1].textContent.trim()).toContain('b.png');
  });

  it('should render center panel', () => {
    fixture.detectChanges();
    const req = httpMock.expectOne('/api/medias');
    req.flush(mockMedias);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('vt-center-panel')).toBeTruthy();
  });

  it('should auto-advance on vote', () => {
    component.medias = mockMedias;
    component.selectedMedia = mockMedias[0];
    component.onMediaVoted({ id: 1, vote: 'good' });
    expect(component.selectedMedia?.id).toBe(2);
  });

  it('should not advance past last media', () => {
    component.medias = mockMedias;
    component.selectedMedia = mockMedias[1];
    component.onMediaVoted({ id: 2, vote: 'good' });
    expect(component.selectedMedia?.id).toBe(2);
  });
});
