import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { CenterPanelComponent } from './center-panel.component';
import { MediaItem } from '../../models/api.models';

describe('CenterPanelComponent', () => {
  let component: CenterPanelComponent;
  let fixture: ComponentFixture<CenterPanelComponent>;
  let httpMock: HttpTestingController;

  const mockMedia: MediaItem = {
    id: 1,
    type: 'audio',
    duration: 5.0,
    file_size: 1024,
    filename: 'test.wav',
    category: 'test',
    md5: 'abc123',
  };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CenterPanelComponent],
      providers: [provideHttpClient(), provideHttpClientTesting()],
    }).compileComponents();
    fixture = TestBed.createComponent(CenterPanelComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should show placeholder when no media selected', () => {
    fixture.detectChanges();
    expect(fixture.nativeElement.textContent).toContain('Select a media item to view');
  });

  it('should show audio player for audio media', () => {
    component.media = mockMedia;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('vt-audio-player')).toBeTruthy();
  });

  it('should show image viewer for image media', () => {
    component.media = { ...mockMedia, type: 'image' };
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('vt-image-viewer')).toBeTruthy();
  });

  it('should show video player for video media', () => {
    component.media = { ...mockMedia, type: 'video' };
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('vt-video-player')).toBeTruthy();
  });

  it('should show text viewer for paragraph media', () => {
    component.media = { ...mockMedia, type: 'paragraph' };
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('vt-text-viewer')).toBeTruthy();
  });

  it('should show document viewer for document media', () => {
    component.media = { ...mockMedia, type: 'document' };
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('vt-document-viewer')).toBeTruthy();
  });

  it('should show voting overlay when media is selected', () => {
    component.media = mockMedia;
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('vt-voting-overlay')).toBeTruthy();
  });

  it('should display metadata', () => {
    component.media = mockMedia;
    fixture.detectChanges();
    const text = fixture.nativeElement.textContent;
    expect(text).toContain('test.wav');
    expect(text).toContain('MD5');
    expect(text).toContain('abc123');
  });

  it('should send vote request on castVote', () => {
    component.media = mockMedia;
    component.swipeAnimation = false;
    fixture.detectChanges();

    let emitted: { id: number; vote: string } | undefined;
    component.mediaVoted.subscribe((e: { id: number; vote: string }) => (emitted = e));

    component.castVote('good');

    const voteReq = httpMock.expectOne('/api/medias/1/vote');
    expect(voteReq.request.body).toEqual({ vote: 'good' });
    voteReq.flush({ ok: true });

    const votesReq = httpMock.expectOne('/api/votes');
    votesReq.flush({ good: [1], bad: [], click_times: {}, learned_scores: {} });

    expect(emitted).toEqual({ id: 1, vote: 'good' });
    expect(component.voteState.goodVotes.has(1)).toBeTrue();
  });

  it('should prevent double voting', () => {
    component.media = mockMedia;
    component.swipeAnimation = false;
    fixture.detectChanges();
    component.isVoting = true;
    component.castVote('good');
    httpMock.expectNone('/api/medias/1/vote');
  });

  it('should load votes via voteState', () => {
    component.voteState.loadVotes();
    const req = httpMock.expectOne('/api/votes');
    req.flush({ good: [1, 2], bad: [3], click_times: {}, learned_scores: {} });
    expect(component.voteState.goodVotes.has(1)).toBeTrue();
    expect(component.voteState.goodVotes.has(2)).toBeTrue();
    expect(component.voteState.badVotes.has(3)).toBeTrue();
  });

  it('should format metadata values', () => {
    expect(component.formatMetadataValue('File Size', 2048)).toBe('2.0 KB');
    expect(component.formatMetadataValue('Duration', 3.5)).toBe('3.5s');
    expect(component.formatMetadataValue('Frequency', 44100)).toBe('44100 Hz');
    expect(component.formatMetadataValue('Other', 'hello')).toBe('hello');
  });
});
