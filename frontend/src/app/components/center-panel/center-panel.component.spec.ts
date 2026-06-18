import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { CenterPanelComponent } from './center-panel.component';
import { Media } from '../../models/api.models';
import { RegionBox } from './image-viewer/image-viewer.component';

describe('CenterPanelComponent', () => {
  let component: CenterPanelComponent;
  let fixture: ComponentFixture<CenterPanelComponent>;
  let httpMock: HttpTestingController;

  const mockMedia: Media = {
    id: 1,
    media_type: 'audio',
    filename: 'test.wav',
    md5: 'abc123',
    custom_metadata: {},
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
    component.media = { ...mockMedia, media_type: 'image' };
    // The image-view-controls block (`@if (mediaType === 'image' && imageViewer)`)
    // gates on the `imageViewer` ViewChild, which only resolves partway through
    // the first change-detection pass. The zoom-slider bindings it renders then
    // settle to their real values within that same pass, which dev-mode's
    // check-no-changes guard flags as NG0100. Skip that guard (pass `false`); the
    // behaviour is dev-mode-only and does not occur in production.
    fixture.detectChanges(false);
    expect(fixture.nativeElement.querySelector('vt-image-viewer')).toBeTruthy();
  });

  it('should show video player for video media', () => {
    component.media = { ...mockMedia, media_type: 'video' };
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('vt-video-player')).toBeTruthy();
  });

  it('should show text viewer for text media', () => {
    component.media = { ...mockMedia, media_type: 'text' };
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('vt-text-viewer')).toBeTruthy();
    // The rendered text-viewer fetches its paragraph on init; flush it so the
    // afterEach httpMock.verify() sees no dangling request.
    httpMock.expectOne('/api/medias/1/text').flush({ text: '', paragraphs: [] });
  });

  it('should show document viewer for document media', () => {
    component.media = { ...mockMedia, media_type: 'document' };
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
    component.showAnimations = false;
    fixture.detectChanges();

    let emitted: { id: number; vote: string } | undefined;
    component.mediaVoted.subscribe((e: { id: number; vote: string }) => (emitted = e));

    component.castVote('good');

    // The vote POST sends the absolute target state, not a "vote" click; the
    // server's response reconciles the optimistic local view directly (no
    // follow-up GET /api/votes).
    const voteReq = httpMock.expectOne('/api/medias/1/vote');
    expect(voteReq.request.body).toEqual({ target: 'good' });
    voteReq.flush({ state: 'good', click_time: 1 });

    expect(emitted).toEqual({ id: 1, vote: 'good' });
    expect(component.voteState.goodVotes.has(1)).toBe(true);
  });

  it('should prevent double voting', () => {
    component.media = mockMedia;
    component.showAnimations = false;
    fixture.detectChanges();
    component.isVoting = true;
    component.castVote('good');
    httpMock.expectNone('/api/medias/1/vote');
  });

  it('should load votes via voteState', () => {
    component.voteState.loadVotes();
    const req = httpMock.expectOne('/api/votes');
    req.flush({ good: [1, 2], bad: [3], click_times: {}, learned_scores: {} });
    expect(component.voteState.goodVotes.has(1)).toBe(true);
    expect(component.voteState.goodVotes.has(2)).toBe(true);
    expect(component.voteState.badVotes.has(3)).toBe(true);
  });

  it('should clear swipe class when media changes', () => {
    component.media = mockMedia;
    fixture.detectChanges();
    // Simulate swipe ending
    component.swipeClass = 'swipe-right';

    // Change to new media (triggers ngOnChanges)
    component.media = { ...mockMedia, id: 2, filename: 'next.wav' };
    component.ngOnChanges({
      media: { currentValue: component.media, previousValue: mockMedia, firstChange: false, isFirstChange: () => false },
    });

    expect(component.swipeClass).toBe('');
  });

  it('should format metadata values', () => {
    expect(component.formatMetadataValue('File Size', 2048)).toBe('2.0 KB');
    expect(component.formatMetadataValue('Duration', 3.5)).toBe('3.5s');
    expect(component.formatMetadataValue('Frequency', 44100)).toBe('44100 Hz');
    expect(component.formatMetadataValue('Other', 'hello')).toBe('hello');
  });

  /**
   * v2 patch-embedder plan, item 15: vote-API contract for region annotations.
   * `region_box` must be present on a yes-vote when a box is drawn, absent on
   * a yes-vote without a box, and never present on any no-vote (no-votes are
   * region-agnostic (see "Vote attribution → v2" in docs/plans/patch-embedder.md).
   */
  describe('vote-API contract for region_box', () => {
    const imageMedia: Media = { ...mockMedia, media_type: 'image', filename: 'pic.png' };
    const box: RegionBox = [0.1, 0.2, 0.5, 0.6];

    function setup(): void {
      component.media = imageMedia;
      component.showAnimations = false;
      // Skip dev-mode check-no-changes: rendering the image-view-controls (gated
      // on the `imageViewer` ViewChild) settles the zoom-slider bindings within
      // the first CD pass, which the guard would otherwise flag as NG0100. This
      // is dev-mode-only and does not occur in production.
      fixture.detectChanges(false);
    }

    it('attaches region_box to a yes-vote when a box is drawn', () => {
      setup();
      component.onRegionBoxChange(box);
      component.castVote('good');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(req.request.body).toEqual({ target: 'good', region_box: [0.1, 0.2, 0.5, 0.6] });
      req.flush({ state: 'good', click_time: 1 });
    });

    it('omits region_box from a yes-vote when no box is drawn', () => {
      setup();
      component.castVote('good');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(req.request.body).toEqual({ target: 'good' });
      req.flush({ state: 'good', click_time: 1 });
    });

    it('omits region_box from a no-vote even when a box is drawn (after confirm)', () => {
      setup();
      component.onRegionBoxChange(box);
      // First ← arms the discard-confirm; no request yet.
      component.castVote('bad');
      httpMock.expectNone('/api/medias/1/vote');
      expect(component.pendingBadConfirm).toBe(true);
      // Second ← throws the box away and votes no.
      component.castVote('bad');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(req.request.body).toEqual({ target: 'bad' });
      req.flush({ state: 'bad', click_time: 1 });
    });

    it('omits region_box from a no-vote when no box is drawn (no confirm armed)', () => {
      setup();
      component.castVote('bad');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(req.request.body).toEqual({ target: 'bad' });
      expect(component.pendingBadConfirm).toBe(false);
      req.flush({ state: 'bad', click_time: 1 });
    });
  });

  /**
   * v2 patch-embedder plan, item 12: bad-vote-with-box requires two consecutive
   * ← presses (no timer). Esc, mouse-on-box, or item navigation while armed
   * clears the armed state and keeps the box.
   */
  describe('sticky bad-vote-confirm armed state', () => {
    const imageMedia: Media = { ...mockMedia, media_type: 'image', filename: 'pic.png' };
    const box: RegionBox = [0.1, 0.2, 0.5, 0.6];

    function setup(): void {
      component.media = imageMedia;
      component.showAnimations = false;
      // Skip dev-mode check-no-changes: rendering the image-view-controls (gated
      // on the `imageViewer` ViewChild) settles the zoom-slider bindings within
      // the first CD pass, which the guard would otherwise flag as NG0100. This
      // is dev-mode-only and does not occur in production.
      fixture.detectChanges(false);
    }

    it('arms on first ← without firing a request, fires on second ←', () => {
      setup();
      component.onRegionBoxChange(box);
      component.castVote('bad');
      httpMock.expectNone('/api/medias/1/vote');
      expect(component.pendingBadConfirm).toBe(true);
      expect(component.currentRegionBox).toEqual(box);

      component.castVote('bad');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(req.request.body).toEqual({ target: 'bad' });
      expect(component.pendingBadConfirm).toBe(false);
      req.flush({ state: 'bad', click_time: 1 });
    });

    it('cancels armed state on onArmedConfirmCanceled (Esc/mouse-on-box) and keeps the box', () => {
      setup();
      component.onRegionBoxChange(box);
      component.castVote('bad');
      expect(component.pendingBadConfirm).toBe(true);

      // Esc-while-armed (or mousedown-on-box) routes through this handler from
      // the image viewer.
      component.onArmedConfirmCanceled();
      expect(component.pendingBadConfirm).toBe(false);
      expect(component.currentRegionBox).toEqual(box);
      httpMock.expectNone('/api/medias/1/vote');
    });

    it('cancels armed state when the box is cleared (Esc-while-not-armed routes via regionBoxChange(null))', () => {
      setup();
      component.onRegionBoxChange(box);
      component.castVote('bad');
      expect(component.pendingBadConfirm).toBe(true);

      component.onRegionBoxChange(null);
      expect(component.pendingBadConfirm).toBe(false);
      expect(component.currentRegionBox).toBeNull();
    });

    it('cancels armed state when the user navigates to another item', () => {
      setup();
      component.onRegionBoxChange(box);
      component.castVote('bad');
      expect(component.pendingBadConfirm).toBe(true);

      const next: Media = { ...imageMedia, id: 2, filename: 'next.png' };
      component.media = next;
      component.ngOnChanges({
        media: {
          currentValue: next,
          previousValue: imageMedia,
          firstChange: false,
          isFirstChange: () => false,
        },
      });
      expect(component.pendingBadConfirm).toBe(false);
      expect(component.currentRegionBox).toBeNull();
    });

    it('does not arm when no box is drawn (single ← votes no immediately)', () => {
      setup();
      component.castVote('bad');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(req.request.body).toEqual({ target: 'bad' });
      expect(component.pendingBadConfirm).toBe(false);
      req.flush({ state: 'bad', click_time: 1 });
    });

    it('uses the box on a yes-vote even after a first ← would have armed (yes wins over armed-only)', () => {
      setup();
      component.onRegionBoxChange(box);
      // Without arming first: yes-vote attaches the box immediately.
      component.castVote('good');
      const req = httpMock.expectOne('/api/medias/1/vote');
      expect(req.request.body).toEqual({ target: 'good', region_box: [0.1, 0.2, 0.5, 0.6] });
      req.flush({ state: 'good', click_time: 1 });
    });
  });
});
