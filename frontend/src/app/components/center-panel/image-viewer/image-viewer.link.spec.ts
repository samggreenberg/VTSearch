import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ElementRef } from '@angular/core';

import { ImageViewerComponent } from './image-viewer.component';
import { FaceLandmarks, ViewLink, faceAnchor } from './face-link';
import { ActiveContextService } from '../../../services/active-context.service';
import { Media } from '../../../models/api.models';
import { configureZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

/**
 * Two viewers on one link, as the Face compare layout puts them (#5647): the
 * candidate's leads, the reference's follows, sized and placed so the two faces
 * match, zooming and panning together and turning on their own.
 *
 * Each viewer is a 100 × 100 pane, the lead's at x 0–100 on screen and the
 * follower's at x 200–300. The lead's image is square and fills its pane, its
 * face in the middle; the follower's is 1:2, drawn 50 × 100, its face small and
 * up in the top-left corner.
 */
describe('ImageViewerComponent linked (Face compare registration)', () => {
  const CENTRED: FaceLandmarks = [
    [0.4, 0.4],
    [0.6, 0.4],
    [0.5, 0.5],
    [0.45, 0.6],
    [0.55, 0.6],
  ];
  const CORNERED: FaceLandmarks = [
    [0.2, 0.2],
    [0.4, 0.2],
    [0.3, 0.25],
    [0.25, 0.3],
    [0.35, 0.3],
  ];
  /** Eyes at (0.6, 0.5), a little lower and larger than CORNERED. */
  const LOWER: FaceLandmarks = [
    [0.5, 0.5],
    [0.7, 0.5],
    [0.6, 0.6],
    [0.55, 0.7],
    [0.65, 0.7],
  ];

  interface Pane {
    fixture: ComponentFixture<ImageViewerComponent>;
    viewer: ImageViewerComponent;
    left: number;
    natural: [number, number];
  }

  let link: ViewLink;
  let lead: Pane;
  let follower: Pane;

  const media = (id: number): Media => ({ id, media_type: 'face', filename: `face_${id}.jpg`, md5: 'm', custom_metadata: {} });

  beforeEach(async () => {
    await configureZoneless({
      imports: [ImageViewerComponent],
      providers: [ActiveContextService],
    }).compileComponents();
    link = new ViewLink();
    lead = make('lead', 0, [400, 400]);
    follower = make('follow', 200, [200, 400]);
  });

  afterEach(() => {
    lead.fixture.destroy();
    follower.fixture.destroy();
  });

  function make(role: 'lead' | 'follow', left: number, natural: [number, number]): Pane {
    const fixture = TestBed.createComponent(ImageViewerComponent);
    const viewer = fixture.componentInstance;
    const rect = { left, top: 0, width: 100, height: 100, right: left + 100, bottom: 100, x: left, y: 0, toJSON: () => ({}) };
    (viewer as unknown as { wrapRef: () => ElementRef<HTMLDivElement> }).wrapRef = () =>
      ({
        nativeElement: { clientWidth: 100, clientHeight: 100, getBoundingClientRect: () => rect } as unknown as HTMLDivElement,
      }) as ElementRef<HTMLDivElement>;
    fixture.componentRef.setInput('link', link);
    fixture.componentRef.setInput('linkAs', role);
    // An image not yet loaded, so a tick of the other pane finds this one whole.
    fixture.componentRef.setInput('media', media(role === 'lead' ? 90 : 91));
    return { fixture, viewer, left, natural };
  }

  /** Show *id*'s image, with *landmarks*, as far as its load. */
  function open(pane: Pane, id: number, landmarks: FaceLandmarks | null): void {
    show(pane, id, landmarks);
    loaded(pane);
  }

  /** Hand *pane* a new image, not yet loaded. */
  function show(pane: Pane, id: number, landmarks: FaceLandmarks | null): void {
    pane.fixture.componentRef.setInput('media', media(id));
    pane.fixture.componentRef.setInput('landmarks', landmarks);
    TestBed.tick();
  }

  function loaded(pane: Pane, natural = pane.natural): void {
    (pane.viewer as unknown as { imageRef: () => ElementRef<HTMLImageElement> }).imageRef = () =>
      ({ nativeElement: { naturalWidth: natural[0], naturalHeight: natural[1] } as HTMLImageElement }) as ElementRef<HTMLImageElement>;
    pane.viewer.onImageLoad();
    TestBed.tick();
  }

  /** Both panes showing their faces. */
  function openBoth(): void {
    open(lead, 1, CENTRED);
    open(follower, 2, CORNERED);
  }

  /** Which image point is under (*dx*, *dy*) from the middle of *pane*. */
  function under(pane: Pane, dx: number, dy: number): { x: number; y: number } {
    return pane.viewer.screenToImageNormalized({ clientX: pane.left + 50 + dx, clientY: 50 + dy } as MouseEvent)!;
  }

  /** Where image point (*u*, *v*) is drawn, from the middle of *pane*. */
  function onScreen(pane: Pane, u: number, v: number): [number, number] {
    const v_ = pane.viewer;
    const rad = (v_.rotation() * Math.PI) / 180;
    const x = (u - 0.5) * v_.renderedW();
    const y = (v - 0.5) * v_.renderedH();
    return [
      v_.panX() + v_.zoom() * (x * Math.cos(rad) - y * Math.sin(rad)),
      v_.panY() + v_.zoom() * (x * Math.sin(rad) + y * Math.cos(rad)),
    ];
  }

  /** The point of *pane*'s face under (*dx*, *dy*) from its middle, in face units. */
  function faceUnder(pane: Pane, landmarks: FaceLandmarks, dx = 0, dy = 0): [number, number] {
    const anchor = faceAnchor(landmarks)!;
    const [w, h] = pane.natural;
    const unit = Math.hypot((anchor.mouth[0] - anchor.eyes[0]) * w, (anchor.mouth[1] - anchor.eyes[1]) * h);
    const p = under(pane, dx, dy);
    return [((p.x - anchor.eyes[0]) * w) / unit, ((p.y - anchor.eyes[1]) * h) / unit];
  }

  function expectClose(actual: readonly number[], expected: readonly number[], digits = 6): void {
    expect(actual.length).toBe(expected.length);
    actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i], digits));
  }

  /** The two faces' eye midpoints, and their mouth midpoints, are drawn in the same place. */
  function expectMatched(followerMarks: FaceLandmarks = CORNERED): void {
    const leadAnchor = faceAnchor(CENTRED)!;
    const anchor = faceAnchor(followerMarks)!;
    expectClose(onScreen(follower, ...anchor.eyes), onScreen(lead, ...leadAnchor.eyes));
    expectClose(onScreen(follower, ...anchor.mouth), onScreen(lead, ...leadAnchor.mouth));
  }

  it('sizes and places the reference so the two faces match', () => {
    openBoth();
    expect(follower.viewer.linked()).toBe(true);
    // Twice the size, to give its small face the candidate's eye-to-mouth length...
    expect(follower.viewer.zoom()).toBeCloseTo(2, 9);
    // ...and moved so its eyes and mouth sit where the candidate's do.
    expectMatched();
    // The candidate, which leads, is shown as it always was.
    expect(lead.viewer.zoom()).toBe(1);
    expect(lead.viewer.panX()).toBe(0);
    expect(lead.viewer.linked()).toBe(false);
  });

  it('lines the faces up whichever image loads first', () => {
    open(follower, 2, CORNERED);
    open(lead, 1, CENTRED);
    expectMatched();
  });

  it('zooms and pans the reference with the candidate, and repaints it', async () => {
    openBoth();
    lead.viewer.zoomIn();
    lead.viewer.onWheel({ deltaY: -1, clientX: 70, clientY: 30, preventDefault: () => {} } as unknown as WheelEvent);
    await settleZoneless(follower.fixture);
    expectMatched();
    // The same face point under the same place in each pane.
    expectClose(faceUnder(follower, CORNERED, 20, -20), faceUnder(lead, CENTRED, 20, -20));
    // Drawn, under zoneless, with nothing but the lead's gesture to schedule it.
    const img = follower.fixture.nativeElement.querySelector('img') as HTMLImageElement;
    expect(img.style.transform).toBe(follower.viewer.imageTransform);
    expect(img.style.transform).toContain(`scale(${follower.viewer.zoom()})`);
  });

  it('hands a zoom on the reference to the candidate, keeping the point under the cursor', () => {
    openBoth();
    const before = under(follower, 4, 4);
    follower.viewer.onWheel({ deltaY: -1, clientX: 254, clientY: 54, preventDefault: () => {} } as unknown as WheelEvent);
    expect(lead.viewer.zoom()).toBeCloseTo(1.15, 9);
    expectClose(Object.values(under(follower, 4, 4)), Object.values(before));
    expectMatched();
  });

  it('pans both on a drag of the reference, and stops where the candidate\'s image ends', () => {
    openBoth();
    lead.viewer.onZoomInput({ target: { value: '2' } } as unknown as Event);
    follower.viewer.onMouseDown({ button: 0, clientX: 250, clientY: 50, preventDefault: () => {} } as unknown as MouseEvent);
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 260, clientY: 45 }));
    expectClose([lead.viewer.panX(), lead.viewer.panY()], [10, -5]);
    expectMatched();

    // Far past the candidate's left edge: it stops there, and the reference with it.
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 400, clientY: 50 }));
    window.dispatchEvent(new MouseEvent('mouseup'));
    expect(lead.viewer.panX()).toBeCloseTo(50, 9);
    expect(under(lead, -50, 0).x).toBeCloseTo(0, 9);
    expectMatched();
  });

  it('does not pan the reference while the candidate fits its pane', () => {
    openBoth();
    expect(follower.viewer.wrapCursor).toBe('');
    follower.viewer.onMouseDown({ button: 0, clientX: 250, clientY: 50, preventDefault: () => {} } as unknown as MouseEvent);
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 280, clientY: 50 }));
    window.dispatchEvent(new MouseEvent('mouseup'));
    expect(lead.viewer.panX()).toBe(0);
    expectMatched();

    lead.viewer.zoomIn();
    expect(follower.viewer.wrapCursor).toBe('grab');
  });

  it('turns each pane on its own, the same face point in the middle of both', () => {
    openBoth();
    lead.viewer.onZoomInput({ target: { value: '2' } } as unknown as Event);
    lead.viewer.onWheel({ deltaY: 1, clientX: 80, clientY: 20, preventDefault: () => {} } as unknown as WheelEvent);
    const leadView = { zoom: lead.viewer.zoom(), panX: lead.viewer.panX(), panY: lead.viewer.panY() };

    follower.viewer.rotateRight();
    expect(follower.viewer.rotation()).toBe(90);
    expect(lead.viewer.rotation()).toBe(0);
    expectClose([lead.viewer.zoom(), lead.viewer.panX(), lead.viewer.panY()], [leadView.zoom, leadView.panX, leadView.panY]);
    expectClose(faceUnder(follower, CORNERED), faceUnder(lead, CENTRED));

    lead.viewer.rotateLeft();
    expect(follower.viewer.rotation()).toBe(90);
    expectClose(faceUnder(follower, CORNERED), faceUnder(lead, CENTRED));
    // A zoom on the turned reference still zooms both about their middles.
    follower.viewer.zoomIn();
    expectClose(faceUnder(follower, CORNERED), faceUnder(lead, CENTRED));
  });

  it('shows the candidate\'s zoom on the reference\'s toolbar, and its slider moves both', () => {
    openBoth();
    expect(follower.viewer.zoomLabel()).toBe('1×');
    lead.viewer.onZoomInput({ target: { value: '2.5' } } as unknown as Event);
    expect(follower.viewer.shownZoom()).toBe(2.5);
    expect(follower.viewer.zoomLabel()).toBe('2.5×');

    follower.viewer.onZoomInput({ target: { value: '3' } } as unknown as Event);
    expect(lead.viewer.zoom()).toBeCloseTo(3, 9);
    expect(follower.viewer.zoomLabel()).toBe('3×');
    expectMatched();
  });

  it('stops at the candidate\'s zoom range', () => {
    openBoth();
    follower.viewer.onZoomInput({ target: { value: '9' } } as unknown as Event);
    expect(lead.viewer.zoom()).toBe(lead.viewer.maxZoom);
    expect(follower.viewer.shownZoom()).toBe(lead.viewer.maxZoom);
    follower.viewer.onZoomInput({ target: { value: '0.2' } } as unknown as Event);
    expect(lead.viewer.zoom()).toBeCloseTo(1, 9);
    expectMatched();
  });

  it('takes both back to the whole faces on the reference\'s Reset view, the candidate keeping its turn', () => {
    openBoth();
    lead.viewer.rotateRight();
    lead.viewer.onZoomInput({ target: { value: '3' } } as unknown as Event);
    follower.viewer.rotateLeft();

    follower.viewer.resetView();
    expect(lead.viewer.zoom()).toBe(1);
    expectClose([lead.viewer.panX(), lead.viewer.panY()], [0, 0]);
    expect(lead.viewer.rotation()).toBe(90);
    expect(follower.viewer.rotation()).toBe(0);
    expect(follower.viewer.zoom()).toBeCloseTo(2, 9);
  });

  it('takes both back to the whole faces on a double-click at the cap', () => {
    openBoth();
    lead.viewer.onZoomInput({ target: { value: '5' } } as unknown as Event);
    follower.viewer.onDoubleClick({ button: 0, clientX: 230, clientY: 30, preventDefault: () => {} } as unknown as MouseEvent);
    expect(lead.viewer.zoom()).toBe(1);
    expectMatched();
  });

  it('leaves the reference on its own when it has no landmarks', () => {
    open(lead, 1, CENTRED);
    open(follower, 2, null);
    expect(follower.viewer.linked()).toBe(false);
    expect(follower.viewer.zoom()).toBe(1);
    lead.viewer.zoomIn();
    TestBed.tick();
    expect(follower.viewer.zoom()).toBe(1);
    follower.viewer.zoomIn();
    expect(lead.viewer.zoom()).toBeCloseTo(1.15, 9);
    expect(follower.viewer.zoomLabel()).toBe('1.1×');
  });

  it('leaves the reference on its own when the candidate has no landmarks', () => {
    open(lead, 1, null);
    open(follower, 2, CORNERED);
    expect(follower.viewer.linked()).toBe(false);
    expect(follower.viewer.zoom()).toBe(1);
    lead.viewer.zoomIn();
    TestBed.tick();
    expect(follower.viewer.zoom()).toBe(1);
  });

  it('is off while either is framed on a box (Wider), or shows the original payload', () => {
    openBoth();
    follower.fixture.componentRef.setInput('fitBox', [0.1, 0.1, 0.5, 0.5]);
    TestBed.tick();
    expect(follower.viewer.linked()).toBe(false);
    follower.fixture.componentRef.setInput('fitBox', null);
    TestBed.tick();
    expect(follower.viewer.linked()).toBe(true);

    lead.fixture.componentRef.setInput('variant', 'original');
    TestBed.tick();
    expect(follower.viewer.linked()).toBe(false);
  });

  it('holds still through the candidate\'s next image, then lines the next face up', () => {
    openBoth();
    lead.viewer.onZoomInput({ target: { value: '3' } } as unknown as Event);
    TestBed.tick();
    const zoomed = follower.viewer.zoom();

    show(lead, 3, CENTRED);
    // The next candidate is loading: the reference neither jumps to its own
    // whole image nor follows a lead with nothing on screen.
    expect(follower.viewer.zoom()).toBe(zoomed);
    expect(follower.viewer.zoomLabel()).toBe('1×');

    loaded(lead);
    expect(follower.viewer.zoom()).toBeCloseTo(2, 9);
    expectMatched();
  });

  it('goes back to its own whole image when the next candidate has no landmarks', () => {
    openBoth();
    lead.viewer.zoomIn();
    TestBed.tick();
    open(lead, 3, null);
    expect(follower.viewer.linked()).toBe(false);
    expect(follower.viewer.zoom()).toBe(1);
    expectClose([follower.viewer.panX(), follower.viewer.panY()], [0, 0]);
  });

  it('keeps the view when the reference steps to its next face', () => {
    openBoth();
    lead.viewer.onZoomInput({ target: { value: '2' } } as unknown as Event);
    lead.viewer.onWheel({ deltaY: 1, clientX: 30, clientY: 70, preventDefault: () => {} } as unknown as WheelEvent);
    const leadZoom = lead.viewer.zoom();

    show(follower, 4, LOWER);
    loaded(follower, [400, 400]);
    follower.natural = [400, 400];
    expect(lead.viewer.zoom()).toBe(leadZoom);
    expect(follower.viewer.shownZoom()).toBe(leadZoom);
    expectMatched(LOWER);
    expectClose(faceUnder(follower, LOWER), faceUnder(lead, CENTRED));
  });

  it('lets the candidate go its own way once the link is gone', () => {
    openBoth();
    follower.fixture.destroy();
    lead.viewer.zoomIn();
    expect(lead.viewer.zoom()).toBeCloseTo(1.15, 9);
    expect(link.lead()).toBe(lead.viewer);
    lead.fixture.componentRef.setInput('link', null);
    TestBed.tick();
    expect(link.lead()).toBeNull();
  });
});
