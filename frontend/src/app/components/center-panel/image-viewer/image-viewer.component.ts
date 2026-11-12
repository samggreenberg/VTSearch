import { ChangeDetectionStrategy, Component, computed, effect, ElementRef, inject, input, OnDestroy, output, Signal, signal, untracked, viewChild } from '@angular/core';
import { NgStyle } from '@angular/common';
import { Media, PayloadVariant } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';
import {
  FaceFrame,
  FaceLandmarks,
  FaceView,
  LEAD_OFF,
  LEAD_PENDING,
  LeadState,
  LinkLead,
  PaneView,
  ViewLink,
  faceAnchor,
  faceFrame,
  faceViewOf,
  paneViewOf,
} from './face-link';

export type RegionBox = readonly [number, number, number, number];
export type ResizeHandle = 'n' | 's' | 'e' | 'w' | 'nw' | 'ne' | 'sw' | 'se';

/**
 * Whose face an {@link OutlineBox} outlines, on a photo several people share:
 * `primary` is the person in focus, `secondary` the person they are seen
 * with, and `other` anyone else in the photo, drawn faintly.
 */
export type OutlineRole = 'primary' | 'secondary' | 'other';

/** A normalised `[x0, y0, x1, y1]` box on the image, and whose it is. */
export interface OutlineBox {
  readonly box: RegionBox;
  readonly role: OutlineRole;
}

/**
 * Each role's hover title, so colour is never the only cue; and the drawing
 * order, faint ones first, so the person in focus is on top where faces
 * overlap.
 */
const OUTLINE_ROLES: Readonly<Record<OutlineRole, { title: string; layer: number }>> = {
  other: { title: 'Someone else in this photo', layer: 0 },
  secondary: { title: 'Appears with the person in focus', layer: 1 },
  primary: { title: 'The person in focus', layer: 2 },
};

/** An {@link OutlineBox} the viewer can draw: its role, title and style. */
export interface DrawnOutlineBox {
  readonly role: OutlineRole;
  readonly title: string;
  readonly style: { [k: string]: string };
}

/**
 * Where a viewer that follows a {@link ViewLink} stands: not following (its
 * own zoom and pan, as any viewer), holding still while it or its lead lays out
 * a new image, or drawn from the lead's view. `unit` is its zoom for each of the
 * lead's, the two faces' sizes on screen being matched.
 */
type Following =
  | { readonly kind: 'off' }
  | { readonly kind: 'hold'; readonly lead: LinkLead }
  | { readonly kind: 'on'; readonly lead: LinkLead; readonly view: FaceView; readonly frame: FaceFrame; readonly unit: number };

const NOT_FOLLOWING: Following = { kind: 'off' };

/** How far a zoom may fall short of the cap and still count as at it: a
 *  following viewer's zoom is the lead's times a ratio, so it lands a hair off. */
const AT_CAP = 1e-9;

type DragMode =
  | { kind: 'pan'; startX: number; startY: number; originX: number; originY: number }
  | { kind: 'draw'; anchor: { x: number; y: number }; previousBox: RegionBox | null }
  | { kind: 'move'; startLocal: { x: number; y: number }; startBox: RegionBox }
  | { kind: 'resize'; handle: ResizeHandle; startBox: RegionBox };

const MIN_BOX_SIZE = 0.01; // 1% of the image; below this we treat a draw as a stray click

/** How far past the opening view the user can zoom in (see `maxZoom`). */
const MAX_ZOOM = 5;

// What a region draw started from outside the canvas does NOT claim (see
// `tryStartOffCanvasDraw`). Two different reasons:
//
//   - Interactive controls keep their own click semantics. Without this, turning
//     the sticky Marquee toggle on would make the Good/Bad buttons and the zoom
//     slider unusable, because every mousedown on them would anchor a box.
//   - `.metadata-tray` is excluded wholesale, controls and prose alike. It sits at
//     the very bottom of the panel, far enough from the image that a drag begun
//     there reads as "select this text", not as "box that corner"; and selecting a
//     filename or an MD5 by dragging across it is a thing people actually do.
const OFF_CANVAS_DRAW_EXCLUDED =
  'button, input, select, textarea, a, label, [role="button"], [contenteditable], .metadata-tray';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-image-viewer',
  standalone: true,
  imports: [NgStyle],
  templateUrl: './image-viewer.component.html',
  styleUrl: './image-viewer.component.scss',
})
export class ImageViewerComponent implements OnDestroy, LinkLead {
  private activeContext = inject(ActiveContextService);

  readonly media = input.required<Media>();
  /**
   * Which payload to show: `''` (canonical) or `'original'` (the pre-clean
   * snapshot of an item a cleaner rewrote at load time).  The parent's
   * Clean/Original toggle drives it; it only ever leaves `''` for media whose
   * `has_original` flag is set.
   */
  readonly variant = input<PayloadVariant>('');
  /**
   * True while the parent is in the v2 "bad-vote-with-box discard confirm" armed state.
   * The viewer uses it to (a) render the box with a sticky red pulse, and (b) route Esc /
   * mouse-on-box back to the parent via `armedConfirmCanceled` instead of clearing the box.
   */
  readonly pendingBadConfirm = input(false);
  /**
   * The region the detector/embedder matched best at inference, as a normalised
   * ``[x0, y0, x1, y1]`` box (the argmax over patch regions that produced this
   * media's score).  ``null`` when the active dataset isn't patch-region-aware
   * or the focused media hasn't been scored.  Rendered as a neutral white/black
   * dashed overlay (distinct from the user's solid yellow voting box) only while
   * the Highlight toggle (``highlightMode``) is on; purely informational, never
   * interactive (no drag/resize, no vote semantics).
   */
  readonly highlightBox = input<RegionBox | null>(null);
  /**
   * A normalised ``[x0, y0, x1, y1]`` box drawn whatever the Highlight toggle
   * says, in the Highlight overlay's style: where a face crop sat in the photo
   * Show in photo opened (#4750), which has to show on a dataset with no patch
   * embedder, where there is no Highlight toggle to turn on.
   */
  readonly outlineBox = input<RegionBox | null>(null);
  /**
   * Several faces outlined on one photo, each coloured by whose it is (see
   * {@link OutlineRole}): the People view's shared photo, where two people
   * are outlined and everyone else in it is marked faintly. Drawn beside
   * {@link outlineBox}, not instead of it; a box `overlayBoxStyle` rejects is
   * skipped.
   */
  readonly outlineBoxes = input<readonly OutlineBox[] | null>(null);
  /**
   * Show the item out of this dataset rather than the active one (`''`, the
   * default, is the active one). Show in photo's photo lives in the Image
   * dataset of the face's import, while the Face pair stays active; see
   * `ActiveContextService.mediaUrlIn`.
   */
  readonly datasetId = input('');
  /**
   * Show the picture at this URL instead of {@link media}'s own (`''`, the
   * default, is the item's). A detector's example is a file of the user's,
   * not an item of any dataset, and the Face compare layout shows one as the
   * reference (#5642); `media` then only names it.
   */
  readonly srcOverride = input('');
  /**
   * A normalised `[x0, y0, x1, y1]` box to open the image on, zoomed and
   * panned so the box fills the pane, instead of the whole image fitted to it
   * (`null`, the default). The Face compare layout's Wider view (#5645) shows a
   * face in its photo this way. The box may run past the image's edges; the
   * pan stops at them as it always does. It is the view's home: Reset view, and
   * a double-click at the zoom cap, come back to it. The cap grows with it
   * ({@link maxZoom}), so a small face in a big photo can still be looked at
   * as closely as its crop.
   */
  readonly fitBox = input<RegionBox | null>(null);
  /** {@link fitBox} by value: a parent recomputing the same box (a metadata
   *  batch landing re-reads its item) must not re-frame under the user's zoom. */
  private readonly fitTarget = computed(() => this.fitBox(), { equal: sameBox });
  /**
   * Whether a Shift- or Marquee-drag draws a region box. Off where a box would
   * mean nothing, so a drag only pans: the Show in photo overlay, where there
   * is nothing to vote on, and a face, whose embedder reads the whole crop.
   */
  readonly regionDraw = input(true);
  /**
   * The face's five landmarks (`face_landmarks`), which place it for a
   * {@link link}. Ignored while the viewer is framed on a {@link fitBox} or
   * shows the `original` payload, which the points were not measured on.
   */
  readonly landmarks = input<FaceLandmarks | null | undefined>(null);
  /**
   * Tie this viewer's zoom and pan to another's, with the two faces matched in
   * size and position (the Face compare layout, #5647; see `face-link.ts`).
   * The one marked `lead` keeps its own view; the one marked `follow` is drawn
   * from it, and hands its gestures to it. They are linked while both show a
   * face with {@link landmarks}; otherwise each is on its own, as with no link.
   */
  readonly link = input<ViewLink | null>(null);
  readonly linkAs = input<'lead' | 'follow'>('lead');
  readonly regionBoxChange = output<RegionBox | null>();
  /**
   * Fired when the user does something that cancels the armed bad-vote-confirm without
   * voting (Esc while armed, or any mousedown on the box body/handles, or starting a
   * fresh Shift-drag). The parent clears its armed state but keeps the box.
   */
  readonly armedConfirmCanceled = output<void>();

  // Public: the coord-transform specs substitute a stub wrap element here.
  readonly wrapRef = viewChild<ElementRef<HTMLDivElement>>('imageWrap');
  private readonly imageRef = viewChild<ElementRef<HTMLImageElement>>('imageEl');

  // Signals: written from the media-change effect below and read in the
  // template, so plain fields would leave the view stale under zoneless.
  readonly imageSrc = signal('');
  readonly imageReady = signal(false);
  // zoom/rotation are signals: they are written from the keyboard-shortcut
  // dispatch (an un-bound RxJS callback in CenterPanelComponent) and from the
  // parent's toolbar buttons (click handlers bound in the PARENT's template,
  // which mark only the parent dirty) — with a plain field, this OnPush view's
  // transform binding silently went stale until some unrelated CD repainted it.
  readonly zoom = signal(1);
  readonly rotation = signal(0);
  /** The zoom the toolbar shows and its slider moves: {@link zoom}, except
   *  that a following viewer shows its lead's, the faces being the same size,
   *  and goes on showing it while either lays out its next image. */
  readonly shownZoom: Signal<number> = computed(() => {
    const following = this.following();
    return following.kind === 'off' ? this.zoom() : following.lead.shownZoom();
  });
  readonly zoomLabel = computed(() => {
    const z = this.shownZoom();
    return (z === Math.floor(z) ? z.toFixed(0) : z.toFixed(1)) + '×';
  });
  // Track the id of the media we last reset for. The `media` input reference
  // changes whenever `MediaMetadataCacheService` hydrates richer metadata for
  // the same id; re-running the media-change effect for those enrichments would clobber
  // `imageReady=true` back to false and, since `imageSrc` is the same string,
  // Angular wouldn't re-fire the `<img>` load event, leaving the canvas
  // permanently hidden behind `visibility: hidden`.
  private lastMediaId: number | null = null;
  // Same guard for the payload variant: flipping Clean/Original keeps the media
  // id, so the effect has to notice the variant changed to refetch the image.
  private lastVariant: PayloadVariant = '';
  // And for the dataset: media ids are per dataset, so the same id out of
  // another one is another image.
  private lastDatasetId = '';
  // And for an overriding source, which is the whole identity of what it shows.
  private lastSrcOverride = '';
  /** A {@link fitBox} framing is owed: set for a new image or a new box, paid
   *  once the image is laid out, so later resizes leave the user's zoom alone. */
  private fitOwed = false;
  /** Whether the image on screen has loaded, is loading, or failed: what a
   *  {@link link} waits for before it lines a face up. */
  private readonly loadState = signal<'loading' | 'loaded' | 'failed'>('loading');
  /** Last seen following a lead, so leaving the link resets the view it left. */
  private wasFollowing = false;

  /** Where the face is on its image, while there is one to line up. */
  private readonly anchor = computed(() =>
    this.variant() === '' && !this.fitTarget() ? faceAnchor(this.landmarks()) : null,
  );
  /** The face as laid out in this pane, once its image has loaded. */
  private readonly frame = computed<FaceFrame | null>(() => {
    const anchor = this.anchor();
    if (!anchor || this.loadState() !== 'loaded') return null;
    return faceFrame(anchor, this.renderedW(), this.renderedH());
  });

  /** As a lead: the face view this viewer shows, for the one that follows it. */
  readonly leadState = computed<LeadState>(() => {
    if (!this.anchor() || this.loadState() === 'failed') return LEAD_OFF;
    const frame = this.frame();
    if (!frame) return LEAD_PENDING;
    return { kind: 'on', frame, view: faceViewOf(frame, this.paneView(), this.rotation()) };
  });

  /** As a follower: what it follows (see {@link Following}). */
  private readonly following = computed<Following>(() => {
    const link = this.link();
    if (!link || this.linkAs() !== 'follow') return NOT_FOLLOWING;
    const lead = link.lead();
    const state = lead?.leadState();
    if (!lead || !state || state.kind === 'off' || !this.anchor() || this.loadState() === 'failed') {
      return NOT_FOLLOWING;
    }
    const frame = this.frame();
    if (state.kind === 'pending' || !frame) return { kind: 'hold', lead };
    return { kind: 'on', lead, view: state.view, frame, unit: state.frame.unit / frame.unit };
  });

  /** This viewer follows a lead's view right now. */
  readonly linked = computed(() => this.following().kind === 'on');

  /** This viewer's zoom for each of its lead's (1 when it leads or is alone). */
  private readonly zoomUnit = computed(() => {
    const following = this.following();
    return following.kind === 'on' ? following.unit : 1;
  });

  // Region voting state (UI only; see docs/ARCHITECTURE.md → "Embedder binding and routing").
  // These are signals because they are written from un-patched callbacks (window
  // mouse/key listeners, the shake `setTimeout`) that schedule no change detection
  // under zoneless; a signal write read in the template is on the notification path.
  readonly regionBox = signal<RegionBox | null>(null);
  readonly regionBoxShake = signal(false);
  readonly shiftHeld = signal(false);
  // Sticky toggle exposed by the Marquee button in .image-view-controls. While true the
  // viewer behaves as if Shift were held: cursor is a crosshair and a left-drag draws a
  // new region instead of panning. Shift+drag remains a power-user shortcut even when
  // the toggle is off; toggling the button just turns the gesture on without a modifier.
  // Signal for the same parent-toolbar reason as `zoom`: toggled from the
  // parent's template, read here (crosshair cursor / region-mode class).
  readonly marqueeMode = signal(false);
  // Sticky toggle exposed by the Highlight button in .image-view-controls. While
  // true the viewer overlays a neutral white/black dashed box (`highlightBox`)
  // around the region the detector matched best at inference. Independent of marqueeMode -
  // both can be on at once (the highlight is read-only and sits behind the
  // interactive voting box).
  readonly highlightMode = signal(false);
  // renderedW/H are written from a raw ResizeObserver callback (un-patched, no CD
  // under zoneless), so they are signals; reading them in the template keeps the
  // overlay geometry fresh when the wrap resizes.
  readonly renderedW = signal(0);
  readonly renderedH = signal(0);

  // panX/panY are signals (written from the window-level mousemove drag handler,
  // an un-patched callback). They are not `private` so tests can drive
  // screenToImageNormalized() with non-zero pan without simulating a full wheel +
  // drag sequence.
  readonly panX = signal(0);
  readonly panY = signal(0);
  private drag: DragMode | null = null;

  private mouseMoveHandler: ((e: MouseEvent) => void) | null = null;
  private mouseUpHandler: (() => void) | null = null;
  private keyDownHandler: ((e: KeyboardEvent) => void) | null = null;
  private keyUpHandler: ((e: KeyboardEvent) => void) | null = null;
  private blurHandler: (() => void) | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private shakeTimer: ReturnType<typeof setTimeout> | null = null;

  readonly minZoom = 1;
  /** The zoom the {@link fitBox} framing landed on, or 1 with no box. */
  private readonly fitZoom = signal(1);
  /** As far as the user can zoom in: five times the view the image opened
   *  on, which is the whole image unless {@link fitBox} framed a part of it. */
  get maxZoom(): number {
    return MAX_ZOOM * Math.max(1, this.fitZoom());
  }
  readonly zoomStep = 0.05;
  /** How hard a double-click zooms in about the cursor. Matches the Browse
   *  canvas's own `DOUBLE_CLICK_ZOOM`, and is deliberately larger than a wheel
   *  notch so the gesture lands a decisive jump rather than a nudge. */
  readonly doubleClickZoom = 2;

  constructor() {
    this.setupWindowKeyListeners();

    // A lead puts itself on its link, for as long as both are its.
    effect((onCleanup) => {
      const link = this.link();
      if (!link || this.linkAs() !== 'lead') return;
      link.attach(this);
      onCleanup(() => link.detach(this));
    });

    // A follower draws itself from its lead's view whenever that moves, or its
    // own face is laid out anew or turned. Leaving the link (the lead's next
    // face has no landmarks, say) resets it, since the view it was left on
    // was the lead's and need not even fit its image.
    effect(() => {
      const following = this.following();
      const rotation = this.rotation();
      untracked(() => {
        if (following.kind === 'hold') return;
        if (following.kind === 'off') {
          if (this.wasFollowing) {
            this.wasFollowing = false;
            this.resetView();
          }
          return;
        }
        this.wasFollowing = true;
        this.setPaneView(paneViewOf(following.frame, following.view, rotation));
      });
    });

    // Reset the viewport for a new image when the media id changes. The id
    // guard skips same-id metadata-enrichment reference changes (see
    // lastMediaId above); zoom/rotation/marquee state intentionally persists.
    effect(() => {
      const media = this.media();
      const variant = this.variant();
      const datasetId = this.datasetId();
      const srcOverride = this.srcOverride();
      const sameSource = datasetId === this.lastDatasetId && srcOverride === this.lastSrcOverride;
      if (media.id === this.lastMediaId && variant === this.lastVariant && sameSource) return;
      const sameMedia = media.id === this.lastMediaId && sameSource;
      this.lastMediaId = media.id;
      this.lastVariant = variant;
      this.lastDatasetId = datasetId;
      this.lastSrcOverride = srcOverride;
      this.imageReady.set(false);
      this.loadState.set('loading');
      const path = `/api/medias/${media.id}/image`;
      const src =
        srcOverride ||
        (datasetId
          ? this.activeContext.mediaUrlIn(datasetId, path, { variant })
          : this.activeContext.mediaUrl(path, { variant }));
      // A review flow may already have warmed this URL into the HTTP cache
      // (`MediaPrefetchService`, #3896); the `<img>` is then answered from
      // there, or joins the warm's download if it is still in flight.
      this.imageSrc.set(src);
      // A variant flip is the same item shown differently: keep the user's
      // zoom / pan and their voting box instead of resetting as for a new item.
      if (sameMedia) return;
      untracked(() => {
        this.resetView();
        this.clearRegionBox({ emit: true });
        // Framed once the new image is laid out (`onImageLoad`).
        this.fitOwed = this.fitTarget() !== null;
      });
    });

    // A new box on the image already on screen (the next face out of the same
    // photo) is framed now; on an image still loading, once it has. No box is
    // the plain view, and the zoom cap goes back to its own.
    effect(() => {
      const box = this.fitTarget();
      untracked(() => {
        this.fitOwed = box !== null;
        if (box === null) {
          this.fitZoom.set(1);
          this.zoom.update((z) => this.clampZoom(z));
          this.applyTransform();
          return;
        }
        if (this.imageReady()) this.payFit();
      });
    });
  }

  onImageLoad(): void {
    this.imageReady.set(true);
    this.recomputeRenderedSize();
    this.attachWrapResizeObserver();
    this.loadState.set('loaded');
  }

  onImageError(): void {
    this.imageReady.set(true);
    this.loadState.set('failed');
  }

  ngOnDestroy(): void {
    this.removeWindowMouseListeners();
    this.removeWindowKeyListeners();
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    if (this.shakeTimer) clearTimeout(this.shakeTimer);
  }

  onZoomInput(event: Event): void {
    // The slider moves the zoom it shows (`shownZoom`); `zoomAboutCentre` brings
    // a follower up to its lead first, so the unit is read after.
    this.followLead();
    this.zoomAboutCentre(parseFloat((event.target as HTMLInputElement).value) * this.zoomUnit());
  }

  // The toolbar's and the keys' rotate and zoom keep whatever is in the middle
  // of the pane there, which matters once the view is panned: a face framed by
  // `fitBox` stays in view through a turn or a zoom rather than swinging off
  // round the image's own centre. With no pan, the only view there was before
  // `fitBox`, it is the same transform as turning or zooming the image alone.

  rotateLeft(): void {
    this.rotateAboutCentre(-90);
  }

  rotateRight(): void {
    this.rotateAboutCentre(90);
  }

  zoomIn(): void {
    this.zoomAboutCentre(this.zoom() * 1.15);
  }

  zoomOut(): void {
    this.zoomAboutCentre(this.zoom() * 0.85);
  }

  /** Turn by a quarter (*deg* is ±90), turning the pan with the image. */
  private rotateAboutCentre(deg: 90 | -90): void {
    this.followLead();
    this.rotation.update((r) => r + deg);
    // Clockwise on screen, y pointing down: (x, y) -> (-y, x).
    const [x, y] = [this.panX(), this.panY()];
    this.panX.set(deg === 90 ? -y : y);
    this.panY.set(deg === 90 ? x : -x);
    this.applyTransform();
  }

  /** Zoom to *newZoom* (clamped), scaling the pan with it. */
  private zoomAboutCentre(newZoom: number): void {
    this.followLead();
    const oldZoom = this.zoom();
    this.zoom.set(this.clampZoom(newZoom));
    const ratio = this.zoom() / oldZoom;
    this.panX.update((p) => p * ratio);
    this.panY.update((p) => p * ratio);
    this.applyTransform();
  }

  /** Back to the view the image opened on: upright, and the whole image, or
   *  framed on {@link fitBox} when there is one and the image is on screen.
   *  A follower turns upright and takes both panes back to the lead's whole
   *  image, which keeps its own turn. */
  resetView(): void {
    const following = this.following();
    if (following.kind === 'on') {
      this.rotation.set(0);
      following.lead.homeView();
      this.followLead();
      return;
    }
    this.zoom.set(1);
    this.rotation.set(0);
    this.panX.set(0);
    this.panY.set(0);
    if (this.imageReady() && this.fitTarget()) this.fitToBox();
    this.applyTransform();
  }

  /** Frame {@link fitBox} if a framing is owed and the image can be measured. */
  private payFit(): void {
    if (this.fitOwed && this.fitToBox()) this.fitOwed = false;
  }

  /**
   * Zoom and pan so {@link fitBox} fills the pane, its centre on the pane's
   * where the image's edges allow. Keeps the rotation, which only ever turns
   * a quarter at a time, so a box on its side swaps width for height. Returns
   * false, changing nothing, when there is no box or nothing is laid out yet.
   */
  private fitToBox(): boolean {
    const box = this.fitTarget();
    const wrap = this.wrapRef()?.nativeElement;
    const renderedW = this.renderedW();
    const renderedH = this.renderedH();
    if (!box || !wrap || !renderedW || !renderedH) return false;
    const [x0, y0, x1, y1] = box;
    const boxW = (x1 - x0) * renderedW;
    const boxH = (y1 - y0) * renderedH;
    if (!(boxW > 0 && boxH > 0) || !wrap.clientWidth || !wrap.clientHeight) return false;
    const rot = ((this.rotation() % 360) + 360) % 360;
    const swapped = rot === 90 || rot === 270;
    const onScreenW = swapped ? boxH : boxW;
    const onScreenH = swapped ? boxW : boxH;
    const fit = Math.min(wrap.clientWidth / onScreenW, wrap.clientHeight / onScreenH);
    this.fitZoom.set(fit);
    const zoom = this.clampZoom(fit);
    this.zoom.set(zoom);
    // The box centre's offset from the image centre, turned and scaled as the
    // transform turns and scales the image; panning by its opposite puts it
    // in the middle of the pane.
    const dx = ((x0 + x1) / 2 - 0.5) * renderedW;
    const dy = ((y0 + y1) / 2 - 0.5) * renderedH;
    const rad = (rot * Math.PI) / 180;
    const cos = Math.round(Math.cos(rad));
    const sin = Math.round(Math.sin(rad));
    this.panX.set(-zoom * (dx * cos - dy * sin));
    this.panY.set(-zoom * (dx * sin + dy * cos));
    this.applyTransform();
    return true;
  }

  onWheel(event: WheelEvent): void {
    event.preventDefault();
    this.followLead();
    const oldZoom = this.zoom();
    const delta = event.deltaY > 0 ? -0.15 : 0.15;
    this.zoomAbout(oldZoom + delta * oldZoom, event.clientX, event.clientY);
  }

  /** Double-click the image to look closer.
   *
   *  The gesture people reach for when a voting decision needs more detail, and
   *  the one the Browse canvas already answers, so it means the same thing in
   *  both places: a decisive jump in (`DOUBLE_CLICK_ZOOM`, larger than a wheel
   *  notch) anchored on the point under the cursor, suppressed while a region
   *  draw owns the gesture.
   *
   *  The one thing it does that the map's version doesn't is close the loop:
   *  the image viewer caps at `maxZoom`, so a double-click there would
   *  otherwise be a dead gesture, leaving the toolbar or `-` as the only way
   *  back from a zoom the user got into with the mouse alone. At the cap the
   *  same gesture returns to fit instead. Rotation is deliberately kept - the
   *  user rotated on purpose, and this is a zoom control, not `resetView()`. */
  onDoubleClick(event: MouseEvent): void {
    if (event.button !== 0) return;
    // Region-draw owns the gesture while it is active (Shift held or the sticky
    // Marquee toggle on): two quick draws must not also zoom. Same suppression
    // the Browse canvas applies to its own double-click zoom.
    if (this.regionDrawActive) return;
    // Without this a double-click paints a native selection over the canvas.
    event.preventDefault();
    this.followLead();
    if (this.shownZoom() >= this.maxZoom - AT_CAP) {
      const following = this.following();
      if (following.kind === 'on') {
        following.lead.homeView();
        this.followLead();
        return;
      }
      this.zoom.set(this.minZoom);
      this.panX.set(0);
      this.panY.set(0);
      // Framed on its box, the view the image opened on is that framing.
      if (this.fitTarget()) this.fitToBox();
      this.applyTransform();
      return;
    }
    this.zoomAbout(this.zoom() * this.doubleClickZoom, event.clientX, event.clientY);
  }

  /** Zoom to `newZoom` (clamped) keeping the image point currently under
   *  (`clientX`, `clientY`) pinned there, by counter-panning about that point.
   *  Shared by the wheel and the double-click so both anchor identically. */
  private zoomAbout(newZoom: number, clientX: number, clientY: number): void {
    const oldZoom = this.zoom();
    this.zoom.set(this.clampZoom(newZoom));

    const wrap = this.wrapRef()?.nativeElement;
    if (wrap) {
      const rect = wrap.getBoundingClientRect();
      const s = this.layoutScale(wrap, rect);
      const cx = (clientX - rect.left - rect.width / 2) / s;
      const cy = (clientY - rect.top - rect.height / 2) / s;
      const ratio = this.zoom() / oldZoom;
      this.panX.set(cx - ratio * (cx - this.panX()));
      this.panY.set(cy - ratio * (cy - this.panY()));
    }
    this.applyTransform();
  }

  /** True when a drag should draw a region (either Shift-held or Marquee toggle on). */
  get regionDrawActive(): boolean {
    return this.regionDraw() && (this.shiftHeld() || this.marqueeMode());
  }

  toggleMarqueeMode(): void {
    this.marqueeMode.update((v) => !v);
  }

  toggleHighlightMode(): void {
    this.highlightMode.update((v) => !v);
  }

  /** True when the best-match highlight box should be drawn over the image. */
  get highlightVisible(): boolean {
    return this.highlightMode() && this.highlightBoxStyle !== null;
  }

  /** Percent-position style for the best-match highlight overlay.  Returns null
   *  when the box is missing, malformed, degenerate, or covers (effectively) the
   *  whole image, so neither the near-full single-vector fallback box nor a
   *  patch dataset's winning image-level row paints a frame round everything.  A
   *  patch dataset's other outcome - one grid cell - is small but deliberate:
   *  it points at the exact patch the detector scored highest.  (This overlay is
   *  the only place a best-match region is drawn - thumbnails never render a
   *  best-region outline.) */
  get highlightBoxStyle(): { [k: string]: string } | null {
    return overlayBoxStyle(this.highlightBox(), { dropWholeImage: true });
  }

  /** Percent-position style for {@link outlineBox}, or null when there is none
   *  to draw. Unlike the highlight, a box round (nearly) the whole image still
   *  draws: a face that fills its photo is still the face the user came from. */
  get outlineBoxStyle(): { [k: string]: string } | null {
    return overlayBoxStyle(this.outlineBox(), { dropWholeImage: false });
  }

  /** {@link outlineBoxes} as drawn: malformed boxes and unknown roles left
   *  out, the rest in drawing order (the person in focus last, so on top).
   *  Like {@link outlineBoxStyle}, a box round the whole image still draws. */
  readonly drawnOutlineBoxes = computed<readonly DrawnOutlineBox[]>(() => {
    const drawn: DrawnOutlineBox[] = [];
    for (const entry of this.outlineBoxes() ?? []) {
      const role = Object.hasOwn(OUTLINE_ROLES, entry.role) ? OUTLINE_ROLES[entry.role] : undefined;
      const style = role ? overlayBoxStyle(entry.box, { dropWholeImage: false }) : null;
      if (role && style) drawn.push({ role: entry.role, title: role.title, style });
    }
    // Array.prototype.sort is stable, so boxes in one role keep their order.
    return drawn.sort((a, b) => OUTLINE_ROLES[a.role].layer - OUTLINE_ROLES[b.role].layer);
  });

  onMouseDown(event: MouseEvent): void {
    if (event.button !== 0) return;

    // Drag from anywhere on the canvas (Shift-held or marquee mode) starts a fresh box.
    if (this.beginDraw(event)) return;

    // Default: pan-when-zoomed.
    if (!this.panRoom()) return;
    this.followLead();
    this.drag = {
      kind: 'pan',
      startX: event.clientX,
      startY: event.clientY,
      originX: this.panX(),
      originY: this.panY(),
    };
    event.preventDefault();
    this.setupWindowMouseListeners();
  }

  /** Start a region draw from a point OUTSIDE the image canvas.
   *
   *  The horizontal case has always worked by accident of layout: the image is
   *  ``object-fit: contain`` inside a full-width ``.image-wrap``, so for a tall
   *  image the letterbox columns beside it are still *inside* the wrap and hit
   *  `onMouseDown` above; `screenToImageNormalized` returns ``x < 0`` and
   *  `clamp01` snaps the anchor to the nearest edge. Vertically there is no such
   *  slack - the wrap is exactly as tall as the media area, and below it sit the
   *  toolbar, the vote buttons and the metadata tray as SIBLINGS under
   *  ``.center-panel``, so a mousedown there never reached this component at all
   *  (it just started a native text selection over the buttons).
   *
   *  This is the same gesture, delegated from the panel. Nothing downstream needs
   *  to change: the anchor is clamped exactly as the sideways case is, and the
   *  move/up listeners were already on ``window``, so dragging out of the canvas
   *  has always worked once a drag was under way. Only the *start* was gated.
   *
   *  Interactive controls are deliberately excluded - a click on the Good/Bad
   *  buttons, the zoom slider or the metadata toggle keeps doing exactly what it
   *  does today, including while the sticky Marquee toggle is on. */
  tryStartOffCanvasDraw(event: MouseEvent): void {
    if (event.button !== 0 || !this.regionDrawActive) return;
    const target = event.target as HTMLElement | null;
    const wrap = this.wrapRef()?.nativeElement;
    // Inside the canvas the wrap's own handler owns the gesture; this delegated
    // one sees the bubbled event afterwards and must not start a second draw.
    if (!wrap || !target || wrap.contains(target)) return;
    if (target.closest?.(OFF_CANVAS_DRAW_EXCLUDED)) return;
    this.beginDraw(event);
  }

  /** Anchor a fresh draw at `event`, clamped to the image. Returns false (leaving
   *  the caller to fall through to its own default) when region-draw isn't active
   *  or the image isn't laid out yet. */
  private beginDraw(event: MouseEvent): boolean {
    if (!this.regionDrawActive || this.renderedW() <= 0 || this.renderedH() <= 0) return false;
    const local = this.screenToImageNormalized(event);
    if (!local) return false;
    const x = clamp01(local.x);
    const y = clamp01(local.y);
    if (this.pendingBadConfirm()) this.armedConfirmCanceled.emit();
    // Remember the prior box so we can restore it on a zero-area release;
    // a stray Shift-click on empty space must not throw away real work.
    this.drag = { kind: 'draw', anchor: { x, y }, previousBox: this.regionBox() };
    this.regionBox.set([x, y, x, y]);
    // Also what stops the drag from painting a native text selection across the
    // buttons and labels it passes over.
    event.preventDefault();
    this.setupWindowMouseListeners();
    return true;
  }

  onRegionBodyMouseDown(event: MouseEvent): void {
    const box = this.regionBox();
    if (event.button !== 0 || !box) return;
    event.stopPropagation();
    event.preventDefault();
    const local = this.screenToImageNormalized(event);
    if (!local) return;
    if (this.pendingBadConfirm()) this.armedConfirmCanceled.emit();
    this.drag = { kind: 'move', startLocal: local, startBox: box };
    this.setupWindowMouseListeners();
  }

  onResizeHandleMouseDown(handle: ResizeHandle, event: MouseEvent): void {
    const box = this.regionBox();
    if (event.button !== 0 || !box) return;
    event.stopPropagation();
    event.preventDefault();
    if (this.pendingBadConfirm()) this.armedConfirmCanceled.emit();
    this.drag = { kind: 'resize', handle, startBox: box };
    this.setupWindowMouseListeners();
  }

  /** Clear the current region box and notify the parent. */
  clearRegionBox(opts: { emit: boolean } = { emit: true }): void {
    if (this.regionBox() === null) return;
    this.regionBox.set(null);
    if (opts.emit) this.regionBoxChange.emit(null);
  }

  /** Visually flash the region box (used by bad-vote-confirm flow). */
  pulseRegionBox(): void {
    if (!this.regionBox()) return;
    this.regionBoxShake.set(true);
    if (this.shakeTimer) clearTimeout(this.shakeTimer);
    this.shakeTimer = setTimeout(() => this.regionBoxShake.set(false), 500);
  }

  get imageTransform(): string {
    return `translate(${this.panX()}px, ${this.panY()}px) scale(${this.zoom()}) rotate(${this.rotation()}deg)`;
  }

  get wrapCursor(): string {
    if (this.regionDrawActive) return 'crosshair';
    return this.panRoom() ? 'grab' : '';
  }

  // --- As the lead of a link (`LinkLead`) ---

  /** Show *view* of the face, zoom and pan kept as any gesture's are. */
  showFaceView(view: FaceView): void {
    const frame = this.frame();
    if (!frame) return;
    const zoom = this.clampZoom(view.scale / frame.unit);
    this.setPaneView(paneViewOf(frame, { ...view, scale: zoom * frame.unit }, this.rotation()));
    this.applyTransform();
  }

  /** The whole image, keeping the turn. */
  homeView(): void {
    this.zoom.set(1);
    this.panX.set(0);
    this.panY.set(0);
    if (this.imageReady() && this.fitTarget()) this.fitToBox();
    this.applyTransform();
  }

  canPan(): boolean {
    const max = this.getMaxPan();
    return max.x > 0 || max.y > 0;
  }

  /** Whether a drag pans: this image is zoomed past the pane, or, following, the lead's is. */
  private panRoom(): boolean {
    const following = this.following();
    return following.kind === 'on' ? following.lead.canPan() : this.canPan();
  }

  private paneView(): PaneView {
    return { zoom: this.zoom(), panX: this.panX(), panY: this.panY() };
  }

  private setPaneView(pane: PaneView): void {
    this.zoom.set(pane.zoom);
    this.panX.set(pane.panX);
    this.panY.set(pane.panY);
  }

  /**
   * As a follower: show the lead's view as it now stands. The effect that keeps
   * a follower drawn from its lead runs on the next change detection, so each
   * gesture calls this first: one begun on a view the lead has since left would
   * hand the old view back and undo the lead's move. A no-op on any other viewer.
   */
  private followLead(): void {
    const following = this.following();
    if (following.kind === 'on') this.setPaneView(paneViewOf(following.frame, following.view, this.rotation()));
  }

  get regionBoxStyle(): { [k: string]: string } | null {
    const box = this.regionBox();
    if (!box) return null;
    const [x0, y0, x1, y1] = box;
    return {
      left: pct(x0),
      top: pct(y0),
      width: pct(x1 - x0),
      height: pct(y1 - y0),
    };
  }

  // Clamp the pan back into range for the current zoom/rotation; call after
  // any zoom/rotation change. (zoomLabel is a computed off `zoom` now.) A
  // follower hands its new view to its lead instead, which keeps it on the
  // lead's image, and shows what the lead settled on: its own image may be
  // smaller or larger than its face's match, so it has no range of its own.
  private applyTransform(): void {
    const following = this.following();
    if (following.kind === 'on') {
      following.lead.showFaceView(faceViewOf(following.frame, this.paneView(), this.rotation()));
      this.followLead();
      return;
    }
    const max = this.getMaxPan();
    this.panX.set(Math.max(-max.x, Math.min(max.x, this.panX())));
    this.panY.set(Math.max(-max.y, Math.min(max.y, this.panY())));
  }

  /** Keep *val* within the zoom range, which a follower reads in its lead's zoom. */
  private clampZoom(val: number): number {
    const unit = this.zoomUnit();
    return unit * Math.min(this.maxZoom, Math.max(this.minZoom, val / unit));
  }

  private recomputeRenderedSize(): void {
    const img = this.imageRef()?.nativeElement;
    const wrap = this.wrapRef()?.nativeElement;
    if (!img || !wrap) return;
    const natW = img.naturalWidth;
    const natH = img.naturalHeight;
    const wrapW = wrap.clientWidth;
    const wrapH = wrap.clientHeight;
    if (!natW || !natH || !wrapW || !wrapH) {
      this.renderedW.set(0);
      this.renderedH.set(0);
      return;
    }
    const imgAspect = natW / natH;
    const wrapAspect = wrapW / wrapH;
    if (imgAspect > wrapAspect) {
      this.renderedW.set(wrapW);
      this.renderedH.set(wrapW / imgAspect);
    } else {
      this.renderedH.set(wrapH);
      this.renderedW.set(wrapH * imgAspect);
    }
    // The first measurement of a new image is when its framing can be paid;
    // a framing paid already is not redone on a resize.
    this.payFit();
  }

  private attachWrapResizeObserver(): void {
    const wrap = this.wrapRef()?.nativeElement;
    if (!wrap || this.resizeObserver) return;
    if (typeof ResizeObserver === 'undefined') return;
    this.resizeObserver = new ResizeObserver(() => this.recomputeRenderedSize());
    this.resizeObserver.observe(wrap);
  }

  private getMaxPan(): { x: number; y: number } {
    const wrap = this.wrapRef()?.nativeElement;
    const renderedW = this.renderedW();
    const renderedH = this.renderedH();
    if (!wrap || !renderedW || !renderedH) return { x: 0, y: 0 };
    const wrapW = wrap.clientWidth;
    const wrapH = wrap.clientHeight;
    const rot = ((this.rotation() % 360) + 360) % 360;
    const swapped = rot === 90 || rot === 270;
    const effW = swapped ? renderedH : renderedW;
    const effH = swapped ? renderedW : renderedH;
    return {
      x: Math.max(0, (effW * this.zoom() - wrapW) / 2),
      y: Math.max(0, (effH * this.zoom() - wrapH) / 2),
    };
  }

  /** Ratio between the wrap's on-screen (visual) size and its layout size.
   *  The app renders ``<html>`` at ``zoom: 1.1`` (see styles.scss), and a CSS
   *  ``zoom`` scales an element's rendered box without changing its layout box:
   *  ``getBoundingClientRect()`` and ``MouseEvent`` client coords come back in
   *  VISUAL pixels, while ``clientWidth`` / ``renderedW`` / ``panX`` are LAYOUT
   *  pixels. Dividing a client-space delta by this ratio converts it back to the
   *  layout space the region overlay is drawn in, so the box lands under the
   *  cursor. Returns 1 when there's no zoom (ratio is exactly 1) or the wrap
   *  isn't measurable (e.g. the test mock exposes no ``clientWidth``). */
  private layoutScale(wrap: HTMLElement, rect: DOMRect): number {
    const cw = wrap.clientWidth;
    return cw && rect.width ? rect.width / cw : 1;
  }

  /** Convert a screen-space mouse event to normalised image coords (pre-rotation).
   *  Returns null when the image isn't laid out yet. Public so tests can drive it
   *  with a mocked wrapRef + arbitrary pan/zoom/rotate state. */
  screenToImageNormalized(event: MouseEvent): { x: number; y: number } | null {
    const wrap = this.wrapRef()?.nativeElement;
    const renderedW = this.renderedW();
    const renderedH = this.renderedH();
    if (!wrap || !renderedW || !renderedH) return null;
    const rect = wrap.getBoundingClientRect();
    const s = this.layoutScale(wrap, rect);
    const dx = (event.clientX - (rect.left + rect.width / 2)) / s - this.panX();
    const dy = (event.clientY - (rect.top + rect.height / 2)) / s - this.panY();
    const sx = dx / this.zoom();
    const sy = dy / this.zoom();
    const rad = (-this.rotation() * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    const rx = sx * cos - sy * sin;
    const ry = sx * sin + sy * cos;
    return {
      x: (rx + renderedW / 2) / renderedW,
      y: (ry + renderedH / 2) / renderedH,
    };
  }

  private setupWindowMouseListeners(): void {
    this.removeWindowMouseListeners();
    this.mouseMoveHandler = (e: MouseEvent) => this.onWindowMouseMove(e);
    this.mouseUpHandler = () => this.onWindowMouseUp();
    window.addEventListener('mousemove', this.mouseMoveHandler);
    window.addEventListener('mouseup', this.mouseUpHandler);
  }

  private removeWindowMouseListeners(): void {
    if (this.mouseMoveHandler) {
      window.removeEventListener('mousemove', this.mouseMoveHandler);
      this.mouseMoveHandler = null;
    }
    if (this.mouseUpHandler) {
      window.removeEventListener('mouseup', this.mouseUpHandler);
      this.mouseUpHandler = null;
    }
  }

  private onWindowMouseMove(e: MouseEvent): void {
    if (!this.drag) return;
    const d = this.drag;
    if (d.kind === 'pan') {
      // clientX deltas are VISUAL px (the app renders at zoom: 1.1) but panX is
      // a LAYOUT-px transform value; convert so the image tracks the cursor 1:1.
      const wrap = this.wrapRef()?.nativeElement;
      const s = wrap ? this.layoutScale(wrap, wrap.getBoundingClientRect()) : 1;
      this.panX.set(d.originX + (e.clientX - d.startX) / s);
      this.panY.set(d.originY + (e.clientY - d.startY) / s);
      this.applyTransform();
      return;
    }
    const local = this.screenToImageNormalized(e);
    if (!local) return;
    if (d.kind === 'draw') {
      const ax = d.anchor.x;
      const ay = d.anchor.y;
      const bx = clamp01(local.x);
      const by = clamp01(local.y);
      this.regionBox.set([Math.min(ax, bx), Math.min(ay, by), Math.max(ax, bx), Math.max(ay, by)]);
      return;
    }
    if (d.kind === 'move') {
      const dx = local.x - d.startLocal.x;
      const dy = local.y - d.startLocal.y;
      const [sx0, sy0, sx1, sy1] = d.startBox;
      const w = sx1 - sx0;
      const h = sy1 - sy0;
      const x0 = clamp(sx0 + dx, 0, 1 - w);
      const y0 = clamp(sy0 + dy, 0, 1 - h);
      this.regionBox.set([x0, y0, x0 + w, y0 + h]);
      return;
    }
    // resize: like draw, the dragged handle may cross the opposite edge and
    // flip the box rather than hitting a wall. The edge(s) the handle does not
    // control stay anchored at their start position; the controlled edge follows
    // the cursor, and per-axis min/max re-normalisation swaps which side is which
    // when the cursor crosses over. (e.g. drag the west handle past the east edge
    // and the dragged handle becomes the new east edge.)
    const [sx0, sy0, sx1, sy1] = d.startBox;
    const lx = clamp01(local.x);
    const ly = clamp01(local.y);
    let [x0, y0, x1, y1] = d.startBox;
    if (d.handle.includes('w')) [x0, x1] = [Math.min(lx, sx1), Math.max(lx, sx1)];
    else if (d.handle.includes('e')) [x0, x1] = [Math.min(sx0, lx), Math.max(sx0, lx)];
    if (d.handle.includes('n')) [y0, y1] = [Math.min(ly, sy1), Math.max(ly, sy1)];
    else if (d.handle.includes('s')) [y0, y1] = [Math.min(sy0, ly), Math.max(sy0, ly)];
    this.regionBox.set([x0, y0, x1, y1]);
  }

  private onWindowMouseUp(): void {
    if (!this.drag) {
      this.removeWindowMouseListeners();
      return;
    }
    const drag = this.drag;
    this.drag = null;
    this.removeWindowMouseListeners();
    const box = this.regionBox();
    if (!box) return;
    const [x0, y0, x1, y1] = box;
    const tooSmall = x1 - x0 < MIN_BOX_SIZE || y1 - y0 < MIN_BOX_SIZE;
    if (drag.kind === 'draw' && tooSmall) {
      // Zero-area Shift-drag (a stray click without motion). Restore the
      // prior box rather than discarding it; drawing a box is real work.
      // Don't emit: the parent's last-known state was already previousBox
      // (the transient zero-area draw was never emitted).
      this.regionBox.set(drag.previousBox);
      return;
    }
    if (drag.kind === 'resize' && tooSmall) {
      // The handle was released right at the flip point, collapsing the box to
      // (near) zero area. Restore the pre-resize box; startBox was the parent's
      // last-known state, so no emit is needed.
      this.regionBox.set(drag.startBox);
      return;
    }
    this.regionBoxChange.emit(box);
  }

  private setupWindowKeyListeners(): void {
    this.keyDownHandler = (e: KeyboardEvent) => this.onWindowKeyDown(e);
    this.keyUpHandler = (e: KeyboardEvent) => this.onWindowKeyUp(e);
    this.blurHandler = () => {
      // Releasing focus (alt-tab, etc.) drops the Shift state; don't leave the
      // user stuck in region mode invisibly.
      this.shiftHeld.set(false);
    };
    window.addEventListener('keydown', this.keyDownHandler);
    window.addEventListener('keyup', this.keyUpHandler);
    window.addEventListener('blur', this.blurHandler);
  }

  private removeWindowKeyListeners(): void {
    if (this.keyDownHandler) {
      window.removeEventListener('keydown', this.keyDownHandler);
      this.keyDownHandler = null;
    }
    if (this.keyUpHandler) {
      window.removeEventListener('keyup', this.keyUpHandler);
      this.keyUpHandler = null;
    }
    if (this.blurHandler) {
      window.removeEventListener('blur', this.blurHandler);
      this.blurHandler = null;
    }
  }

  private onWindowKeyDown(e: KeyboardEvent): void {
    if (e.key === 'Shift') {
      this.shiftHeld.set(true);
      return;
    }
    if (e.key !== 'Escape' || this.isTyping()) return;
    // Skip when a modal is open (matching KeyboardService): the same Esc that
    // closes the modal must not also clear the drawn region box underneath.
    if (document.querySelector('.modal-backdrop')) return;
    // Esc while a bad-vote-with-box discard is armed cancels the armed state but
    // keeps the box (drawing a box is real work,
    // and Esc should be the "I changed my mind about voting no" out, not "throw
    // away the box"). Only consume the key if we actually had an action to take.
    if (this.pendingBadConfirm()) {
      e.preventDefault();
      this.armedConfirmCanceled.emit();
      return;
    }
    if (this.regionBox()) {
      e.preventDefault();
      this.clearRegionBox({ emit: true });
    }
  }

  private onWindowKeyUp(e: KeyboardEvent): void {
    if (e.key === 'Shift') this.shiftHeld.set(false);
  }

  private isTyping(): boolean {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName;
    if (tag === 'INPUT') {
      const type = (el as HTMLInputElement).type;
      if (type !== 'checkbox' && type !== 'radio' && type !== 'range') return true;
    }
    if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if ((el as HTMLElement).isContentEditable) return true;
    return false;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function clamp01(v: number): number {
  return clamp(v, 0, 1);
}

/** Two boxes with the same corners, or both none. */
function sameBox(a: RegionBox | null, b: RegionBox | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
}

function pct(v: number): string {
  return (v * 100).toFixed(3) + '%';
}

/** Percent-position style for a normalised overlay box, or null when the box
 *  is missing, malformed or degenerate (or, with *dropWholeImage*, covers
 *  effectively the whole image). */
function overlayBoxStyle(
  box: RegionBox | null,
  { dropWholeImage }: { dropWholeImage: boolean },
): { [k: string]: string } | null {
  if (!box || box.length !== 4) return null;
  const [x0, y0, x1, y1] = box;
  if (![x0, y0, x1, y1].every((v) => Number.isFinite(v))) return null;
  const w = x1 - x0;
  const h = y1 - y0;
  if (w <= 0 || h <= 0) return null;
  if (dropWholeImage && w >= 0.99 && h >= 0.99) return null;
  return {
    left: pct(x0),
    top: pct(y0),
    width: pct(w),
    height: pct(h),
  };
}
