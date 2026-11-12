import { Signal, computed, signal } from '@angular/core';

/**
 * Two faces shown the same size and in the same place, zooming and panning
 * together: the Face compare layout's registration and linked zoom (#5647).
 *
 * The idea is a coordinate space each face brings with it, **face space**: its
 * origin is the midpoint of the eyes, one unit is the distance from there to
 * the midpoint of the mouth, and its axes are the image's own. Both crops are
 * levelled when they are cut (`align="roll"`), so the axes of two faces agree,
 * and a point of face space is "the same place" on either face: a point a
 * little left of and above the left eye is that on both.
 *
 * A view of a face ({@link FaceView}) is then the face-space point in the
 * middle of the pane and the size one unit is drawn at. Two panes showing the
 * same view show their faces with the eye midpoints in the same place and the
 * eye-to-mouth distances the same length. That takes a resize and a shift of
 * one image against the other, and nothing else: no warp, no stretch along one
 * axis, the aspect ratio kept. Both standards the design follows
 * (`docs/plans/face-vote-helper.md`) allow that much and no more, and the two
 * faces stay in their own panes.
 *
 * The candidate's viewer **leads**: its zoom and pan are the shared view, kept
 * on its image as they always were. The reference's viewer **follows**, drawn
 * from the lead's view at its own size and its own rotation, which stays per
 * pane; a gesture on it is handed to the lead as a view, and it shows what the
 * lead settled on.
 */

/** A face's five landmarks as served (`face_landmarks`): `[x, y]` fractions of
 *  the crop, left eye, right eye, nose tip, left and right mouth corners. */
export type FaceLandmarks = readonly (readonly number[])[];

type Point = readonly [number, number];

/** Where a face sits on its image: the midpoints of its eyes and of its mouth,
 *  as fractions of the image. */
export interface FaceAnchor {
  readonly eyes: Point;
  readonly mouth: Point;
}

/**
 * A face laid out in a pane at the image's fitted size: where its eye midpoint
 * is, in layout pixels from the image's centre before any turn, and how many
 * pixels one face unit (eye midpoint to mouth midpoint) is.
 */
export interface FaceFrame {
  readonly x: number;
  readonly y: number;
  readonly unit: number;
}

/** A view of a face, the same for any face: the face-space point in the middle
 *  of the pane, and how many layout pixels one face unit is drawn at. */
export interface FaceView {
  readonly cx: number;
  readonly cy: number;
  readonly scale: number;
}

/** A rectangle of face space, `x0 < x1` and `y0 < y1`, y pointing down as on screen. */
export interface FaceRect {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

/**
 * A face in face space: its five landmarks, in the order they are served, and
 * the rectangle its image covers. What a part of the face is located from, and
 * what says whether the image holds it (`face-regions.ts`).
 */
export interface FaceShape {
  readonly points: readonly (readonly [number, number])[];
  readonly bounds: FaceRect;
}

/** A viewer's own transform: its zoom over the fitted image, and its pan in layout pixels. */
export interface PaneView {
  readonly zoom: number;
  readonly panX: number;
  readonly panY: number;
}

/**
 * The anchor of a face from its landmarks, or null when there are none to go
 * on: not five points, a point that is not a pair of finite numbers, or eyes
 * and mouth in the same place.
 */
export function faceAnchor(landmarks: FaceLandmarks | null | undefined): FaceAnchor | null {
  if (!Array.isArray(landmarks) || landmarks.length !== 5) return null;
  const points: Point[] = [];
  for (const p of landmarks) {
    if (!Array.isArray(p) || p.length < 2 || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) return null;
    points.push([p[0], p[1]]);
  }
  const [leftEye, rightEye, , leftMouth, rightMouth] = points;
  const eyes = midpoint(leftEye, rightEye);
  const mouth = midpoint(leftMouth, rightMouth);
  if (Math.hypot(mouth[0] - eyes[0], mouth[1] - eyes[1]) < 1e-6) return null;
  return { eyes, mouth };
}

/** The face of *anchor* on an image drawn *renderedW* × *renderedH* at zoom 1,
 *  or null while it has no size. */
export function faceFrame(anchor: FaceAnchor, renderedW: number, renderedH: number): FaceFrame | null {
  if (!(renderedW > 0 && renderedH > 0)) return null;
  const unit = Math.hypot((anchor.mouth[0] - anchor.eyes[0]) * renderedW, (anchor.mouth[1] - anchor.eyes[1]) * renderedH);
  if (!(unit > 0)) return null;
  return { x: (anchor.eyes[0] - 0.5) * renderedW, y: (anchor.eyes[1] - 0.5) * renderedH, unit };
}

/**
 * The face of *landmarks* in face space, on an image drawn *renderedW* ×
 * *renderedH* at zoom 1; null when {@link faceAnchor} or {@link faceFrame} is.
 */
export function faceShape(
  landmarks: FaceLandmarks | null | undefined,
  renderedW: number,
  renderedH: number,
): FaceShape | null {
  const anchor = faceAnchor(landmarks);
  const frame = anchor && faceFrame(anchor, renderedW, renderedH);
  if (!frame) return null;
  const at = (u: number, v: number): Point => [
    ((u - 0.5) * renderedW - frame.x) / frame.unit,
    ((v - 0.5) * renderedH - frame.y) / frame.unit,
  ];
  const [x0, y0] = at(0, 0);
  const [x1, y1] = at(1, 1);
  return { points: landmarks!.map((p) => at(p[0], p[1])), bounds: { x0, y0, x1, y1 } };
}

/**
 * The view that fits *rect* in a pane *paneW* × *paneH*, the rectangle's centre
 * in the middle of it, for a face turned by *rotationDeg* (quarter turns, so a
 * rectangle on its side swaps width for height). Null when there is nothing to
 * fit, or nothing to fit it in.
 */
export function viewOfRect(rect: FaceRect, paneW: number, paneH: number, rotationDeg: number): FaceView | null {
  const w = rect.x1 - rect.x0;
  const h = rect.y1 - rect.y0;
  if (!(w > 0 && h > 0 && paneW > 0 && paneH > 0)) return null;
  const quarter = ((Math.round(rotationDeg / 90) % 4) + 4) % 4;
  const swapped = quarter === 1 || quarter === 3;
  const scale = Math.min(paneW / (swapped ? h : w), paneH / (swapped ? w : h));
  return { cx: (rect.x0 + rect.x1) / 2, cy: (rect.y0 + rect.y1) / 2, scale };
}

/** Two views that draw a face the same, to within what arithmetic leaves over. */
export function sameFaceView(a: FaceView, b: FaceView): boolean {
  const near = (x: number, y: number, size: number) => Math.abs(x - y) <= 1e-6 * size;
  return near(a.cx, b.cx, 1) && near(a.cy, b.cy, 1) && near(a.scale, b.scale, Math.max(a.scale, b.scale));
}

/*
 * The viewer draws an image point P (layout px from the image's centre, before
 * the turn) at `pan + zoom · R(θ) · P` from the pane's centre, where R turns
 * clockwise by θ on a screen whose y points down. A face-space point q is the
 * image point `E + unit · q`, E being the frame's eye midpoint. So the point in
 * the middle of the pane is the q whose image point is `R(-θ) · (-pan) / zoom`,
 * and a unit is `zoom · unit` pixels; going back, `pan = -zoom · R(θ) · (E + unit · c)`.
 * Under either, q lands at `scale · R(θ) · (q - c)` from the pane's centre, the
 * same for every face: that is the registration.
 */

/** The face view a pane shows with its own transform. */
export function faceViewOf(frame: FaceFrame, pane: PaneView, rotationDeg: number): FaceView {
  const [px, py] = turn(-pane.panX / pane.zoom, -pane.panY / pane.zoom, -rotationDeg);
  return { cx: (px - frame.x) / frame.unit, cy: (py - frame.y) / frame.unit, scale: pane.zoom * frame.unit };
}

/** The transform that shows *view* of the face in *frame*, turned by *rotationDeg*. */
export function paneViewOf(frame: FaceFrame, view: FaceView, rotationDeg: number): PaneView {
  const zoom = view.scale / frame.unit;
  const [x, y] = turn(frame.x + frame.unit * view.cx, frame.y + frame.unit * view.cy, rotationDeg);
  return { zoom, panX: -zoom * x, panY: -zoom * y };
}

/** What the lead's face view is: none to share, one still being laid out, or one. */
export type LeadState =
  | { readonly kind: 'off' }
  | { readonly kind: 'pending' }
  | { readonly kind: 'on'; readonly frame: FaceFrame; readonly view: FaceView };

export const LEAD_OFF: LeadState = { kind: 'off' };
export const LEAD_PENDING: LeadState = { kind: 'pending' };

/** What a leading viewer offers the viewer that follows it. */
export interface LinkLead {
  /** The face view it shows, or why there is none. */
  readonly leadState: Signal<LeadState>;
  /** Its face in face space, once its image is laid out. */
  readonly faceShape: Signal<FaceShape | null>;
  /** The zoom its toolbar shows, which the follower's shows too. */
  readonly shownZoom: Signal<number>;
  /** Show *view*, kept within its own zoom range and on its own image. */
  showFaceView(view: FaceView): void;
  /** Show *rect* of its face, as large as its pane and its zoom range allow. */
  showFaceRect(rect: FaceRect): void;
  /** Back to its whole image, keeping its own turn. */
  homeView(): void;
  /** Whether its image is zoomed far enough to be panned. */
  canPan(): boolean;
}

/** What a following viewer tells the link about itself. */
export interface LinkFollower {
  /** It is drawn from the lead's view right now. */
  readonly linked: Signal<boolean>;
  /** Its face in face space, once its image is laid out. */
  readonly faceShape: Signal<FaceShape | null>;
}

/** The two faces of a link, while they are matched. */
export interface LinkedFaces {
  readonly lead: FaceShape;
  readonly follower: FaceShape;
}

/**
 * The tie between two viewers: both register here, the one that follows reads
 * the one that leads, and anything else that wants to move the pair (the
 * checklist's rows) reads {@link faces} and asks the lead. A plain object
 * rather than a service, made by the component that lays the two panes out and
 * handed to both, so each pair of panes has its own.
 */
export class ViewLink {
  private readonly current = signal<LinkLead | null>(null);
  private readonly following = signal<LinkFollower | null>(null);
  readonly lead = this.current.asReadonly();

  /** Both faces, while the follower is drawn from the lead's view; else null. */
  readonly faces = computed<LinkedFaces | null>(() => {
    const lead = this.current();
    const follower = this.following();
    if (!lead || !follower?.linked()) return null;
    const a = lead.faceShape();
    const b = follower.faceShape();
    return a && b ? { lead: a, follower: b } : null;
  });

  attachFollower(follower: LinkFollower): void {
    this.following.set(follower);
  }

  detachFollower(follower: LinkFollower): void {
    if (this.following() === follower) this.following.set(null);
  }

  attach(lead: LinkLead): void {
    this.current.set(lead);
  }

  detach(lead: LinkLead): void {
    if (this.current() === lead) this.current.set(null);
  }
}

function midpoint(a: Point, b: Point): Point {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

/** (*x*, *y*) turned clockwise by *deg* on a screen whose y points down. */
function turn(x: number, y: number, deg: number): Point {
  const rad = (deg * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return [x * cos - y * sin, x * sin + y * cos];
}
