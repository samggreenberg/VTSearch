import { Signal, signal } from '@angular/core';

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
  /** The zoom its toolbar shows, which the follower's shows too. */
  readonly shownZoom: Signal<number>;
  /** Show *view*, kept within its own zoom range and on its own image. */
  showFaceView(view: FaceView): void;
  /** Back to its whole image, keeping its own turn. */
  homeView(): void;
  /** Whether its image is zoomed far enough to be panned. */
  canPan(): boolean;
}

/**
 * The tie between two viewers: the leading one registers here, and the one
 * that follows reads it. A plain object rather than a service, made by the
 * component that lays the two panes out and handed to both, so each pair of
 * panes has its own.
 */
export class ViewLink {
  private readonly current = signal<LinkLead | null>(null);
  readonly lead = this.current.asReadonly();

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
