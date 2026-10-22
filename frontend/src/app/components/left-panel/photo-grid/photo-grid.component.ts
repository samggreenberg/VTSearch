import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { HttpErrorResponse } from '@angular/common/http';
import { DecimalPipe } from '@angular/common';
import { Subject, catchError, finalize, map, of, switchMap } from 'rxjs';

import type { CollapsedFace } from '../../../generated/api-client/models/collapsed-face';
import type { CollapsedPhoto } from '../../../generated/api-client/models/collapsed-photo';
import type { SortCollapsedResponse } from '../../../generated/api-client/models/sort-collapsed-response';
import { ActiveContextService } from '../../../services/active-context.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { SiblingLoadService } from '../../../services/sibling-load.service';
import { SortingApiService } from '../../../services/sorting-api.service';

/** Photo rows asked for per page; the server caps a page at 2000. */
export const PHOTO_PAGE = 200;
const PHOTO_PAGE_MAX = 2000;
/** Page in more when fewer rows than this are left to show (Find hides verified faces). */
const MIN_VISIBLE_ROWS = 24;
/** Page in more once the scroll is this close to the bottom, in px. */
const SCROLL_SLACK_PX = 400;

/** Why the grid has no photos to show. */
type Problem = { kind: 'not-loaded'; datasetId: string } | { kind: 'expired' } | { kind: 'error'; message: string };

/** A photo row as shown: only the faces not hidden, scored as the best of them. */
export interface ShownPhoto {
  row: CollapsedPhoto;
  faces: CollapsedFace[];
  score: number;
  /** The decision line falls just before this row. */
  lineBefore: boolean;
}

interface Page {
  token: string;
  replace: boolean;
  response: SortCollapsedResponse | null;
  problem: Problem | null;
}

/**
 * Collapse to photos (#4751): the ranking on screen as the photos its faces
 * were cut from, each with every matching face outlined.
 *
 * The server groups the whole ranking under the sort token by photo before it
 * windows it (`GET /api/sort/collapsed`), so this pages photo rows and never
 * regroups. A photo scores as its best face and keeps that face's place.
 *
 * Faces scoring under the line are drawn faint: in the ranking, and votable,
 * but not matches.
 *
 * Every outlined face, and its chip under the photo, is that face's card:
 * the same click / hover / right-click gestures as `vt-media-item` in the
 * current focus mode, emitted as the face's media id, so votes land on faces
 * and the host's vote flow (undo, autoselect, a learned re-sort) is the
 * list's. Votes are drawn from the host's live vote sets, not the page.
 *
 * Photos are read out of the Image dataset by that dataset's id alone
 * (`ActiveContextService.mediaUrlIn`), never with the active detector's, as
 * Show in photo does. When it is not loaded the server answers 409 and the
 * grid offers to load it.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-photo-grid',
  standalone: true,
  imports: [DecimalPipe],
  providers: [SiblingLoadService],
  templateUrl: './photo-grid.component.html',
  styleUrl: './photo-grid.component.scss',
})
export class PhotoGridComponent {
  private readonly sortingApi = inject(SortingApiService);
  private readonly activeContext = inject(ActiveContextService);
  private readonly datasetState = inject(DatasetStateService);
  private readonly siblingLoad = inject(SiblingLoadService);
  private readonly destroyRef = inject(DestroyRef);

  /** The ranking's `sort_token`; null before there is a ranking. */
  readonly token = input<string | null>(null);
  readonly threshold = input<number | null>(null);
  readonly selectedId = input<number | null>(null);
  readonly goodVotes = input<Set<number>>(new Set());
  readonly badVotes = input<Set<number>>(new Set());
  /** Faces to leave out (Tune's verified faces, as its list leaves them out). */
  readonly hiddenIds = input<Set<number>>(new Set());
  /** The list's thumbnail goal width; a photo column is twice it. */
  readonly gridGoalWidth = input(80);
  readonly focusMode = input<'click' | 'hover'>('click');
  /** Said when every face of the ranking is hidden. */
  readonly emptyNote = input('No faces to show.');
  /** What the header says stays per face (Tune adds its work-queue actions). */
  readonly scopeNote = input('counts and votes are per face');

  readonly mediaSelect = output<number>();
  readonly mediaVote = output<{ id: number; vote: 'good' | 'bad' }>();
  readonly mediaContextRequest = output<{ id: number; x: number; y: number }>();

  private readonly scroller = viewChild<ElementRef<HTMLElement>>('scroller');

  private readonly rows = signal<CollapsedPhoto[]>([]);
  readonly total = signal(0);
  readonly faceTotal = signal(0);
  readonly hasMore = signal(false);
  readonly loading = signal(false);
  readonly problem = signal<Problem | null>(null);
  /** The token the rows belong to, so a page of a superseded ranking is dropped. */
  private rowsToken: string | null = null;

  readonly siblingLoading = this.siblingLoad.loading;
  readonly loadError = this.siblingLoad.error;

  private readonly pages$ = new Subject<{ token: string; offset: number; limit: number; replace: boolean }>();

  /** The rows with their hidden faces left out, and where the line falls among them. */
  readonly shown = computed<ShownPhoto[]>(() => {
    const hidden = this.hiddenIds();
    const threshold = this.threshold();
    const out: ShownPhoto[] = [];
    let lineDrawn = threshold == null;
    for (const row of this.rows()) {
      const faces = hidden.size ? row.faces.filter((f) => !hidden.has(f.media_id)) : row.faces;
      if (!faces.length) continue;
      const score = Math.max(...faces.map((f) => f.score));
      const lineBefore = !lineDrawn && threshold != null && score < threshold;
      if (lineBefore) lineDrawn = true;
      out.push({ row, faces, score, lineBefore });
    }
    return out;
  });

  /** The name of the dataset the photos live in. */
  readonly siblingName = computed(() => {
    const p = this.problem();
    const id = p?.kind === 'not-loaded' ? p.datasetId : '';
    return (id && this.datasetState.datasetById().get(id)?.name) || 'the photo dataset';
  });

  readonly photoWidth = computed(() => Math.max(120, this.gridGoalWidth() * 2));

  constructor() {
    this.pages$
      .pipe(
        switchMap((req) => {
          this.loading.set(true);
          return this.sortingApi.getSortCollapsed(req.token, req.offset, req.limit).pipe(
            map((response): Page => ({ token: req.token, replace: req.replace, response, problem: null })),
            catchError((err: unknown) =>
              of<Page>({ token: req.token, replace: req.replace, response: null, problem: problemOf(err) }),
            ),
            finalize(() => this.loading.set(false)),
          );
        }),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((page) => this.land(page));

    // A new ranking (a re-sort, a learned re-rank after a vote, a Find pass):
    // ask again from the top, for as many rows as were showing, so the grid
    // does not jump back to one page.
    effect(() => {
      const token = this.token();
      untracked(() => this.restart(token));
    });

    // Find hides faces as they are verified: keep a screenful showing.
    effect(() => {
      const few = this.shown().length < MIN_VISIBLE_ROWS;
      if (few && this.hasMore() && !this.loading() && !this.problem()) untracked(() => this.loadMore());
    });

    // Keep the selected face's photo in view, as the list keeps its card.
    effect(() => {
      const id = this.selectedId();
      if (id == null) return;
      const index = untracked(() => this.shown().findIndex((p) => p.faces.some((f) => f.media_id === id)));
      if (index < 0) return;
      queueMicrotask(() => {
        const card = this.scroller()?.nativeElement.querySelector<HTMLElement>(`[data-row="${index}"]`);
        card?.scrollIntoView?.({ block: 'nearest' });
      });
    });
  }

  private restart(token: string | null): void {
    const showing = this.rows().length;
    if (!token) {
      this.rowsToken = null;
      this.rows.set([]);
      this.total.set(0);
      this.faceTotal.set(0);
      this.hasMore.set(false);
      this.problem.set(null);
      return;
    }
    const limit = Math.min(PHOTO_PAGE_MAX, Math.max(PHOTO_PAGE, showing));
    this.pages$.next({ token, offset: 0, limit, replace: true });
  }

  private land(page: Page): void {
    if (page.problem) {
      if (page.replace) {
        // Photos of the ranking this one replaced are no photos of it.
        this.rowsToken = null;
        this.rows.set([]);
        this.hasMore.set(false);
      }
      if (page.replace || page.token === this.rowsToken) this.problem.set(page.problem);
      return;
    }
    const response = page.response!;
    if (page.replace) {
      this.rowsToken = page.token;
      this.rows.set(response.results);
    } else {
      if (page.token !== this.rowsToken) return;
      this.rows.update((rows) => [...rows, ...response.results]);
    }
    this.problem.set(null);
    this.total.set(response.total);
    this.faceTotal.set(response.face_total);
    this.hasMore.set(response.has_more);
  }

  /** Page in the next photo rows. */
  loadMore(): void {
    const token = this.rowsToken;
    if (!token || !this.hasMore() || this.loading()) return;
    this.pages$.next({ token, offset: this.rows().length, limit: PHOTO_PAGE, replace: false });
  }

  onScroll(event: Event): void {
    const el = event.target as HTMLElement;
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - SCROLL_SLACK_PX) this.loadMore();
  }

  /** Load the photo dataset the server named, then ask again. */
  loadSibling(datasetId: string): void {
    this.siblingLoad.load(datasetId, () => this.restart(this.token()));
  }

  /** Ask again after a failure. */
  retry(): void {
    this.restart(this.token());
  }

  // --- what a card shows ---

  photoUrl(row: CollapsedPhoto): string {
    if (row.source_dataset_id != null && row.source_media_id != null) {
      return this.activeContext.mediaUrlIn(row.source_dataset_id, `/api/medias/${row.source_media_id}/thumbnail`);
    }
    return this.faceUrl(row.faces[0]);
  }

  faceUrl(face: CollapsedFace): string {
    return this.activeContext.mediaUrl(`/api/medias/${face.media_id}/thumbnail`);
  }

  boxStyle(face: CollapsedFace): Record<string, string> | null {
    const box = face.box;
    if (!box || box.length !== 4) return null;
    const [x0, y0, x1, y1] = box.map((v) => Math.min(1, Math.max(0, v)));
    if (x1 <= x0 || y1 <= y0) return null;
    return { left: pct(x0), top: pct(y0), width: pct(x1 - x0), height: pct(y1 - y0) };
  }

  voteOf(id: number): 'good' | 'bad' | null {
    if (this.goodVotes().has(id)) return 'good';
    if (this.badVotes().has(id)) return 'bad';
    return null;
  }

  /**
   * The face scores under the line: it is in the ranking, and votable, but
   * not one of the matches, so it is drawn faint beside the faces that are.
   */
  isBelow(face: CollapsedFace): boolean {
    const threshold = this.threshold();
    return threshold != null && face.score < threshold;
  }

  holdsSelected(photo: ShownPhoto): boolean {
    const id = this.selectedId();
    return id != null && photo.faces.some((f) => f.media_id === id);
  }

  faceLabel(face: CollapsedFace): string {
    const vote = this.voteOf(face.media_id);
    const below = this.isBelow(face) ? ', below the line' : '';
    return `Face #${face.media_id}, score ${face.score.toFixed(2)}${below}${vote ? `, voted ${vote}` : ''}`;
  }

  photoTitle(photo: ShownPhoto): string {
    const n = photo.faces.length;
    if (photo.row.source_media_id == null) return "This face's photo isn't in the photo dataset";
    return `${n} matching face${n === 1 ? '' : 's'} in this photo; best score ${photo.score.toFixed(2)}`;
  }

  // --- gestures: a face is its card ---

  /** A click on the photo outside every face: select its best face. */
  onPhotoClick(photo: ShownPhoto): void {
    this.mediaSelect.emit(photo.faces[0].media_id);
  }

  onFaceClick(id: number, event: Event): void {
    event.stopPropagation();
    if (this.focusMode() === 'hover') this.mediaVote.emit({ id, vote: 'bad' });
    else this.mediaSelect.emit(id);
  }

  onFaceContextMenu(id: number, event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
    if (this.focusMode() === 'hover') this.mediaVote.emit({ id, vote: 'good' });
    else this.mediaContextRequest.emit({ id, x: event.clientX, y: event.clientY });
  }

  onFaceEnter(id: number): void {
    if (this.focusMode() === 'hover') this.mediaSelect.emit(id);
  }

  trackRow(photo: ShownPhoto): string {
    const row = photo.row;
    return row.source_media_id != null ? `p${row.source_dataset_id}:${row.source_media_id}` : `f${row.faces[0].media_id}`;
  }
}

function pct(v: number): string {
  return `${(v * 100).toFixed(3)}%`;
}

function problemOf(err: unknown): Problem {
  if (!(err instanceof HttpErrorResponse)) return { kind: 'error', message: String(err) };
  const body = (err.error && typeof err.error === 'object' ? err.error : {}) as Record<string, unknown>;
  if (err.status === 409 && body['error_code'] === 'source_not_loaded' && typeof body['dataset_id'] === 'string') {
    return { kind: 'not-loaded', datasetId: body['dataset_id'] };
  }
  if (err.status === 404) return { kind: 'expired' };
  return { kind: 'error', message: typeof body['message'] === 'string' ? body['message'] : err.message };
}
