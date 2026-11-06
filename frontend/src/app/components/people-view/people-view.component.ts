import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  OnDestroy,
  computed,
  effect,
  inject,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import { DecimalPipe, TitleCasePipe } from '@angular/common';
import { takeUntilDestroyed, toSignal } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router } from '@angular/router';
import { ScrollingModule } from '@angular/cdk/scrolling';
import { map } from 'rxjs';

import type { CoPerson } from '../../generated/api-client/models/co-person';
import type { PeopleList } from '../../generated/api-client/models/people-list';
import type { PersonSummary } from '../../generated/api-client/models/person-summary';
import type { SharedPhoto } from '../../generated/api-client/models/shared-photo';
import type { SharedPhotoFace } from '../../generated/api-client/models/shared-photo-face';
import { PanelResizeDirective } from '../../directives/panel-resize.directive';
import type { ProgressEvent } from '../../models/api.models';
import { ActiveContextService } from '../../services/active-context.service';
import { ActiveDatasetService } from '../../services/active-dataset.service';
import { DatasetStateService } from '../../services/dataset-state.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { MediaTypeCapabilityService } from '../../services/media-type-capability.service';
import { PairRecoveryService } from '../../services/pair-recovery.service';
import { PeopleViewStateService, type PeoplePick } from '../../services/people-view-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { SiblingLoadService } from '../../services/sibling-load.service';
import { SourcePhotoService } from '../../services/source-photo.service';
import { binGridColumns } from '../../utils/bin-grid-metrics';
import { formatEta, formatProgressHeader, progressBarState } from '../../utils/format-progress';
import { formatMetadataValue } from '../../utils/format-metadata';
import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';
import { BrowseBinMemberGridComponent } from '../browse-bin-member-grid/browse-bin-member-grid.component';
import type { OutlineBox, OutlineRole } from '../center-panel/image-viewer/image-viewer.component';
import { IconComponent } from '../icon/icon.component';
import { SourcePhotoModalComponent } from '../modals/source-photo-modal/source-photo-modal.component';
import { ProgressBarComponent } from '../progress-bar/progress-bar.component';

/** One row of the people list: a person, or the unplaced faces at the end. */
type ListRow = { kind: 'person'; person: PersonSummary } | { kind: 'unplaced'; count: number };

/** Height (px) of one people-list row: the thumbnail and its padding. */
export const PEOPLE_ROW_PX = 60;
/** The side panes' starting widths and their floors. */
const LIST_WIDTH = 280;
const WITH_WIDTH = 340;
const SIDE_MIN = 200;
const CENTER_MIN = 320;
const DIVIDER_TOTAL = 16;
/** Horizontal room the centre's padding and scrollbar take from the grid. */
const GRID_CHROME = 32;

/** The Face media type: what a Face dataset's faces are. */
const FACE_TYPE = 'face';

/**
 * The People view (#5590): a Face dataset's people (its face crops merged at
 * the face ball's cut), each one's faces, and who they appear with in the
 * photos the faces were cut from. A dataset-only view at `/people/:datasetId`,
 * beside Browse; the design is `docs/plans/co-face-browse.md`.
 *
 * Three panes, in Browse's layout idiom:
 *
 * - **People** (left): one row per person, sorted by faces, with a minimum
 *   count that folds away the one-face tail, and an **Unplaced** row last.
 * - **Person** (centre): the face shown large with its metadata, over the
 *   person's faces in the bin details' grid (`vt-browse-bin-member-grid`).
 * - **Appears with** (right): who shares a photo with the person, and the
 *   photos they share with the one picked, every face in each outlined by
 *   whose it is. A photo opens in Show in photo with the same outlines.
 *
 * State lives in the component-provided {@link PeopleViewStateService}. The
 * photos are items of the Image dataset the import made beside the faces, read
 * with that dataset's id alone (`mediaUrlIn`) while the Face dataset stays
 * active; when it is not loaded the strip offers to load it
 * ({@link SiblingLoadService}), and the list and the faces never need it.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-people-view',
  standalone: true,
  imports: [
    DecimalPipe,
    TitleCasePipe,
    ScrollingModule,
    PanelResizeDirective,
    BrowseBinMemberGridComponent,
    IconComponent,
    ProgressBarComponent,
    SourcePhotoModalComponent,
  ],
  providers: [PeopleViewStateService, SiblingLoadService],
  templateUrl: './people-view.component.html',
  styleUrl: './people-view.component.scss',
})
export class PeopleViewComponent implements OnDestroy {
  readonly state = inject(PeopleViewStateService);
  readonly sourcePhoto = inject(SourcePhotoService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly activeContext = inject(ActiveContextService);
  private readonly activeDataset = inject(ActiveDatasetService);
  private readonly datasetState = inject(DatasetStateService);
  private readonly metadataCache = inject(MediaMetadataCacheService);
  private readonly mediaTypes = inject(MediaTypeCapabilityService);
  private readonly settingsState = inject(SettingsStateService);
  private readonly siblingLoad = inject(SiblingLoadService);
  private readonly pairRecovery = inject(PairRecoveryService);
  private readonly destroyRef = inject(DestroyRef);

  private readonly centre = viewChild<ElementRef<HTMLElement>>('centre');

  readonly ROW_PX = PEOPLE_ROW_PX;
  readonly SIDE_MIN = SIDE_MIN;
  readonly CENTER_MIN = CENTER_MIN;
  readonly DIVIDER_TOTAL = DIVIDER_TOTAL;
  readonly faceType = FACE_TYPE;

  /** The dataset the URL names; the guard has made it the active one. */
  readonly datasetId = toSignal(this.route.paramMap.pipe(map((p) => p.get('datasetId') ?? '')), {
    initialValue: '',
  });
  readonly datasetName = this.activeDataset.datasetName;

  readonly listWidth = signal(LIST_WIDTH);
  readonly withWidth = signal(WITH_WIDTH);

  // --- the list ---

  /** The list's rows: the people the filter keeps, then the unplaced faces. */
  readonly rows = computed<ListRow[]>(() => {
    const rows: ListRow[] = this.state.shownPeople().map((person) => ({ kind: 'person' as const, person }));
    const unplaced = this.state.unplacedIds().length;
    if (unplaced > 0) rows.push({ kind: 'unplaced', count: unplaced });
    return rows;
  });

  /** The headline count: people, and faces in no one. */
  readonly summary = computed(() => {
    const people = this.state.people().length;
    const unplaced = this.state.unplacedIds().length;
    const p = `${people.toLocaleString()} ${people === 1 ? 'person' : 'people'}`;
    return unplaced ? `${p} · ${unplaced.toLocaleString()} unplaced ${unplaced === 1 ? 'face' : 'faces'}` : p;
  });

  // --- building ---

  readonly buildBar = computed(() => progressBarState(asProgress(this.state.progress())));
  readonly buildHeader = computed(() => formatProgressHeader(asProgress(this.state.progress()), 'people'));
  readonly buildEta = computed(() => formatEta(this.state.progress()?.eta_seconds));

  // --- the person ---

  /** The centre's heading: the person, or the unplaced faces. */
  readonly pickTitle = computed(() => {
    const pick = this.state.pick();
    if (pick === 'unplaced') return 'Unplaced faces';
    return pick == null ? '' : personName(pick);
  });
  readonly pickSubtitle = computed(() => {
    const pick = this.state.pick();
    if (pick === 'unplaced') return 'Faces that matched no one, at the same cut Find uses.';
    const s = this.state.pickedSummary();
    return s ? countsText(s) : '';
  });

  /** The grid's thumbnail size: the bin details' own preference, for faces. */
  private readonly gridSizePref = this.settingsState.perMediaType<string>('grid_icon_size_popup', signal(FACE_TYPE), {
    fallback: 'M',
    coerce: (raw) => (typeof raw === 'string' && raw ? raw : undefined),
  });
  readonly cellWidth = computed(() => iconSizeToGoalWidth(this.gridSizePref.value()));
  private readonly centreWidth = signal(0);
  readonly columns = computed(() => binGridColumns(Math.max(0, this.centreWidth() - GRID_CHROME), this.cellWidth()));
  readonly faceRows = computed(() => chunk(this.state.faceIds(), this.columns()));

  private readonly hoveredFace = signal<number | null>(null);
  /** The face in the large pane: the one hovered in the grid, else the one clicked. */
  readonly shownFace = computed(() => {
    const hovered = this.hoveredFace();
    return hovered != null && this.state.faceIds().includes(hovered) ? hovered : this.state.focusedFace();
  });

  /** The large face's metadata rows (name first), once its batch has landed. */
  readonly faceMetadata = computed(() => {
    const id = this.shownFace();
    this.metadataCache.version();
    const media = id == null ? undefined : this.metadataCache.get(id);
    if (!media) return [];
    const rows: { label: string; value: string }[] = [{ label: 'Name', value: media.filename || `Face #${id}` }];
    for (const [label, value] of Object.entries((media.custom_metadata as Record<string, unknown>) ?? {})) {
      rows.push({ label, value: formatMetadataValue(label, value) });
    }
    return rows;
  });

  // --- appears with ---

  readonly coPick = this.state.coPick;
  readonly siblingLoading = this.siblingLoad.loading;
  readonly siblingLoadError = this.siblingLoad.error;

  /** Where the photos are, by name (or would be, once loaded). */
  readonly photoDatasetName = computed(() => {
    const read = this.state.shared();
    const fromRead = read?.kind === 'not-loaded' ? read.datasetId : '';
    const fromPhotos = read?.kind === 'found' ? (read.photos.find((p) => p.source_dataset_id)?.source_dataset_id ?? '') : '';
    const id = fromRead || fromPhotos;
    return (id && this.datasetState.datasetById().get(id)?.name) || 'the photo dataset';
  });

  constructor() {
    this.mediaTypes.ensureLoaded();

    // A new dataset in the URL (or the first) reads its people from the start.
    effect(() => {
      if (this.datasetId()) untracked(() => this.state.open());
    });

    // The backend lost the dataset and has loaded it again (#5158): whatever
    // it held for this view went with it, so start over.
    this.pairRecovery.reloaded$.pipe(takeUntilDestroyed()).subscribe(() => this.state.open());

    // The large face's metadata.
    effect(() => {
      const id = this.shownFace();
      if (id != null) untracked(() => this.metadataCache.ensureLoaded([id]));
    });

    // The centre's width decides the grid's columns.
    effect((onCleanup) => {
      const el = this.centre()?.nativeElement;
      if (!el || typeof ResizeObserver === 'undefined') return;
      this.centreWidth.set(el.clientWidth);
      const observer = new ResizeObserver(() => this.centreWidth.set(el.clientWidth));
      observer.observe(el);
      onCleanup(() => observer.disconnect());
    });
  }

  ngOnDestroy(): void {
    // The overlay is rendered here; leaving the view must not leave it armed
    // to reopen over the next one.
    this.sourcePhoto.close();
  }

  // --- toolbar ---

  backToDashboard(): void {
    this.router.navigate(['/dashboard']);
  }

  openBrowse(): void {
    this.router.navigate(['/browse', this.datasetId()]);
  }

  rebuild(): void {
    this.state.rebuild();
  }

  retry(): void {
    this.state.open();
  }

  // --- list ---

  isPicked(pick: PeoplePick): boolean {
    return this.state.pick() === pick;
  }

  choose(pick: PeoplePick): void {
    this.state.choose(pick);
  }

  onMinFacesInput(event: Event): void {
    this.state.setMinFaces(Number((event.target as HTMLInputElement).value));
  }

  showAllPeople(): void {
    this.state.setMinFaces(1);
  }

  trackRow(_index: number, row: ListRow): string {
    return row.kind === 'person' ? `p${row.person.id}` : 'unplaced';
  }

  // --- faces ---

  onFaceClick(id: number): void {
    this.state.focusFace(id);
  }

  onFaceEnter(id: number): void {
    this.hoveredFace.set(id);
  }

  onGridLeave(): void {
    this.hoveredFace.set(null);
  }

  /** Show the large face's photo, the face outlined. */
  showFaceInPhoto(): void {
    const id = this.shownFace();
    if (id != null) this.sourcePhoto.open(id, { fromName: 'People' });
  }

  // --- appears with ---

  chooseCoPerson(co: CoPerson): void {
    this.state.chooseCoPerson(co.id);
  }

  loadPhotos(datasetId: string): void {
    this.siblingLoad.load(datasetId, () => this.state.reloadShared());
  }

  /** Whose face *face* is, on a photo of the picked person and their co-person. */
  roleOf(face: SharedPhotoFace): OutlineRole {
    if (face.person != null && face.person === this.state.person()?.id) return 'primary';
    if (face.person != null && face.person === this.coPick()) return 'secondary';
    return 'other';
  }

  /** A face's hover title on a shared photo: whose it is, in words. */
  faceTitle(face: SharedPhotoFace): string {
    const role = this.roleOf(face);
    if (face.person == null) return 'Someone in no person (an unplaced face)';
    const who = personName(face.person);
    return role === 'other' ? `${who}, also in this photo` : who;
  }

  /** The photos, faintest outlines first, so the two people are drawn on top. */
  facesByLayer(photo: SharedPhoto): SharedPhotoFace[] {
    const layer: Record<OutlineRole, number> = { other: 0, secondary: 1, primary: 2 };
    return [...photo.faces].sort((a, b) => layer[this.roleOf(a)] - layer[this.roleOf(b)]);
  }

  /** A face's box on its photo, as percentages, or null when it has none. */
  boxStyle(face: SharedPhotoFace): Record<string, string> | null {
    const box = face.box;
    if (!box || box.length !== 4 || !box.every((v) => Number.isFinite(v))) return null;
    const [x0, y0, x1, y1] = box.map((v) => Math.min(1, Math.max(0, v)));
    if (x1 <= x0 || y1 <= y0) return null;
    return { left: pct(x0), top: pct(y0), width: pct(x1 - x0), height: pct(y1 - y0) };
  }

  photoUrl(photo: SharedPhoto): string {
    if (photo.source_dataset_id != null && photo.source_media_id != null) {
      return this.activeContext.mediaUrlIn(photo.source_dataset_id, `/api/medias/${photo.source_media_id}/thumbnail`);
    }
    return '';
  }

  /** Open *photo* in Show in photo, every face outlined by whose it is. */
  openPhoto(photo: SharedPhoto): void {
    const own = photo.faces.find((f) => this.roleOf(f) === 'primary') ?? photo.faces[0];
    if (!own) return;
    const boxes: OutlineBox[] = [];
    for (const face of photo.faces) {
      const box = face.box;
      if (box && box.length === 4) boxes.push({ box: [box[0], box[1], box[2], box[3]], role: this.roleOf(face) });
    }
    this.sourcePhoto.open(own.media_id, { boxes, fromName: 'People' });
  }

  photoTitle(photo: SharedPhoto): string {
    const n = photo.faces.length;
    return `${n} ${n === 1 ? 'face' : 'faces'} in this photo. Click to open it.`;
  }

  // --- urls and names ---

  faceUrl(id: number): string {
    return this.activeContext.mediaUrl(`/api/medias/${id}/thumbnail`);
  }

  largeFaceUrl(id: number): string {
    return this.activeContext.mediaUrl(`/api/medias/${id}/image`);
  }

  personName(id: number): string {
    return personName(id);
  }

  countsText(person: PersonSummary): string {
    return countsText(person);
  }

  sharedText(co: CoPerson): string {
    return `${co.shared.toLocaleString()} shared ${co.shared === 1 ? 'photo' : 'photos'}`;
  }

  // --- panes ---

  onListWidthChange(width: number): void {
    this.listWidth.set(width);
  }

  onWithWidthChange(width: number): void {
    this.withWidth.set(width);
  }
}

/** People are numbered by rank, which a rebuild can change. */
function personName(id: number): string {
  return `Person ${id}`;
}

function countsText(person: PersonSummary): string {
  const faces = `${person.face_count.toLocaleString()} ${person.face_count === 1 ? 'face' : 'faces'}`;
  const photos = `${person.photo_count.toLocaleString()} ${person.photo_count === 1 ? 'photo' : 'photos'}`;
  return `${faces} · ${photos}`;
}

function chunk(ids: number[], size: number): number[][] {
  const cols = Math.max(1, size);
  const rows: number[][] = [];
  for (let i = 0; i < ids.length; i += cols) rows.push(ids.slice(i, i + cols));
  return rows;
}

function pct(v: number): string {
  return (v * 100).toFixed(3) + '%';
}

/** A `building` people list as the progress shape the bar and header helpers take. */
function asProgress(list: PeopleList | null): ProgressEvent | null {
  if (!list) return null;
  return {
    status: list.status,
    message: list.message ?? undefined,
    current: list.current ?? undefined,
    total: list.total ?? undefined,
    step: list.step,
    total_steps: list.total_steps,
    eta_seconds: list.eta_seconds,
    overall: list.overall,
    overall_step_end: list.overall_step_end,
  };
}
