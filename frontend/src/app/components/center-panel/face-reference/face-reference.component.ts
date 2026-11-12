import { ChangeDetectionStrategy, Component, computed, effect, inject, input, signal } from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';
import { Media } from '../../../models/api.models';
import { ActiveDatasetService } from '../../../services/active-dataset.service';
import { FaceReference, FaceReferenceService, referenceKey } from '../../../services/face-reference.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { SourcePhotoService, WiderView, widerView } from '../../../services/source-photo.service';
import { ImageViewerComponent } from '../image-viewer/image-viewer.component';
import { ImageViewControlsComponent } from '../image-view-controls/image-view-controls.component';

/**
 * The reference half of the Face compare layout (#5642): a face of the person
 * being looked for, shown beside the candidate the user is voting on, so the
 * two are compared on screen rather than against memory.
 *
 * A person is not one photo, so the reference is a set the user steps through
 * (`FaceReferenceService.pool`) without the candidate moving. Each new
 * candidate starts on the best reference again, or on the pinned one: a step
 * is a look at this candidate from another angle, and a pin is the answer to
 * "always start from this one".
 *
 * The two faces are never laid over each other, blinked or wiped. The practice
 * manuals disagree on overlays, and the stricter one (ENFSI-BPM-DI-01) cites
 * studies in which they lowered matching accuracy, so this pane has its own
 * viewer and its own zoom.
 *
 * Under the layout's Wider toggle (#5645) a reference that is a face of the
 * dataset is shown from its photo, framed wide on it, as the candidate is. An
 * example picture is a file of the user's with no photo behind it, so it stays
 * as it is.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-face-reference',
  standalone: true,
  imports: [ImageViewerComponent, ImageViewControlsComponent],
  templateUrl: './face-reference.component.html',
  styleUrl: './face-reference.component.scss',
})
export class FaceReferenceComponent {
  private readonly refs = inject(FaceReferenceService);
  private readonly metadataCache = inject(MediaMetadataCacheService);
  private readonly sourcePhoto = inject(SourcePhotoService);
  private readonly activeDataset = inject(ActiveDatasetService);

  /** The id of the face being voted on, which is never its own reference. */
  readonly candidateId = input.required<number>();
  /** Show the reference from its photo, wider (#5645); the centre panel says
   *  when, since it owns the toggle and knows whether the photos are loaded. */
  readonly wider = input(false);

  /**
   * The reference stepped to, and the candidate it was stepped to for. Kept
   * together so a new candidate drops the step in the same read that sees the
   * new id, with no effect to run a frame late.
   */
  private readonly step = signal<{ candidateId: number; key: string } | null>(null);

  /** The references for this candidate: the pinned one first, then the rest. */
  readonly references = computed<FaceReference[]>(() => {
    const candidateId = this.candidateId();
    const isCandidate = (ref: FaceReference): boolean => ref.kind === 'media' && ref.id === candidateId;
    const pool = this.refs.pool().filter((ref) => !isCandidate(ref));
    const pin = this.refs.pinned();
    if (!pin || isCandidate(pin)) return pool;
    const pinKey = referenceKey(pin);
    return [pin, ...pool.filter((ref) => referenceKey(ref) !== pinKey)];
  });

  /** Where the reference on screen sits in {@link references}. */
  readonly index = computed(() => {
    const step = this.step();
    if (!step || step.candidateId !== this.candidateId()) return 0;
    // Found by key, not by position: the pile re-ranks as qualities arrive.
    const at = this.references().findIndex((ref) => referenceKey(ref) === step.key);
    return at === -1 ? 0 : at;
  });

  readonly current = computed<FaceReference | null>(() => this.references()[this.index()] ?? null);

  readonly isPinned = computed(() => {
    const current = this.current();
    const pin = this.refs.pinned();
    return !!current && !!pin && referenceKey(current) === referenceKey(pin);
  });

  /** Whether the reference on screen is a hard view, for the checklist's fitness notice (#5643). */
  readonly currentHardView = computed<boolean | null>(() => {
    const ref = this.current();
    return ref ? this.refs.hardView(ref) : null;
  });

  /** What the viewer is handed: the Good pile's item, or a stand-in naming the example. */
  readonly viewerMedia = computed<Media>(() => {
    const ref = this.current();
    if (ref?.kind !== 'media') return { id: -1, media_type: 'image', filename: 'Example picture' };
    this.metadataCache.version();
    return this.metadataCache.get(ref.id) ?? { id: ref.id, media_type: 'face' };
  });

  /** The example's own picture, full size; `''` for an item of the dataset. */
  readonly exampleSrc = computed(() => {
    const ref = this.current();
    return ref?.kind === 'example' ? `/api/server-media-files/${encodeURIComponent(ref.filename)}/image` : '';
  });

  /** Where the reference comes from, in the caption. */
  readonly sourceLabel = computed(() => {
    const ref = this.current();
    if (!ref) return '';
    if (ref.kind === 'example') return 'Your example';
    return this.viewerMedia().filename || `Face #${ref.id}`;
  });

  /**
   * The reference to look up a photo for under Wider: a face of the dataset,
   * keyed on the dataset too, since media ids are per dataset. Compared by
   * value, so a new candidate starting on the same reference asks nothing new.
   */
  private readonly sourceKey = computed(
    () => {
      const ref = this.current();
      if (!this.wider() || ref?.kind !== 'media') return undefined;
      return { datasetId: this.activeDataset.activeId(), mediaId: ref.id };
    },
    { equal: (a, b) => a?.datasetId === b?.datasetId && a?.mediaId === b?.mediaId },
  );

  private readonly sourceLookup = rxResource({
    params: () => this.sourceKey(),
    stream: ({ params }) => this.sourcePhoto.lookup(params.mediaId),
  });

  /** What the viewer shows: the reference as it is, or its photo (see {@link WiderView}). */
  readonly view = computed<WiderView>(() => {
    if (!this.sourceKey()) return { kind: 'crop' };
    return widerView(this.sourceLookup.value(), this.sourceLookup.isLoading(), this.viewerMedia().filename);
  });

  constructor() {
    // The Good pile's ranking needs each face's quality, which rides on its
    // metadata; ask for whatever the pile has gained.
    effect(() => this.refs.loadQualities());
    // An example's hard-view verdict is read by the server, once a page.
    effect(() => {
      const ref = this.current();
      if (ref) this.refs.loadHardView(ref);
    });
  }

  /** Step to the next reference, wrapping; the candidate stays where it is. */
  next(): void {
    const references = this.references();
    if (references.length < 2) return;
    const to = references[(this.index() + 1) % references.length];
    this.step.set({ candidateId: this.candidateId(), key: referenceKey(to) });
  }

  togglePin(): void {
    const current = this.current();
    if (!current) return;
    if (this.isPinned()) this.refs.unpin();
    else this.refs.pin(current);
  }
}
