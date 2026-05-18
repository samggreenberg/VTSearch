import { Injectable } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { DemoDataset } from '../models/api.models';

export interface ImporterFlowState {
  open: boolean;
  initialTab: string;
  guessedMediaType: string;
  guessedMediaEmbedder: string;
}

export interface NewDetectorFlowState {
  open: boolean;
  defaultMediaType: string;
  /** ID of a loaded media to use as the new detector's seed example.
   *  When set, the modal materialises this media to ``example_media/``
   *  on open and pre-fills the example field, skipping the picker. */
  seedMediaId?: number;
  /** Optional crop bounds applied when materialising the seed media.
   *  Shape matches the bounded clippers (audio: ``{start, end}``;
   *  image: ``{box: [x1, y1, x2, y2]}``). */
  seedCropParams?: Record<string, unknown>;
}

export interface ThingCreatedEvent {
  kind: 'dataset' | 'detector';
  /** Empty when the creator did not return an ID (e.g. the dataset
   *  importer kicks off a background load and the new ID isn't known
   *  until the load completes — callers should listen to the registry
   *  for new entries in that case). */
  id: string;
}

export interface DemoSelectedEvent {
  demo: DemoDataset;
}

/**
 * Singleton openers for the "add new dataset" and "add new detector"
 * flows. Decouples the modal invocations from the Dashboard component so
 * the top-bar context pulldowns (`vt-context-pulldown`) can open them
 * in-place over Train / Find.
 *
 * The modal components themselves are rendered as siblings of the
 * router-outlet in `AppComponent` and bind to the state subjects on
 * this service.
 */
@Injectable({ providedIn: 'root' })
export class NewThingFlowsService {
  private readonly importerSubject = new BehaviorSubject<ImporterFlowState>({
    open: false,
    initialTab: '',
    guessedMediaType: '',
    guessedMediaEmbedder: '',
  });
  private readonly newDetectorSubject = new BehaviorSubject<NewDetectorFlowState>({
    open: false,
    defaultMediaType: '',
    seedMediaId: undefined,
    seedCropParams: undefined,
  });
  private readonly createdSubject = new Subject<ThingCreatedEvent>();
  private readonly demoSelectedSubject = new Subject<DemoSelectedEvent>();
  private readonly importStartedSubject = new Subject<void>();

  readonly importer$ = this.importerSubject.asObservable();
  readonly newDetector$ = this.newDetectorSubject.asObservable();
  /** Fires after a successful import or detector-creation. The detector
   *  flow knows the new id immediately; the dataset importer flow leaves
   *  `id` empty because the new dataset's id isn't known until the
   *  background load registers it. */
  readonly created$ = this.createdSubject.asObservable();
  readonly demoSelected$ = this.demoSelectedSubject.asObservable();
  /** Fires when the dataset importer kicks off an import (background
   *  load started, but new id not yet known). Consumers refresh the
   *  registry and/or start polling for the new dataset. */
  readonly importStarted$ = this.importStartedSubject.asObservable();

  get importer(): ImporterFlowState {
    return this.importerSubject.value;
  }

  get newDetector(): NewDetectorFlowState {
    return this.newDetectorSubject.value;
  }

  openImporter(opts: Partial<Omit<ImporterFlowState, 'open'>> = {}): void {
    this.importerSubject.next({
      open: true,
      initialTab: opts.initialTab ?? '',
      guessedMediaType: opts.guessedMediaType ?? '',
      guessedMediaEmbedder: opts.guessedMediaEmbedder ?? '',
    });
  }

  closeImporter(): void {
    this.importerSubject.next({
      open: false,
      initialTab: '',
      guessedMediaType: '',
      guessedMediaEmbedder: '',
    });
  }

  openNewDetector(opts: Partial<Omit<NewDetectorFlowState, 'open'>> = {}): void {
    this.newDetectorSubject.next({
      open: true,
      defaultMediaType: opts.defaultMediaType ?? '',
      seedMediaId: opts.seedMediaId,
      seedCropParams: opts.seedCropParams,
    });
  }

  closeNewDetector(): void {
    this.newDetectorSubject.next({
      open: false,
      defaultMediaType: '',
      seedMediaId: undefined,
      seedCropParams: undefined,
    });
  }

  emitImportStarted(): void {
    this.importStartedSubject.next();
  }

  emitDemoSelected(demo: DemoDataset): void {
    this.demoSelectedSubject.next({ demo });
  }

  emitDetectorCreated(id: string): void {
    this.createdSubject.next({ kind: 'detector', id });
  }

  emitDatasetCreated(id: string): void {
    this.createdSubject.next({ kind: 'dataset', id });
  }
}
