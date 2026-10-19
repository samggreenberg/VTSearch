import { ChangeDetectionStrategy, Component, DestroyRef, inject, input, OnInit, output, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DuplicatesModalComponent } from '../duplicates-modal/duplicates-modal.component';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import { ProgressEventsService } from '../../../services/progress-events.service';
import type { LoadingTask } from '../../../models/api.models';
import type { DatasetRegistryStatsResponse } from '../../../generated/api-client/models/dataset-registry-stats-response';
import { formatTimestamp as formatTs } from '../../../utils/format-date';
import { apiErrorMessage } from '../../../utils/api-error';
import { isSoloMediaType } from '../../../utils/solo-media-type';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-dataset-stats-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent, DuplicatesModalComponent],
  templateUrl: './dataset-stats-modal.component.html',
  styleUrl: './dataset-stats-modal.component.scss',
})
export class DatasetStatsModalComponent implements OnInit {
  private datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private progressEvents = inject(ProgressEventsService);
  private destroyRef = inject(DestroyRef);

  readonly datasetId = input('');
  readonly datasetName = input('');
  /** Mirrors the Dashboard grid's own column gating so the Stats window
   *  hides exactly what the grid hides: Creator/Readers are noise on a
   *  single-user (default-login) install. */
  readonly isDefaultLogin = input(true);
  /** True when the server stamps datasets with an age-off. Gates the
   *  Age-Off row the same way the grid gates its Age-Off column — except
   *  a dataset that already carries an expiry always shows it. */
  readonly serverSetsAgeOff = input(false);
  /** The server's one media type when it is locked to one. Gates the Type row
   *  as the grid gates its Type column; see {@link showMediaType}. */
  readonly soloMediaType = input<string | null>(null);
  readonly closed = output<void>();

  // Signalized so the `ngOnInit` subscribe (an unpatched callback under zoneless)
  // schedules CD when the stats land.
  readonly loading = signal(true);
  readonly error = signal('');
  readonly stats = signal<DatasetRegistryStatsResponse | null>(null);

  /** Child Duplicates modal (issue #2697), opened from the Duplicate-groups row. */
  readonly showDuplicates = signal(false);

  /** A coverage-atlas build is on (#4995): from the Build click, or stats
   *  that report one running, until a re-read of the stats says it ended.
   *  Disables the Build button. */
  readonly atlasBuilding = signal(false);
  /** The Build POST has not answered: until it does, a task channel with
   *  nothing running on this dataset means "not started", not "finished". */
  private atlasRequestPending = false;
  private statsRereadPending = false;
  private latestTasks: LoadingTask[] = [];

  ngOnInit(): void {
    this.progressEvents.loadingTasks$.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((tasks) => {
      this.latestTasks = tasks;
      this.checkAtlasBuild();
    });
    this.datasetsRegistryApi.getDatasetStats(this.datasetId()).subscribe({
      next: (data) => {
        this.applyStats(data);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(apiErrorMessage(err, 'Failed to load stats'));
        this.loading.set(false);
      },
    });
  }

  close(): void {
    this.closed.emit();
  }

  private applyStats(data: DatasetRegistryStatsResponse): void {
    this.stats.set(data);
    this.atlasBuilding.set(!!data.coverage_atlas?.building);
  }

  /** The Coverage atlas row shows only for a loaded dataset without an atlas:
   *  one past the auto-build threshold, or whose build was cancelled. */
  get showAtlasRow(): boolean {
    const atlas = this.stats()?.coverage_atlas;
    return !!atlas && !atlas.built;
  }

  /** The item count past which loading skips the build, when this dataset is
   *  past it (the reason it has no atlas); `null` otherwise. */
  get atlasSkipThreshold(): number | null {
    const s = this.stats();
    const threshold = s?.coverage_atlas?.auto_build_threshold;
    return s && threshold != null && s.num_items > threshold ? threshold : null;
  }

  /** Start the build. Its progress shows on the dataset's Dashboard row, as
   *  any loading task's does; this window only waits for it to end. */
  buildAtlas(): void {
    if (this.atlasBuilding()) return;
    this.atlasBuilding.set(true);
    this.atlasRequestPending = true;
    this.datasetsRegistryApi.buildCoverageAtlas(this.datasetId()).subscribe({
      next: () => {
        this.atlasRequestPending = false;
        this.checkAtlasBuild();
      },
      // The error interceptor toasts the server's reason (say, the dataset
      // was unloaded meanwhile); the button comes back for a retry.
      error: () => {
        this.atlasRequestPending = false;
        this.atlasBuilding.set(false);
      },
    });
  }

  /** While a build is on, re-read the stats whenever no task keyed to this
   *  dataset is running: the build's row has ended, or has not shown up yet.
   *  The stats decide which, so a build still running keeps the button
   *  disabled until the next snapshot. */
  private checkAtlasBuild(): void {
    if (!this.atlasBuilding() || this.atlasRequestPending || this.statsRereadPending) return;
    const id = this.datasetId();
    if (this.latestTasks.some((t) => t.dataset_id === id && t.status !== 'idle')) return;
    this.statsRereadPending = true;
    this.datasetsRegistryApi.getDatasetStats(id).subscribe({
      next: (data) => {
        this.statsRereadPending = false;
        this.applyStats(data);
      },
      error: () => {
        this.statsRereadPending = false;
        this.atlasBuilding.set(false);
      },
    });
  }

  get fileTypes(): { ext: string; label: string; count: number }[] {
    const counts = this.stats()?.file_type_counts;
    if (!counts) return [];
    return Object.entries(counts)
      .sort(([, a], [, b]) => b - a)
      .map(([ext, count]) => ({ ext, label: this.fileTypeLabel(ext), count }));
  }

  /** Render a file-type bucket. Real types get the leading dot users expect
   *  (`jpg` → `.jpg`); the server's parenthesised sentinel for items whose
   *  type nothing could establish is passed through as-is, so the row reads
   *  "(unknown)" and not ".(unknown)". */
  private fileTypeLabel(ext: string): string {
    return ext.startsWith('(') ? ext : `.${ext}`;
  }

  /** Whether the Type row shows: not on a server locked to the dataset's
   *  type, where it could be nothing else (#5106). */
  get showMediaType(): boolean {
    return !isSoloMediaType(this.soloMediaType(), this.stats()?.media_type);
  }

  /** Media type as the grid renders it: capitalized, `-` when unset. */
  get mediaType(): string {
    const t = this.stats()?.media_type;
    if (!t) return '-';
    return t.charAt(0).toUpperCase() + t.slice(1);
  }

  /** Age-off date, or the grid's "Never" when the dataset has no expiry. */
  get ageOff(): string {
    const expires = this.stats()?.expires_at;
    return expires != null ? formatTs(expires) : 'Never';
  }

  /** Show the Age-Off row when the server stamps age-offs, or whenever this
   *  particular dataset already carries one (a stamp survives the setting
   *  being turned back off, and hiding a real death date would be a lie). */
  get showAgeOff(): boolean {
    return this.serverSetsAgeOff() || this.stats()?.expires_at != null;
  }

  /** Creator/Readers are meaningless on a single-user install, exactly as
   *  in the grid, which drops both columns under the default login. */
  get showAccess(): boolean {
    return !this.isDefaultLogin();
  }

  get readers(): string {
    const list = this.stats()?.readers ?? [];
    return list.length ? list.join(', ') : '-';
  }

  get importerName(): string {
    const src = this.stats()?.source as { importer?: string } | undefined;
    return src?.importer || '';
  }

  get originParams(): { key: string; label: string; value: string }[] {
    const src = this.stats()?.source as { params?: Record<string, unknown> } | undefined;
    const params = src?.params;
    if (!params) return [];
    return Object.entries(params)
      .filter(([, v]) => v !== '' && v != null)
      .map(([key, value]) => ({ key, label: this.formatParamKey(key), value: String(value) }));
  }

  /** Render a raw origin-param key (e.g. ``media_type``, ``name``) as a label
   *  that matches the hardcoded stat labels: underscores become spaces and the
   *  first letter is capitalized, so a demo dataset's ``name`` param reads
   *  "Name" alongside "Importer" / "Clipper" / "Embedder" instead of lowercase. */
  private formatParamKey(key: string): string {
    const spaced = key.replace(/_/g, ' ');
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
  }

  formatTimestamp(ts: number | null): string {
    return formatTs(ts);
  }

  get duration(): string {
    const s = this.stats();
    if (!s?.ingest_started_at || !s?.ingest_finished_at) return '-';
    const seconds = Math.round(s.ingest_finished_at - s.ingest_started_at);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (minutes < 60) return `${minutes}m ${secs}s`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m ${secs}s`;
  }
}
