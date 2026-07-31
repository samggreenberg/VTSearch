import { ChangeDetectionStrategy, Component, inject, input, OnInit, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DuplicatesModalComponent } from '../duplicates-modal/duplicates-modal.component';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import type { DatasetRegistryStatsResponse } from '../../../generated/api-client/models/dataset-registry-stats-response';
import { formatTimestamp as formatTs } from '../../../utils/format-date';
import { apiErrorMessage } from '../../../utils/api-error';

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
  readonly closed = output<void>();

  // Signalized so the `ngOnInit` subscribe (an unpatched callback under zoneless)
  // schedules CD when the stats land. See docs/plans/zoneless-migration.md.
  readonly loading = signal(true);
  readonly error = signal('');
  readonly stats = signal<DatasetRegistryStatsResponse | null>(null);

  /** Child Duplicates modal (issue #2697), opened from the Duplicate-groups row. */
  readonly showDuplicates = signal(false);

  ngOnInit(): void {
    this.datasetsRegistryApi.getDatasetStats(this.datasetId()).subscribe({
      next: (data) => {
        this.stats.set(data);
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
