import { ChangeDetectionStrategy, Component, computed, inject, input, OnInit, output, signal } from '@angular/core';
import { forkJoin } from 'rxjs';

import { ModalComponent } from '../../modal/modal.component';
import {
  fieldsComplete,
  LabelsetSourceFieldsComponent,
} from '../../labelset-source-fields/labelset-source-fields.component';
import { LabelsetSourcesApiService } from '../../../services/labelset-sources-api.service';
import { ToastService } from '../../../services/toast.service';
import { ImporterField } from '../../../models/api.models';
import { visibleFields } from '../../../utils/plugin-fields';
import { apiErrorMessage } from '../../../utils/api-error';
import type { DetectorLabelsetSource } from '../../../generated/api-client/models/detector-labelset-source';

/**
 * Where one detector's labels are auto-exported (#4913), opened from its
 * Dashboard row.
 *
 * The detector syncs to a labelset source - usually the deployment's
 * default, whose admin-fixed values never leave the server - and this edits
 * the values the user supplies: the source's fields minus the fixed ones.
 * It is how a detector created before the admin turned auto-export on, or one
 * whose values were refused, gets set up. Saving links the detector, which
 * also restores anything the source already held for it.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-auto-export-modal',
  standalone: true,
  imports: [ModalComponent, LabelsetSourceFieldsComponent],
  templateUrl: './auto-export-modal.component.html',
})
export class AutoExportModalComponent implements OnInit {
  private labelsetSourcesApi = inject(LabelsetSourcesApiService);
  private toast = inject(ToastService);

  readonly detectorId = input('');
  readonly detectorName = input('');
  readonly closed = output<void>();
  readonly saved = output<void>();

  readonly loading = signal(true);
  readonly saving = signal(false);
  readonly error = signal('');
  /** The detector's current link, or null when it syncs nowhere. */
  readonly binding = signal<DetectorLabelsetSource | null>(null);
  /** The fields the user fills in: the source's, minus the admin-fixed ones. */
  readonly fields = signal<ImporterField[]>([]);
  readonly values = signal<Record<string, string>>({});
  readonly ready = computed(() => fieldsComplete(this.fields(), this.values()));

  ngOnInit(): void {
    forkJoin({
      binding: this.labelsetSourcesApi.getDetectorSource(this.detectorId()),
      sources: this.labelsetSourcesApi.listSources(),
    }).subscribe({
      next: ({ binding, sources }) => {
        this.binding.set(binding);
        if (binding) {
          const source = sources.find((s) => s.name === binding.source_name);
          const fixed = new Set(binding.fixed_keys ?? []);
          const fields = visibleFields((source?.fields ?? []) as ImporterField[]).filter((f) => !fixed.has(f.key));
          this.fields.set(fields);
          const own = (binding.field_values ?? {}) as Record<string, unknown>;
          const values: Record<string, string> = {};
          for (const f of fields) {
            const v = own[f.key];
            values[f.key] = v === undefined || v === null ? f.default ?? '' : String(v);
          }
          this.values.set(values);
        }
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(apiErrorMessage(err, 'Failed to load the auto-export settings'));
        this.loading.set(false);
      },
    });
  }

  save(): void {
    const binding = this.binding();
    if (!binding || !this.ready() || this.saving()) return;
    this.saving.set(true);
    this.error.set('');
    this.labelsetSourcesApi.setDetectorSource(this.detectorId(), binding.source_name, this.values()).subscribe({
      next: (res) => {
        this.saving.set(false);
        this.toast.success({ message: res.message || 'Auto-export saved' });
        this.saved.emit();
      },
      error: (err) => {
        this.saving.set(false);
        this.error.set(apiErrorMessage(err, 'Failed to save the auto-export settings'));
      },
    });
  }

  close(): void {
    this.closed.emit();
  }
}
