import {
  ChangeDetectionStrategy,
  Component,
  computed,
  ElementRef,
  inject,
  input,
  OnDestroy,
  output,
  signal,
  viewChild,
} from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import { Subscription } from 'rxjs';
import { LabelImportersApiService } from '../../../services/label-importers-api.service';
import { MediasApiService } from '../../../services/medias-api.service';
import { ProgressEventsService } from '../../../services/progress-events.service';
import { VoteStateService } from '../../../services/vote-state.service';
import { ImporterField, LoadingTask } from '../../../models/api.models';
import type { LabelImporterEntry } from '../../../generated/api-client/models/label-importer-entry';
import { apiErrorMessage } from '../../../utils/api-error';
import { DynamicFieldOptions } from '../../../utils/dynamic-field-options';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';
import { formatProgressMessage, progressBarState, type ProgressBarState } from '../../../utils/format-progress';

type ModalView = 'picker' | 'form';

/** Terminal counts an auto-resolve ingest task publishes as `ingest_result`
 *  (see `vtsearch/routes/labels/importers.py::_apply_ingested_labels`). */
interface IngestResult {
  ingested?: number;
  applied?: number;
  unresolved?: number;
  failed?: number;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-label-importer-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, IconComponent, FieldHintIconComponent, ProgressBarComponent],
  templateUrl: './label-importer-modal.component.html',
  styleUrl: './label-importer-modal.component.scss',
})
export class LabelImporterModalComponent implements OnDestroy {
  private labelImportersApi = inject(LabelImportersApiService);
  private mediasApi = inject(MediasApiService);
  private progressEvents = inject(ProgressEventsService);
  private voteState = inject(VoteStateService);

  /** When set, labels are imported directly into this trainable model
   *  instead of into the active dataset's vote state. */
  readonly targetModelName = input<string | null>(null);

  readonly closed = output<void>();
  readonly imported = output<void>();

  // Optional queries: both file inputs live behind `@if (!targetModelName())`,
  // so they resolve to `undefined` when a target model is set.
  readonly addGoodInput = viewChild<ElementRef<HTMLInputElement>>('addGoodInput');
  readonly addBadInput = viewChild<ElementRef<HTMLInputElement>>('addBadInput');

  view: ModalView = 'picker';

  // Eager `rxResource`: loads the label-importer list once on creation, wrapping
  // the generated-client read so the interceptor chain still applies. Mirrors the
  // settings-importer modal; replaces the old `ngOnInit` subscribe so the list
  // commits via a signal (schedules CD under zoneless).
  private readonly importersResource = rxResource({
    stream: () => this.labelImportersApi.list(),
  });
  readonly importers = computed<LabelImporterEntry[]>(() =>
    (this.importersResource.value() ?? []).filter((imp) => !imp.hidden_from_picker),
  );
  readonly loading = computed(() => this.importersResource.isLoading());

  selectedImporter: LabelImporterEntry | null = null;
  formValues: Record<string, string> = {};
  selectedFile: File | null = null;
  selectedFileFieldKey: string | null = null;
  // Mutation-result state, signalized so the submit/add-to-pile subscribes (and
  // the dynamic-field-option fetches) — all unpatched callbacks under zoneless —
  // schedule CD when they land.
  readonly submitting = signal(false);

  /** Error from a failed import action; the list-load failure is merged in. */
  private readonly importError = signal('');
  readonly error = computed(
    () =>
      this.importError() ||
      (this.importersResource.error() ? 'Failed to load label importers' : ''),
  );
  readonly successMessage = signal('');
  /** Option lists for the selected importer's ``dynamic_options`` fields. */
  readonly fieldOptions = new DynamicFieldOptions((key, values) =>
    this.labelImportersApi.getFieldOptions(this.selectedImporter!.name, key, values),
  );
  readonly addingGood = signal(false);
  readonly addingBad = signal(false);

  /** Live snapshot of the background task that pulls in the media of labels
   *  the active dataset didn't have. ``null`` when no ingest is running. */
  readonly ingestTask = signal<LoadingTask | null>(null);
  private ingestSub: Subscription | null = null;
  private closeTimer: ReturnType<typeof setTimeout> | null = null;

  get modalTitle(): string {
    if (this.view === 'form' && this.selectedImporter) {
      return this.selectedImporter.display_name || this.selectedImporter.name;
    }
    const targetModelName = this.targetModelName();
    return targetModelName ? `Import Labels into ${targetModelName}` : 'Import Labels';
  }

  /** Typed view of the selected importer's plugin fields for the template
   *  (the generated LabelImporterEntry types `fields` as an open dict
   *  because plugin field schemas aren't part of the OpenAPI client). */
  get selectedImporterFields(): ImporterField[] {
    return (this.selectedImporter?.fields ?? []) as ImporterField[];
  }

  selectImporter(importer: LabelImporterEntry): void {
    this.selectedImporter = importer;
    this.formValues = {};
    this.selectedFile = null;
    this.selectedFileFieldKey = null;
    this.importError.set('');
    this.successMessage.set('');
    this.fieldOptions.reset();
    const fields = (importer.fields ?? []) as ImporterField[];
    for (const field of fields) {
      if (field.default) {
        this.formValues[field.key] = field.default;
      } else if (
        field.field_type === 'select' &&
        !field.dynamic_options &&
        !field.allow_free_text &&
        (field.options?.length ?? 0) > 0
      ) {
        this.formValues[field.key] = field.options![0];
      }
    }
    this.fieldOptions.refreshAll(fields, this.formValues);
    this.view = 'form';
  }

  onFormFieldChanged(changedKey: string): void {
    if (!this.selectedImporter?.fields) return;
    this.fieldOptions.refreshDependentsOf(changedKey, this.selectedImporterFields, this.formValues);
  }

  back(): void {
    this.view = 'picker';
    this.selectedImporter = null;
    this.importError.set('');
    this.successMessage.set('');
    this.fieldOptions.reset();
  }

  onFileSelected(event: Event, fieldName: string): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.selectedFileFieldKey = fieldName;
      this.formValues[fieldName] = input.files[0].name;
    }
  }

  submit(): void {
    if (!this.selectedImporter) return;
    this.submitting.set(true);
    this.importError.set('');
    this.successMessage.set('');

    const targetModelName = this.targetModelName();
    const request$ = targetModelName
      ? this.labelImportersApi.runModelImport(
          targetModelName,
          this.selectedImporter.name,
          this.formValues,
          this.selectedFile ?? undefined,
          this.selectedFileFieldKey ?? undefined,
        )
      : this.labelImportersApi.runImport(
          this.selectedImporter.name,
          this.formValues,
          this.selectedFile ?? undefined,
          this.selectedFileFieldKey ?? undefined,
        );

    request$.subscribe({
      next: (res: any) => {
        this.successMessage.set(res.message || `Applied ${res.applied ?? 0} labels`);
        // The labels that already resolved are live now, so refresh the grid
        // even while the auto-resolve pass is still fetching the rest.
        this.imported.emit();

        const ingestTaskId = String(res?.ingest_task_id || '');
        if (ingestTaskId) {
          this.awaitAutoResolve(ingestTaskId, Number(res.applied ?? 0));
          return;
        }
        this.submitting.set(false);
        this.finishImport(
          res.message || `Applied ${res.applied ?? 0} labels`,
          res.missing_count ?? 0,
          res.failed_count ?? 0,
        );
      },
      error: (err) => {
        this.submitting.set(false);
        this.importError.set(apiErrorMessage(err, 'Import failed'));
      },
    });
  }

  /**
   * Track the background auto-resolve of entries whose media weren't in the
   * dataset (#2703) and report its outcome once it lands.
   *
   * The import response only describes the in-request pass; fetching and
   * embedding the missing media happens on a detector task, which publishes
   * the rest of the numbers as `ingest_result`. `appliedInRequest` is what the
   * first pass already applied, so the final line can report the total.
   */
  private awaitAutoResolve(taskId: string, appliedInRequest: number): void {
    this.ingestSub?.unsubscribe();
    this.ingestSub = this.progressEvents.detectorTaskUntilDone$(taskId).subscribe({
      next: (task) => this.ingestTask.set(task),
      complete: () => {
        const task = this.ingestTask();
        this.ingestTask.set(null);
        this.ingestSub = null;
        this.submitting.set(false);

        if (task?.error) {
          this.importError.set(`Could not resolve the missing elements: ${task.error}`);
          this.finishImport(`Applied ${appliedInRequest} label(s).`, 0, 0);
          return;
        }
        const result = (task?.ingest_result ?? {}) as IngestResult;
        const resolved = result.ingested ?? 0;
        let message = `Applied ${appliedInRequest + (result.applied ?? 0)} label(s).`;
        if (resolved > 0) {
          message += ` Auto-resolved ${resolved} missing element(s) from their sources.`;
          // The re-applied labels are new votes; refresh so they show up.
          this.imported.emit();
        }
        this.finishImport(message, result.unresolved ?? 0, result.failed ?? 0);
      },
    });
  }

  /** Render the terminal outcome and schedule the auto-close. */
  private finishImport(message: string, missingCount: number, failedCount: number): void {
    this.successMessage.set(message);
    if (missingCount > 0) {
      // Show unresolved elements as a warning but don't prompt; the
      // backend already tried to auto-resolve them.
      this.importError.set(
        `${missingCount} element(s) could not be resolved from their original sources.`,
      );
    } else if (failedCount > 0) {
      // Per-entry application failures (logical-bug-audit H31); the
      // import partially landed and the user should know which entries
      // need to be retried.
      this.importError.set(
        `${failedCount} element(s) failed to apply. The remaining labels were applied successfully.`,
      );
    }
    const hasIssue = missingCount > 0 || failedCount > 0;
    this.closeTimer = setTimeout(() => this.close(), hasIssue ? 3000 : 1500);
  }

  /** Bar geometry for the running auto-resolve ingest. */
  get ingestBar(): ProgressBarState {
    return progressBarState(this.ingestTask());
  }

  /** One-line status for the running auto-resolve ingest. */
  get ingestMessage(): string {
    return formatProgressMessage(this.ingestTask(), 'Resolving missing elements…');
  }


  triggerAddGood(): void {
    this.addGoodInput()?.nativeElement.click();
  }

  triggerAddBad(): void {
    this.addBadInput()?.nativeElement.click();
  }

  onAddToPile(event: Event, label: 'good' | 'bad'): void {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;
    const file = input.files[0];
    input.value = '';

    if (label === 'good') {
      this.addingGood.set(true);
    } else {
      this.addingBad.set(true);
    }
    this.importError.set('');
    this.successMessage.set('');

    this.mediasApi.addToPile(file, label).subscribe({
      next: (result) => {
        const action = result.is_new ? 'Added new media' : 'Matched existing media';
        this.successMessage.set(`${action} to ${label} pile.`);
        this.voteState.loadVotes();
        this.imported.emit();
        if (label === 'good') {
          this.addingGood.set(false);
        } else {
          this.addingBad.set(false);
        }
      },
      error: (err) => {
        this.importError.set(apiErrorMessage(err, `Failed to add media to ${label} pile`));
        if (label === 'good') {
          this.addingGood.set(false);
        } else {
          this.addingBad.set(false);
        }
      },
    });
  }

  close(): void {
    if (this.closeTimer !== null) {
      clearTimeout(this.closeTimer);
      this.closeTimer = null;
    }
    this.closed.emit();
  }

  ngOnDestroy(): void {
    if (this.closeTimer !== null) {
      clearTimeout(this.closeTimer);
      this.closeTimer = null;
    }
    // The ingest itself keeps running on the server (and reports on the
    // dashboard); we just stop rendering it.
    this.ingestSub?.unsubscribe();
    this.ingestSub = null;
  }
}
