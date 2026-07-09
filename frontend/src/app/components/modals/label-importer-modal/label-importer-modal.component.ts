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
  ViewChild,
} from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../../modal/modal.component';
import { IconComponent } from '../../icon/icon.component';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import { LabelImportersApiService } from '../../../services/label-importers-api.service';
import { MediasApiService } from '../../../services/medias-api.service';
import { VoteStateService } from '../../../services/vote-state.service';
import { ImporterField } from '../../../models/api.models';
import type { LabelImporterEntry } from '../../../generated/api-client/models/label-importer-entry';
import { apiErrorMessage } from '../../../utils/api-error';

type ModalView = 'picker' | 'form';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-label-importer-modal',
  standalone: true,
  imports: [FormsModule, ModalComponent, IconComponent, FieldHintIconComponent],
  templateUrl: './label-importer-modal.component.html',
  styleUrl: './label-importer-modal.component.scss',
})
export class LabelImporterModalComponent implements OnDestroy {
  private labelImportersApi = inject(LabelImportersApiService);
  private mediasApi = inject(MediasApiService);
  private voteState = inject(VoteStateService);

  /** When set, labels are imported directly into this trainable model
   *  instead of into the active dataset's vote state. */
  readonly targetModelName = input<string | null>(null);

  readonly closed = output<void>();
  readonly imported = output<void>();

  @ViewChild('addGoodInput') addGoodInput!: ElementRef<HTMLInputElement>;
  @ViewChild('addBadInput') addBadInput!: ElementRef<HTMLInputElement>;

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
  // schedule CD when they land. See docs/plans/zoneless-migration.md.
  readonly submitting = signal(false);

  /** Error from a failed import action; the list-load failure is merged in. */
  private readonly importError = signal('');
  readonly error = computed(
    () =>
      this.importError() ||
      (this.importersResource.error() ? 'Failed to load label importers' : ''),
  );
  readonly successMessage = signal('');
  readonly dynamicFieldOptions = signal<Record<string, string[]>>({});
  readonly dynamicFieldLoading = signal<Record<string, boolean>>({});
  readonly dynamicFieldError = signal<Record<string, string>>({});
  readonly addingGood = signal(false);
  readonly addingBad = signal(false);
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
    this.dynamicFieldOptions.set({});
    this.dynamicFieldLoading.set({});
    this.dynamicFieldError.set({});
    const fields = (importer.fields ?? []) as ImporterField[];
    for (const field of fields) {
      if (field.default) {
        this.formValues[field.key] = field.default;
      } else if (
        field.field_type === 'select' &&
        !field.dynamic_options &&
        (field.options?.length ?? 0) > 0
      ) {
        this.formValues[field.key] = field.options![0];
      }
    }
    for (const field of fields) {
      if (field.dynamic_options) {
        this.refreshDynamicFieldOptions(field);
      }
    }
    this.view = 'form';
  }

  onFormFieldChanged(changedKey: string): void {
    const importer = this.selectedImporter;
    if (!importer?.fields) return;
    for (const field of (importer.fields as ImporterField[])) {
      if (!field.dynamic_options) continue;
      if (!(field.depends_on || []).includes(changedKey)) continue;
      this.formValues[field.key] = '';
      this.refreshDynamicFieldOptions(field);
    }
  }

  optionsFor(field: ImporterField): string[] {
    if (field.dynamic_options) {
      return this.dynamicFieldOptions()[field.key] || [];
    }
    return field.options || [];
  }

  private refreshDynamicFieldOptions(field: ImporterField): void {
    const importer = this.selectedImporter;
    if (!importer) return;
    const key = field.key;
    this.dynamicFieldLoading.update((m) => ({ ...m, [key]: true }));
    this.dynamicFieldError.update((m) => ({ ...m, [key]: '' }));
    this.labelImportersApi
      .getFieldOptions(importer.name, key, { ...this.formValues })
      .subscribe({
        next: (res) => {
          const options = res.options || [];
          this.dynamicFieldOptions.update((m) => ({ ...m, [key]: options }));
          this.dynamicFieldLoading.update((m) => ({ ...m, [key]: false }));
          const current = this.formValues[key];
          if (current && !options.includes(String(current))) {
            this.formValues[key] = '';
          }
          if (!this.formValues[key] && field.required && options.length > 0) {
            this.formValues[key] = options[0];
          }
        },
        error: (err) => {
          this.dynamicFieldLoading.update((m) => ({ ...m, [key]: false }));
          this.dynamicFieldError.update((m) => ({
            ...m,
            [key]: apiErrorMessage(err, 'Could not load options'),
          }));
          this.dynamicFieldOptions.update((m) => ({ ...m, [key]: [] }));
        },
      });
  }

  back(): void {
    this.view = 'picker';
    this.selectedImporter = null;
    this.importError.set('');
    this.successMessage.set('');
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
        this.submitting.set(false);
        this.successMessage.set(res.message || `Applied ${res.applied ?? 0} labels`);
        this.imported.emit();

        if (res.missing_count > 0 && res.missing?.length) {
          // Show unresolved elements as a warning but don't prompt; the
          // backend already tried to auto-resolve them.
          this.importError.set(
            `${res.missing_count} element(s) could not be resolved from their original sources.`,
          );
        } else if (res.failed_count > 0) {
          // Per-entry application failures (logical-bug-audit H31); the
          // import partially landed and the user should know which entries
          // need to be retried.
          this.importError.set(
            `${res.failed_count} element(s) failed to apply. The remaining labels were applied successfully.`,
          );
        }
        const hasIssue = res.missing_count > 0 || res.failed_count > 0;
        this.closeTimer = setTimeout(() => this.close(), hasIssue ? 3000 : 1500);
      },
      error: (err) => {
        this.submitting.set(false);
        this.importError.set(apiErrorMessage(err, 'Import failed'));
      },
    });
  }


  triggerAddGood(): void {
    this.addGoodInput.nativeElement.click();
  }

  triggerAddBad(): void {
    this.addBadInput.nativeElement.click();
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
  }
}
