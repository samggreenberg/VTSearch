import { signal } from '@angular/core';
import { Observable } from 'rxjs';

import { FieldOptions, ImporterField } from '../models/api.models';
import { apiErrorMessage } from './api-error';

/** Fetches the current option list for one ``dynamic_options`` plugin field.
 *  Supplied by the owning component so this helper stays independent of which
 *  importer API (label importers, datasource importers, …) backs the form. */
export type FieldOptionsFetcher = (
  fieldKey: string,
  values: Record<string, string>,
) => Observable<{ options: FieldOptions[] }>;

/** Option-list state for a plugin-field form whose ``dynamic_options`` selects
 *  are populated by a server round-trip.
 *
 *  Every dispatch stamps a monotonically-increasing token onto the field key
 *  and the response handlers bail unless that stamp is still the newest one,
 *  which is what makes a late response harmless in the two ways it can be
 *  stale: a `depends_on` free-text field re-fires on every keystroke, so two
 *  requests for the same key can resolve out of order; and switching to a
 *  different importer that happens to share a field key (``path``,
 *  ``dataset``, …) would otherwise let the previous importer's late response
 *  populate the new form's dropdown and auto-select a value invalid for it.
 *  :meth:`reset` drops the stamps outright, so everything in flight when the
 *  form is re-seeded is discarded.
 *
 *  The three maps are signals so the unpatched HTTP callbacks schedule change
 *  detection under zoneless. */
export class DynamicFieldOptions {
  /** Fetched options, keyed by field key. */
  readonly options = signal<Record<string, FieldOptions[]>>({});
  /** Whether a fetch is in flight, keyed by field key. */
  readonly loading = signal<Record<string, boolean>>({});
  /** Inline fetch-failure message, keyed by field key. */
  readonly error = signal<Record<string, string>>({});

  /** Token of the newest dispatched request per field key.  A response whose
   *  token no longer matches has been superseded and is dropped. */
  private tokens: Record<string, number> = {};
  private nextToken = 0;

  /** @param fetcher  Issues the options request for one field.
   *  @param onApplied  Optional hook run after a fresh (non-superseded)
   *    response has been applied, for owners that need to react to the
   *    auto-selected value. */
  constructor(
    private readonly fetcher: FieldOptionsFetcher,
    private readonly onApplied?: () => void,
  ) {}

  /** Clear all fetched state and invalidate every in-flight request.  Call
   *  when the form is re-seeded for a different importer, or abandoned. */
  reset(): void {
    this.tokens = {};
    this.options.set({});
    this.loading.set({});
    this.error.set({});
  }

  /** Options to render for a field: the dynamically-fetched list for a
   *  ``dynamic_options`` field, else the static strings coerced into
   *  ``{value,label}`` pairs. */
  optionsFor(field: ImporterField): FieldOptions[] {
    if (field.dynamic_options) {
      return this.options()[field.key] || [];
    }
    return (field.options || []).map((o) => ({ value: o, label: o }));
  }

  /** Fetch options for every ``dynamic_options`` field of a freshly-selected
   *  importer. */
  refreshAll(fields: ImporterField[], values: Record<string, string>): void {
    for (const field of fields) {
      if (field.dynamic_options) {
        this.refresh(field, values);
      }
    }
  }

  /** Re-fetch options for any field that depends on the one the user just
   *  changed, blanking each dependent value first so a stale selection can't
   *  survive into the new option set. */
  refreshDependentsOf(
    changedKey: string,
    fields: ImporterField[],
    values: Record<string, string>,
  ): void {
    for (const field of fields) {
      if (!field.dynamic_options) continue;
      if (!(field.depends_on || []).includes(changedKey)) continue;
      values[field.key] = '';
      this.refresh(field, values);
    }
  }

  /** Fetch one field's options, applying the result only if no newer request
   *  for the same key (and no :meth:`reset`) has intervened. */
  refresh(field: ImporterField, values: Record<string, string>): void {
    const key = field.key;
    const token = ++this.nextToken;
    this.tokens[key] = token;
    this.loading.update((m) => ({ ...m, [key]: true }));
    this.error.update((m) => ({ ...m, [key]: '' }));
    this.fetcher(key, { ...values }).subscribe({
      next: (res) => {
        if (this.tokens[key] !== token) return;
        const options: FieldOptions[] = res.options || [];
        this.options.update((m) => ({ ...m, [key]: options }));
        this.loading.update((m) => ({ ...m, [key]: false }));
        const current = values[key];
        const inList = options.some((o) => o.value === String(current));
        // Strict selects clear a value the new list no longer offers; a
        // free-text combobox keeps whatever the user typed.
        if (current && !inList && !field.allow_free_text) {
          values[key] = '';
        }
        if (!values[key] && field.required && !field.allow_free_text && options.length > 0) {
          values[key] = options[0].value;
        }
        this.onApplied?.();
      },
      error: (err) => {
        if (this.tokens[key] !== token) return;
        this.loading.update((m) => ({ ...m, [key]: false }));
        this.error.update((m) => ({
          ...m,
          [key]: apiErrorMessage(err, 'Could not load options'),
        }));
        this.options.update((m) => ({ ...m, [key]: [] }));
      },
    });
  }
}
