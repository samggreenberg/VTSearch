import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ClipperInfo, ConverterInfo, SourceSpec } from '../../../../models/api.models';

/** Checkbox column for choosing which source media types feed a
 *  multi-media import.  The native type sits at the top (always
 *  included by default, no converter); each other source type with at
 *  least one converter to the native type appears as a checkbox with a
 *  "Details ▸" opener for picking among converters (when there are
 *  multiple) and editing their params.  The native row also gets its
 *  own "Details ▸" button when there is more than one MediaClipper to
 *  pick between — clicking it asks the parent to open the shared
 *  clipper-chooser modal.  The picker itself lives inside the outer
 *  importer's "Advanced" section, so a nested "Advanced" label here
 *  would read as "Advanced inside Advanced". */
@Component({
  selector: 'vt-source-specs-picker',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './source-specs-picker.component.html',
  styleUrl: './source-specs-picker.component.scss',
})
export class SourceSpecsPickerComponent implements OnChanges {
  /** All converters whose ``target_type`` matches the current native
   *  type.  Comes from the importer's ``to_dict()`` payload. */
  @Input() availableConverters: ConverterInfo[] = [];
  /** Native media type id (e.g. ``"image"``) — the dataset's output
   *  type.  The native checkbox represents "include direct files of
   *  this type, no conversion". */
  @Input() nativeType = '';
  /** Two-way bound source-spec list submitted to the importer. */
  @Input() specs: SourceSpec[] = [];
  /** Map of type_id → human-readable label.  Falls back to the type_id
   *  when a label is missing. */
  @Input() typeLabels: Record<string, string> = {};
  /** Clippers available for the native media type.  Drives the native
   *  row's "Details" button — shown only when there is more than one
   *  clipper to pick between (same gate as the legacy standalone
   *  Clipper section).  An empty list means "no clipper choice", and
   *  the Details button is suppressed. */
  @Input() clippers: ClipperInfo[] = [];
  /** Name of the clipper currently selected by the parent for the
   *  native row.  Used to look up the matching :class:`ClipperInfo`
   *  for the inline summary preview. */
  @Input() selectedClipperName = '';
  /** Current parameter values for the selected native clipper, keyed by
   *  the clipper's parameter ``key``.  Substituted into the clipper's
   *  ``summary_template`` to render the inline preview. */
  @Input() selectedClipperParams: Record<string, string | number> = {};
  @Output() specsChange = new EventEmitter<SourceSpec[]>();
  /** Fired when the user clicks the native row's "Details" button.
   *  The parent opens the shared clipper-chooser modal in response. */
  @Output() clipperChooserRequested = new EventEmitter<void>();

  /** Per-source-type draft so the user can configure a converter via
   *  the Details panel *before* checking the box.  Edits made while
   *  unchecked are preserved here and applied on check. */
  private drafts = new Map<string, SourceSpec>();
  private detailsOpenSet = new Set<string>();

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['nativeType'] && !changes['nativeType'].firstChange) {
      this.drafts.clear();
      this.detailsOpenSet.clear();
    }
    for (const s of this.specs) {
      if (s.converter !== null) this.drafts.set(s.source_type, s);
    }
  }

  get nonNativeSourceTypes(): string[] {
    const set = new Set<string>();
    for (const c of this.availableConverters) {
      if (c.source_type !== this.nativeType) set.add(c.source_type);
    }
    return Array.from(set).sort();
  }

  labelFor(sourceType: string): string {
    return this.typeLabels[sourceType] || sourceType;
  }

  isNonNativeChecked(sourceType: string): boolean {
    return this.specs.some((s) => s.source_type === sourceType && s.converter !== null);
  }

  convertersFor(sourceType: string): ConverterInfo[] {
    return this.availableConverters.filter((c) => c.source_type === sourceType);
  }

  /** True iff the Details opener should be rendered for *sourceType*:
   *  either multiple converters to pick between, or at least one
   *  user-editable field on the (single) converter. */
  hasDetails(sourceType: string): boolean {
    const cs = this.convertersFor(sourceType);
    if (cs.length > 1) return true;
    return !!(cs[0]?.fields?.length);
  }

  isDetailsOpen(sourceType: string): boolean {
    return this.detailsOpenSet.has(sourceType);
  }

  toggleDetails(sourceType: string): void {
    if (this.detailsOpenSet.has(sourceType)) this.detailsOpenSet.delete(sourceType);
    else this.detailsOpenSet.add(sourceType);
  }

  /** True iff the native row's Details button should be rendered:
   *  there has to be more than one clipper for the user to pick
   *  between.  Same gate as the legacy standalone Clipper section. */
  hasNativeDetails(): boolean {
    return this.clippers.length > 1;
  }

  /** Click handler for the native row's Details button — bubbles up to
   *  the parent, which opens the shared clipper-chooser modal. */
  onNativeDetailsClick(): void {
    this.clipperChooserRequested.emit();
  }

  /** Currently-selected clipper for the native row, looked up in the
   *  ``clippers`` input by ``selectedClipperName``.  Falls back to the
   *  first available clipper so the summary still renders when the
   *  parent hasn't picked one yet. */
  private currentClipper(): ClipperInfo | null {
    if (!this.clippers.length) return null;
    const byName = this.clippers.find((c) => c.name === this.selectedClipperName);
    return byName || this.clippers[0];
  }

  /** Live one-line summary of the native row's clipper, with current
   *  parameter values substituted into its ``summary_template``.  Falls
   *  back to ``description`` when no template is set, and to an empty
   *  string when neither is available.  Missing values fall back to the
   *  clipper's declared parameter defaults so the preview reads cleanly
   *  before the user opens the chooser. */
  nativeSummary(): string {
    const c = this.currentClipper();
    if (!c) return '';
    const params: Record<string, string | number | null | undefined> = {};
    for (const p of c.parameters || []) {
      params[p.key] = p.default;
    }
    for (const k of Object.keys(this.selectedClipperParams)) {
      const v = this.selectedClipperParams[k];
      if (v !== undefined && v !== null && v !== '') params[k] = v;
    }
    return formatSummary(c.summary_template || c.description || '', params);
  }

  /** Live one-line summary of a non-native row's converter, with the
   *  current draft params substituted into its ``summary_template``. */
  nonNativeSummary(sourceType: string): string {
    const c = this.currentConverter(sourceType);
    if (!c) return '';
    return formatSummary(c.summary_template || c.description || '', this.getDraft(sourceType).params);
  }

  currentConverterName(sourceType: string): string {
    return this.getDraft(sourceType).converter || '';
  }

  currentConverter(sourceType: string): ConverterInfo | null {
    const name = this.currentConverterName(sourceType);
    return this.availableConverters.find((c) => c.name === name) || null;
  }

  paramValue(sourceType: string, key: string): string | number | null {
    const v = this.getDraft(sourceType).params[key];
    return v === undefined ? null : v;
  }

  toggleNonNative(sourceType: string, checked: boolean): void {
    const others = this.specs.filter((s) => !(s.source_type === sourceType && s.converter !== null));
    if (checked) {
      this.specsChange.emit([...others, { ...this.getDraft(sourceType) }]);
    } else {
      this.specsChange.emit(others);
    }
  }

  setConverter(sourceType: string, converterName: string): void {
    const c = this.availableConverters.find((x) => x.name === converterName);
    if (!c) return;
    const draft: SourceSpec = {
      source_type: sourceType,
      converter: c.name,
      params: this.defaultParams(c),
    };
    this.drafts.set(sourceType, draft);
    if (this.isNonNativeChecked(sourceType)) {
      this.specsChange.emit(this.specs.map((s) =>
        (s.source_type === sourceType && s.converter !== null) ? { ...draft } : s
      ));
    }
  }

  setParam(sourceType: string, key: string, value: string | number | null): void {
    const draft = this.getDraft(sourceType);
    const updated: SourceSpec = { ...draft, params: { ...draft.params, [key]: value } };
    this.drafts.set(sourceType, updated);
    if (this.isNonNativeChecked(sourceType)) {
      this.specsChange.emit(this.specs.map((s) =>
        (s.source_type === sourceType && s.converter !== null) ? { ...updated } : s
      ));
    }
  }

  private getDraft(sourceType: string): SourceSpec {
    let draft = this.drafts.get(sourceType);
    if (!draft) {
      const c = this.convertersFor(sourceType)[0];
      draft = {
        source_type: sourceType,
        converter: c?.name ?? null,
        params: c ? this.defaultParams(c) : {},
      };
      this.drafts.set(sourceType, draft);
    }
    return draft;
  }

  private defaultParams(c: ConverterInfo): Record<string, string> {
    const out: Record<string, string> = {};
    for (const f of c.fields || []) {
      out[f.key] = String(f.default ?? '');
    }
    return out;
  }
}

/** Substitute ``{key}`` placeholders in *template* with values from
 *  *params*.  Unknown keys and ``null`` / ``undefined`` values are
 *  rendered as ``"?"`` so the surrounding sentence remains readable. */
function formatSummary(
  template: string,
  params: Record<string, string | number | null | undefined>,
): string {
  if (!template) return '';
  return template.replace(/\{([^{}]+)\}/g, (_match, key: string) => {
    const v = params[key];
    if (v === undefined || v === null || v === '') return '?';
    return String(v);
  });
}
