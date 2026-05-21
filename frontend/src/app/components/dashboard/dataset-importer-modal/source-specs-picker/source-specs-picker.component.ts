import { Component, EventEmitter, Input, OnChanges, Output, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { ConverterInfo, SourceSpec } from '../../../../models/api.models';

/** Checkbox column for choosing which source media types feed a
 *  multi-media import.  The native type sits at the top (always
 *  included by default, no converter); each other source type with at
 *  least one converter to the native type appears as a checkbox with an
 *  "Advanced ▸" opener for picking among converters (when there are
 *  multiple) and editing their params. */
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
  @Output() specsChange = new EventEmitter<SourceSpec[]>();

  /** Per-source-type draft so the user can configure a converter via
   *  the Advanced panel *before* checking the box.  Edits made while
   *  unchecked are preserved here and applied on check. */
  private drafts = new Map<string, SourceSpec>();
  private advancedOpenSet = new Set<string>();

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['nativeType'] && !changes['nativeType'].firstChange) {
      this.drafts.clear();
      this.advancedOpenSet.clear();
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

  isNativeChecked(): boolean {
    return this.specs.some((s) => s.source_type === this.nativeType && s.converter === null);
  }

  isNonNativeChecked(sourceType: string): boolean {
    return this.specs.some((s) => s.source_type === sourceType && s.converter !== null);
  }

  convertersFor(sourceType: string): ConverterInfo[] {
    return this.availableConverters.filter((c) => c.source_type === sourceType);
  }

  /** True iff the Advanced opener should be rendered for *sourceType*:
   *  either multiple converters to pick between, or at least one
   *  user-editable field on the (single) converter. */
  hasAdvanced(sourceType: string): boolean {
    const cs = this.convertersFor(sourceType);
    if (cs.length > 1) return true;
    return !!(cs[0]?.fields?.length);
  }

  isAdvancedOpen(sourceType: string): boolean {
    return this.advancedOpenSet.has(sourceType);
  }

  toggleAdvanced(sourceType: string): void {
    if (this.advancedOpenSet.has(sourceType)) this.advancedOpenSet.delete(sourceType);
    else this.advancedOpenSet.add(sourceType);
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

  toggleNative(checked: boolean): void {
    const others = this.specs.filter((s) => !(s.source_type === this.nativeType && s.converter === null));
    if (checked) {
      this.specsChange.emit([{ source_type: this.nativeType, converter: null, params: {} }, ...others]);
    } else {
      this.specsChange.emit(others);
    }
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
