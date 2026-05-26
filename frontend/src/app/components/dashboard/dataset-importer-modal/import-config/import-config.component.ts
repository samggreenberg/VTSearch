import { Component, ElementRef, EventEmitter, HostListener, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IconComponent } from '../../../icon/icon.component';

/** Output-media-type select + auto-detection hint chip shared by the
 *  server-folder (``sf``) and local-folder/local-files (``lf``) flows
 *  of the Add Dataset modal.  The two flows had identical markup for
 *  this block - same label text, same widget shape, same hint
 *  rendering - with only the bound state and field id differing.
 *
 *  The dataset-name input, the source-side widget (folder path /
 *  dropzone), the recursive checkbox, and the Advanced ▾ block stay
 *  in the parent.  Wrapping the Advanced block here is tempting but
 *  rules out by the template structure: the source widget and
 *  recursive toggle sit between the media-type select and the
 *  Advanced block, so collapsing into one child would force a
 *  re-order of the flow.
 *
 *  The "select" is implemented as a custom button + popup listbox so
 *  each option can show the media type's SVG icon (rendered via
 *  ``<vt-icon>``) alongside its label - native ``<option>`` elements
 *  cannot host markup.
 */
@Component({
  selector: 'vt-import-config',
  standalone: true,
  imports: [CommonModule, FormsModule, IconComponent],
  templateUrl: './import-config.component.html',
  styleUrl: './import-config.component.scss',
})
export class ImportConfigComponent {
  /** ``id`` value for the rendered media-type trigger button.  Each
   *  call site supplies a unique value so the modal does not contain
   *  duplicate ids when multiple flows are present in the DOM. */
  @Input() mediaTypeFieldId = 'import-config-media-type';

  /** Two-way bound output media type (folder_import_name, e.g.
   *  ``"images"``).  Parent reloads embedders/clippers/source-specs on
   *  every change. */
  @Input() mediaType = '';
  @Output() mediaTypeChange = new EventEmitter<string>();

  /** ``folder_import_name`` values to show in the dropdown - same list
   *  the importer's ``media_type`` field declares as ``options``. */
  @Input() mediaTypeOptions: string[] = [];

  /** Map of ``folder_import_name`` → human label.  Parent computes this
   *  once from the ``/api/media-types`` response. */
  @Input() mediaTypeOptionLabels: Record<string, string> = {};

  /** Map of ``folder_import_name`` → icon string (emoji or named SVG
   *  type from :class:`MediaTypeInfo.icon`).  When empty for a given
   *  option, the row renders label-only. */
  @Input() mediaTypeOptionIcons: Record<string, string> = {};

  /** Pre-formatted hint string from
   *  :meth:`DatasetImporterModalComponent.detectionHint`.  Empty hides
   *  the chip entirely. */
  @Input() detectionHint = '';

  /** Whether the custom dropdown is currently expanded. */
  open = false;

  constructor(private hostEl: ElementRef<HTMLElement>) {}

  /** Label for an option in the media-type dropdown.  Falls back to the
   *  option value when no label is supplied (e.g. the parent's
   *  ``mediaTypes`` list has not loaded yet). */
  optionLabel(opt: string): string {
    return this.mediaTypeOptionLabels[opt] || opt;
  }

  /** Icon string for an option (emoji or :class:`IconComponent` type
   *  name).  Empty hides the icon for that row. */
  iconFor(opt: string): string {
    return this.mediaTypeOptionIcons[opt] || '';
  }

  toggle(): void {
    this.open = !this.open;
  }

  select(opt: string): void {
    this.open = false;
    if (opt !== this.mediaType) {
      this.mediaTypeChange.emit(opt);
    }
  }

  /** Close the popup when the user clicks outside the host element. */
  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    if (!this.open) return;
    const target = event.target as Node | null;
    if (target && this.hostEl.nativeElement.contains(target)) return;
    this.open = false;
  }

  /** Close on Escape so keyboard users can dismiss the popup. */
  @HostListener('document:keydown.escape')
  onEscape(): void {
    if (this.open) this.open = false;
  }
}
