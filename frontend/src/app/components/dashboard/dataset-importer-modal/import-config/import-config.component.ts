import { Component, ElementRef, EventEmitter, HostListener, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IconComponent } from '../../../icon/icon.component';

/** Output-media-type select + auto-detection hint chip shared by the
 *  server-folder (``sf``) and local-folder/local-files (``lf``) flows
 *  of the Add Dataset modal.  The two flows had identical markup for
 *  this block (same label text, same widget shape, same hint
 *  rendering) with only the bound state and field id differing.
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
 *  ``<vt-icon>``) alongside its label; native ``<option>`` elements
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

  /** ``folder_import_name`` values to show in the dropdown (same list
   *  the importer's ``media_type`` field declares as ``options``). */
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

  /** Index of the keyboard-highlighted option while the popup is open
   *  (the ``aria-activedescendant`` target).  ``-1`` when the popup is
   *  closed or no option is highlighted. */
  activeIndex = -1;

  constructor(private hostEl: ElementRef<HTMLElement>) {}

  /** ``id`` for the popup ``role="listbox"`` element, derived from the
   *  trigger's id so it stays unique across the multiple flows the modal
   *  may render at once. */
  get listboxId(): string {
    return `${this.mediaTypeFieldId}-listbox`;
  }

  /** ``id`` for the option at ``index`` so the combobox trigger can
   *  point ``aria-activedescendant`` at the highlighted row. */
  optionId(index: number): string {
    return `${this.mediaTypeFieldId}-option-${index}`;
  }

  /** ``id`` of the currently active option, or ``null`` when the popup
   *  is closed / nothing is highlighted (so ``aria-activedescendant`` is
   *  dropped from the DOM rather than dangling). */
  get activeDescendantId(): string | null {
    return this.open && this.activeIndex >= 0 ? this.optionId(this.activeIndex) : null;
  }

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
    if (this.open) {
      this.close();
    } else {
      this.openPopup();
    }
  }

  /** Open the popup and seed the keyboard highlight on the current
   *  selection (or the first option) so arrow keys have a starting
   *  point. */
  private openPopup(): void {
    this.open = true;
    const selected = this.mediaTypeOptions.indexOf(this.mediaType);
    this.activeIndex = selected >= 0 ? selected : 0;
  }

  private close(): void {
    this.open = false;
    this.activeIndex = -1;
  }

  select(opt: string): void {
    this.close();
    if (opt !== this.mediaType) {
      this.mediaTypeChange.emit(opt);
    }
  }

  /** Move the keyboard highlight to ``index`` (clamped) and scroll it
   *  into view within the popup. */
  private setActiveIndex(index: number): void {
    const last = this.mediaTypeOptions.length - 1;
    if (last < 0) return;
    this.activeIndex = Math.max(0, Math.min(last, index));
    // Defer until the ``[id]`` binding has flushed so the lookup hits the
    // freshly-highlighted row.
    queueMicrotask(() => {
      const el = this.hostEl.nativeElement.querySelector(`#${CSS.escape(this.optionId(this.activeIndex))}`);
      (el as HTMLElement | null)?.scrollIntoView({ block: 'nearest' });
    });
  }

  /** Keyboard handling for the combobox trigger.  Implements the
   *  WAI-ARIA select-only combobox pattern: arrows / Home / End move the
   *  ``aria-activedescendant`` highlight, Enter / Space commit it, and
   *  Escape dismisses; DOM focus stays on the trigger throughout. */
  onTriggerKeydown(event: KeyboardEvent): void {
    if (!this.open) {
      if (['ArrowDown', 'ArrowUp', 'Enter', ' ', 'Spacebar'].includes(event.key)) {
        event.preventDefault();
        this.openPopup();
      }
      return;
    }

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        this.setActiveIndex(this.activeIndex + 1);
        break;
      case 'ArrowUp':
        event.preventDefault();
        this.setActiveIndex(this.activeIndex - 1);
        break;
      case 'Home':
        event.preventDefault();
        this.setActiveIndex(0);
        break;
      case 'End':
        event.preventDefault();
        this.setActiveIndex(this.mediaTypeOptions.length - 1);
        break;
      case 'Enter':
      case ' ':
      case 'Spacebar':
        event.preventDefault();
        if (this.activeIndex >= 0) {
          this.select(this.mediaTypeOptions[this.activeIndex]);
        }
        break;
      case 'Escape':
        event.preventDefault();
        this.close();
        break;
      case 'Tab':
        this.close();
        break;
    }
  }

  /** Close the popup when the user clicks outside the host element. */
  @HostListener('document:click', ['$event'])
  onDocumentClick(event: MouseEvent): void {
    if (!this.open) return;
    const target = event.target as Node | null;
    if (target && this.hostEl.nativeElement.contains(target)) return;
    this.close();
  }
}
