import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IconComponent } from '../../../icon/icon.component';
import { DropZoneComponent } from '../../../drop-zone/drop-zone.component';
import {
  DemoDataset,
  ImporterInfo,
  ImporterPickerTab,
  MediaTypeInfo,
} from '../../../../models/api.models';
import { ManagedColumns } from '../../../../utils/managed-columns';

/** Shared "where does the media come from?" widget.  Renders the
 *  importer category tab bar + sub-tab bar at the top, then a
 *  source-specific picker below:
 *
 *  - ``demo``                   → media-type tab bar + sortable demo table
 *  - ``server_folder``          → typed absolute-path input
 *  - ``local_folder`` / ``local_files`` → file/folder dropzone
 *  - any other ``picker_view``  → nothing (parent renders its own form)
 *
 *  The component is presentational: the parent owns all state and
 *  passes it in via ``@Input``s, including the precomputed
 *  ``visibleImporterTabs`` and ``importersForActiveTab`` lists.  User
 *  actions surface as ``@Output`` events; the parent decides what to do
 *  (the Add Dataset modal runs the importer, the New Detector modal
 *  materialises a single example file).
 *
 *  Output configuration that visually sits next to the source widget
 *  (Dataset Name input, ``<vt-import-config>``, ``<vt-import-advanced>``,
 *  Include-subfolders checkbox) is *not* part of this component.  The
 *  parent projects those into named content slots so the visual order
 *  on each flow stays unchanged:
 *
 *  - ``[demoExtras]`` — sits between the demo media-type tabs and the
 *    demo table (Add Dataset uses it for the Dataset Name + Advanced
 *    block; New Detector leaves it empty).
 *  - ``[sfBefore]`` / ``[sfAfter]`` — sit above/below the typed path
 *    input on the server-folder flow.
 *  - ``[lfBefore]`` / ``[lfAfter]`` — sit above/below the dropzone on
 *    the local-folder / local-files flows.
 */
@Component({
  selector: 'vt-source-picker',
  standalone: true,
  imports: [CommonModule, FormsModule, IconComponent, DropZoneComponent],
  templateUrl: './source-picker.component.html',
  styleUrl: './source-picker.component.scss',
})
export class SourcePickerComponent {
  // === Importer tab/subtab chrome ===

  /** Tabs to render in the top-level category bar, in display order.
   *  Parent computes this from the importer registry; see
   *  ``DatasetImporterModalComponent.visibleImporterTabs``. */
  @Input() visibleImporterTabs: ImporterPickerTab[] = [];

  /** Importers whose ``category`` matches ``activeTab`` (i.e. the
   *  sub-tabs to render). */
  @Input() importersForActiveTab: ImporterInfo[] = [];

  @Input() activeTab = '';
  @Input() selectedImporter: ImporterInfo | null = null;
  @Input() activeImporterTabLabel = '';

  /** Fired when the user clicks a top-level category tab. */
  @Output() activeTabChange = new EventEmitter<string>();
  /** Fired when the user clicks a sub-tab (importer card). */
  @Output() importerSelected = new EventEmitter<ImporterInfo>();

  // --- Chrome customization ---

  /** Hint shown above the sub-tab area when no top tab is selected. */
  @Input() noTabHint = 'Select what type of dataset to add.';
  /** Hint shown when a tab with multiple importers has none picked yet.
   *  ``{label}`` is replaced with the active tab's label. */
  @Input() noImporterHintTemplate = 'Select how to add a {label} dataset.';
  /** ``title`` attribute on top-level tab buttons.  ``{label}`` is
   *  replaced with the tab label. */
  @Input() tabTitleTemplate = 'Show {label} dataset importers';
  /** Message shown when the active tab has zero importers. */
  @Input() emptyCategoryText = 'No importers in this category.';

  // === Demo source view ===

  @Input() demos: DemoDataset[] = [];
  @Input() filteredDemos: DemoDataset[] = [];
  @Input() demoLoading = false;
  @Input() demoTabs: string[] = [];
  @Input() activeDemoTab = '';
  @Input() demoCols: ManagedColumns | null = null;
  @Input() mediaTypes: MediaTypeInfo[] = [];

  @Output() activeDemoTabChange = new EventEmitter<string>();
  @Output() demoSelected = new EventEmitter<DemoDataset>();

  @Input() demoLoadingText = 'Loading demo datasets...';
  @Input() demoEmptyText = 'No demo datasets available.';
  @Input() demoNoTabHint = 'Select the media type to demonstrate.';

  // === Server folder source view ===

  @Input() sfPathInputValue = '';
  @Output() sfPathInputValueChange = new EventEmitter<string>();
  /** Fired when the user finalises the typed path (Enter or blur). */
  @Output() sfPathApplied = new EventEmitter<void>();

  @Input() sfPathLabel = 'Folder to import';
  @Input() sfPathPlaceholder = '/absolute/server/path/to/folder';
  @Input() sfPathFieldId = 'sf-path-input';

  // === Local folder/files source view ===

  @Input() lfPickerKind: 'folder' | 'files' = 'folder';
  @Input() lfFiles: File[] = [];
  @Input() lfFolderName = '';
  /** Whether to show the per-kind explanatory paragraph above the
   *  dropzone.  Disabled by the new-detector modal which has its own
   *  "pick a file from this computer" prose. */
  @Input() lfShowInfo = true;
  @Output() lfFilesDropped = new EventEmitter<File[]>();

  // === View dispatch ===

  /** ``picker_view`` of the currently selected importer.  Drives which
   *  source widget is rendered below the chrome. */
  @Input() activePickerView = '';

  // -------------------------------------------------------------------

  get noImporterHint(): string {
    return this.noImporterHintTemplate.replace('{label}', this.activeImporterTabLabel);
  }

  tabTitle(label: string): string {
    return this.tabTitleTemplate.replace('{label}', label);
  }

  // === Demo helpers ===

  getDemoTabIcon(mediaType: string): string {
    const mt = this.mediaTypes.find((m) => m.type_id === mediaType);
    return mt?.icon || '';
  }

  getDemoTabText(mediaType: string): string {
    const mt = this.mediaTypes.find((m) => m.type_id === mediaType);
    return mt ? mt.name : mediaType;
  }

  onDemoHeaderClick(col: string): void {
    if (this.demoCols?.meta(col).sortable) this.demoCols.sortBy(col);
  }

  statusBadgeClass(status: string): string {
    if (status === 'ready') return 'badge-ready';
    if (status === 'needs_embedding') return 'badge-embedding';
    return 'badge-download';
  }

  statusBadgeLabel(status: string): string {
    if (status === 'ready') return 'Ready';
    if (status === 'needs_embedding') return 'Needs Embed';
    return 'Needs Download';
  }
}
