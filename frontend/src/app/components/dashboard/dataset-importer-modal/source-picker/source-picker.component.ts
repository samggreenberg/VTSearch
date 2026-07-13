import { ChangeDetectionStrategy, Component, Input, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IconComponent } from '../../../icon/icon.component';
import { DropZoneComponent } from '../../../drop-zone/drop-zone.component';
import { ImporterInfo, ImporterPickerTab, MediaTypeInfo } from '../../../../models/api.models';
import { DemoDatasetEntry } from '../../../../generated/api-client/models/demo-dataset-entry';
import { ManagedColumns } from '../../../../utils/managed-columns';

/** Shared "where does the media come from?" widget.  Renders the
 *  importer category tab bar + sub-tab bar at the top, then a
 *  source-specific picker below.
 *
 *  Navigation is a *persistent* two-level tab bar: the category tabs
 *  (`.importer-tab-bar`) and importer sub-tabs (`.importer-subtab-bar`)
 *  stay on screen while the selected source form renders beneath them.
 *  This intentionally diverges from the New-detector › Trained flow's
 *  picker → form → `← Back` shape, and there is deliberately **no**
 *  `.back-btn` here: the tabs never hide an outer view, so switching
 *  source is done by clicking a different tab rather than retreating.
 *  See CLAUDE.md "Nested-modal back buttons" → persistent-tab-picker
 *  exception (issue #2329).  Source views rendered below the chrome:
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
 *  - ``[demoBefore]`` - sits between the demo media-type tabs and the
 *    demo table (Add Dataset uses it to project a media-type dropdown
 *    that replaces the inner tabs; New Detector leaves it empty).
 *  - ``[demoAfter]`` - sits below the demo table (Add Dataset uses it
 *    for the Dataset Name + Advanced block; New Detector leaves it
 *    empty).
 *  - ``[sfBefore]`` / ``[sfAfter]`` - sit above/below the typed path
 *    input on the server-folder flow.
 *  - ``[lfBefore]`` / ``[lfAfter]`` - sit above/below the dropzone on
 *    the local-folder / local-files flows.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
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
  readonly visibleImporterTabs = input<ImporterPickerTab[]>([]);

  /** Importers whose ``category`` matches ``activeTab`` (i.e. the
   *  sub-tabs to render). */
  @Input() importersForActiveTab: ImporterInfo[] = [];

  readonly activeTab = input('');
  readonly selectedImporter = input<ImporterInfo | null>(null);
  readonly activeImporterTabLabel = input('');

  /** Fired when the user clicks a top-level category tab. */
  readonly activeTabChange = output<string>();
  /** Fired when the user clicks a sub-tab (importer card). */
  readonly importerSelected = output<ImporterInfo>();

  // --- Chrome customization ---

  /** Hint shown above the sub-tab area when no top tab is selected. */
  readonly noTabHint = input('Select what type of dataset to add.');
  /** Hint shown when a tab with multiple importers has none picked yet.
   *  ``{label}`` is replaced with the active tab's label. */
  readonly noImporterHintTemplate = input('Select how to add a {label} dataset.');
  /** ``title`` attribute on top-level tab buttons.  ``{label}`` is
   *  replaced with the tab label. */
  readonly tabTitleTemplate = input('Show {label} dataset importers');
  /** Message shown when the active tab has zero importers. */
  readonly emptyCategoryText = input('No importers in this category.');

  /** When ``false`` (default), the sub-tab row is suppressed when the
   *  active category has exactly one importer - the parent is expected
   *  to auto-select the lone importer and the redundant sub-tab adds
   *  no information.  Callers that always want the sub-tab visible
   *  (e.g. the New Detector media picker, where every category is a
   *  distinct kind of source the user should see labelled) set this to
   *  ``true``. */
  readonly alwaysShowSubtabBar = input(false);

  // === Demo source view ===

  readonly demos = input<DemoDatasetEntry[]>([]);
  readonly filteredDemos = input<DemoDatasetEntry[]>([]);
  readonly demoLoading = input(false);
  readonly demoTabs = input<string[]>([]);
  readonly activeDemoTab = input('');
  @Input() demoCols: ManagedColumns | null = null;
  readonly mediaTypes = input<MediaTypeInfo[]>([]);

  readonly activeDemoTabChange = output<string>();
  readonly demoSelected = output<DemoDatasetEntry>();

  readonly demoLoadingText = input('Loading demo datasets…');
  readonly demoEmptyText = input('No demo datasets available.');
  readonly demoNoTabHint = input('Select the media type to demonstrate.');

  /** Optional predicate run on every demo row.  When provided and it
   *  returns ``true``, the row gets a ``disabled`` class for visual
   *  styling.  Source picker still emits ``demoSelected`` for clicks
   *  - the parent is responsible for treating disabled rows as
   *  no-ops. */
  readonly demoRowDisabledFn = input<((demo: DemoDatasetEntry) => boolean) | null>(null);
  /** Optional formatter for the ``title`` attribute on each demo row. */
  readonly demoRowTitleFn = input<((demo: DemoDatasetEntry) => string) | null>(null);

  /** When ``true`` (default) the inner media-type tab bar renders above
   *  the demo table.  Callers that want to render their own media-type
   *  picker (e.g. the Add Dataset modal, which uses a dropdown to match
   *  the other importer flows) set this to ``false`` and project their
   *  widget into the ``[demoBefore]`` slot. */
  readonly showDemoMediaTypeTabs = input(true);

  /** Name of the currently-selected demo row.  Used to highlight the
   *  selected row when the parent treats row clicks as a selection (not
   *  an immediate submit).  Empty string disables the highlight. */
  readonly selectedDemoName = input('');

  // === Server folder source view ===

  @Input() sfPathInputValue = '';
  readonly sfPathInputValueChange = output<string>();
  /** Fired when the user finalises the typed path (Enter or blur). */
  readonly sfPathApplied = output<void>();

  readonly sfPathLabel = input('Folder to import');
  readonly sfPathPlaceholder = input('/absolute/server/path/to/folder');
  readonly sfPathFieldId = input('sf-path-input');
  /** Whether to fire ``sfPathApplied`` on the path input's ``blur``
   *  event in addition to Enter.  The Add Dataset modal wants
   *  blur-to-detect; the New Detector modal explicitly drives loading
   *  through a Load button and disables blur to avoid double-firing. */
  readonly sfApplyOnBlur = input(true);

  // === Local folder/files source view ===

  readonly lfPickerKind = input<'folder' | 'files'>('folder');
  @Input() lfFiles: File[] = [];
  @Input() lfFolderName = '';
  readonly lfFilesDropped = output<File[]>();

  /** ``vt-drop-zone`` label rendered when ``lfPickerKind === 'folder'``. */
  readonly lfFolderDropLabel = input('Drop a folder here to import');
  /** ``vt-drop-zone`` sublabel rendered when ``lfPickerKind === 'folder'``. */
  readonly lfFolderDropSublabel = input('or click to browse');
  /** ``vt-drop-zone`` label rendered when ``lfPickerKind === 'files'``. */
  readonly lfFilesDropLabel = input('Drop a paths file (.txt, .list, or .npz)');
  /** ``vt-drop-zone`` sublabel rendered when ``lfPickerKind === 'files'``. */
  readonly lfFilesDropSublabel = input('or click to browse');
  /** ``accept`` attribute for the ``files`` kind dropzone (comma-separated
   *  extensions / MIME types).  Empty string accepts anything. */
  readonly lfFilesAcceptAttr = input('.txt,.list,.npz');

  /** Whether to render the "Folder*" / "Paths file*" form-label above the
   *  dropzone.  Disabled by callers (e.g. the New Detector modal) that
   *  bake the affordance entirely into the dropzone's own label. */
  readonly lfShowFieldLabel = input(true);
  /** Label for the dropzone form-label, when ``lfShowFieldLabel`` is on. */
  readonly lfFolderFieldLabel = input('Folder');
  /** Label for the dropzone form-label in ``files`` mode. */
  readonly lfFilesFieldLabel = input('Paths file');

  /** Whether to render the "Selected N files" / "Using paths from …" info
   *  text below the dropzone.  Disabled by callers that only use the
   *  first picked file (e.g. the New Detector modal) and have no
   *  meaningful count to show. */
  readonly lfShowFileCount = input(true);

  // === View dispatch ===

  /** ``picker_view`` of the currently selected importer.  Drives which
   *  source widget is rendered below the chrome. */
  readonly activePickerView = input('');

  // -------------------------------------------------------------------

  get noImporterHint(): string {
    return this.noImporterHintTemplate().replace('{label}', this.activeImporterTabLabel());
  }

  tabTitle(label: string): string {
    return this.tabTitleTemplate().replace('{label}', label);
  }

  // === Demo helpers ===

  getDemoTabIcon(mediaType: string): string {
    const mt = this.mediaTypes().find((m) => m.type_id === mediaType);
    return mt?.icon || '';
  }

  getDemoTabText(mediaType: string): string {
    const mt = this.mediaTypes().find((m) => m.type_id === mediaType);
    return mt ? mt.name : mediaType;
  }

  onDemoHeaderClick(col: string): void {
    if (this.demoCols?.meta(col).sortable) this.demoCols.sortBy(col);
  }

  demoRowDisabled(demo: DemoDatasetEntry): boolean {
    const demoRowDisabledFn = this.demoRowDisabledFn();
    return demoRowDisabledFn ? demoRowDisabledFn(demo) : false;
  }

  demoRowTitle(demo: DemoDatasetEntry): string {
    const demoRowTitleFn = this.demoRowTitleFn();
    return demoRowTitleFn ? demoRowTitleFn(demo) : '';
  }

  statusBadgeClass(status: string): string {
    if (status === 'ready') return 'badge-ready';
    if (status === 'needs_embedding') return 'badge-embedding';
    return 'badge-download';
  }

  statusBadgeLabel(status: string): string {
    if (status === 'ready') return 'Ready';
    if (status === 'needs_embedding') return 'Needs setup';
    return 'Needs Download';
  }
}
