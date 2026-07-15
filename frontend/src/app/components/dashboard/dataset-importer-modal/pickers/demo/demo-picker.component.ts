import { ChangeDetectionStrategy, ChangeDetectorRef, Component, EventEmitter, HostListener, Input, Output, inject, signal } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { ClipperChooserComponent, ClipperSelection } from '../../../clipper-chooser/clipper-chooser.component';
import { ImportAdvancedComponent } from '../../import-advanced/import-advanced.component';
import { DatasetsListingsApiService } from '../../../../../services/datasets-listings-api.service';
import { ClipperInfo, EmbedderInfo, ImporterInfo, MediaTypeInfo } from '../../../../../models/api.models';
import { DemoDatasetEntry } from '../../../../../generated/api-client/models/demo-dataset-entry';
import { ColMeta, ManagedColumns } from '../../../../../utils/managed-columns';
import { ImportDefaultsService } from '../shared/import-defaults.service';
import { composeEmbedders, getTabLabel, mediaTypeIconsById, mediaTypeLabels } from '../shared/media-type.util';

/** Demo-dataset picker view: media-type tab bar + sortable table of
 *  pre-configured, optionally pre-downloaded demo datasets.  Row clicks
 *  record a selection (via the parent's shared `<vt-source-picker>`,
 *  which renders the table itself); the Import footer button commits
 *  it via :meth:`submit`.
 *
 *  Sits behind the shared ``<vt-source-picker>`` chrome; this component
 *  owns the media-type dropdown + Advanced block around the table (the
 *  ``demoBefore`` / ``demoAfter`` projection slots), the demo list/tab
 *  business logic, and the embedder-aware "ready" status recomputation. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-demo-picker',
  standalone: true,
  imports: [FormsModule, ImportAdvancedComponent, ClipperChooserComponent],
  templateUrl: './demo-picker.component.html',
  styleUrl: './demo-picker.component.scss',
})
export class DemoPickerComponent {
  private datasetsListingsApi = inject(DatasetsListingsApiService);
  private importDefaults = inject(ImportDefaultsService);
  private cdr = inject(ChangeDetectorRef);

  @Input() guessedMediaType = '';
  @Input() guessedMediaEmbedder = '';

  @Input() buildProjection = false;
  @Output() buildProjectionChange = new EventEmitter<boolean>();
  @Input() mergeNearDuplicates = false;
  @Output() mergeNearDuplicatesChange = new EventEmitter<boolean>();

  /** Fired when the user commits the current row selection via the
   *  Import footer button; the parent forwards the payload to its own
   *  ``demoSelected`` output and closes the modal. */
  @Output() demoDatasetSelected = new EventEmitter<DemoDatasetEntry & Record<string, unknown>>();

  readonly selectedImporter = signal<ImporterInfo | null>(null);

  readonly demos = signal<DemoDatasetEntry[]>([]);
  /** Re-fetched independently every time the picker opens (mirrors the
   *  original mega-component's behaviour, which re-pulled the registry
   *  before building tabs so newly-registered media types show up
   *  without a full page reload). */
  readonly mediaTypes = signal<MediaTypeInfo[]>([]);
  readonly demoTabs = signal<string[]>([]);
  readonly activeTab = signal('');
  readonly demoLoading = signal(false);
  readonly demoEmbedders = signal<EmbedderInfo[]>([]);
  readonly selectedDemoEmbedder = signal('');
  readonly selectedDemoPatchEmbedder = signal('');
  readonly selectedDemoStructuralEmbedder = signal('');
  readonly demoClippers = signal<ClipperInfo[]>([]);
  readonly selectedDemoClipper = signal('');
  readonly demoClipperParamValues = signal<Record<string, number | string>>({});
  /** For a *convert-out* half-type tab (e.g. ``document``), the embeddable
   *  targets it can convert into (``converts_to``). Empty for a normal
   *  embeddable tab. Drives the "Convert to" selector. */
  readonly activeTabConvertsTo = signal<string[]>([]);
  /** The chosen embeddable target type for a non-embeddable tab (e.g.
   *  ``image``), or ``''`` for a normal tab. */
  readonly selectedConverterTarget = signal('');
  /** The convert-on-load converter name (e.g. ``document2image``), or ``''``
   *  when the active tab is directly embeddable. Passed to demo-list status
   *  queries and the load payload. */
  readonly selectedDemoConverter = signal('');
  /** Optional user-supplied dataset name.  Empty means "use the demo
   *  entry's label". */
  demoDatasetName = '';
  private demoDatasetNameDirty = false;
  /** Currently-selected demo row.  Bound (via the parent) onto the
   *  sibling ``<vt-source-picker>``'s row-highlight input and read by
   *  the parent's footer submit button, hence a signal. */
  readonly selectedDemo = signal<DemoDatasetEntry | null>(null);

  static readonly DEMO_COL_META: Record<string, ColMeta> = {
    label: { label: 'Name', title: 'Demo dataset name (click to sort)', sortable: true },
    num_files: { label: '# Media', title: 'Number of media files in the demo dataset (click to sort)', sortable: true },
    description: { label: 'Description', title: 'Short description of the demo dataset contents (click to sort)', sortable: true },
    status: { label: 'Readiness', title: 'Whether the dataset is pre-downloaded and ready to load immediately, or needs to be fetched first (click to sort)', sortable: true },
  };
  static readonly DEMO_COLUMNS_DEFAULT = ['label', 'num_files', 'description', 'status'];
  private static readonly DEMO_COL_ORDER_KEY = 'vtsearch.dashboard.demoColumnOrder';

  clipperChooserOpen = false;
  clipperChooserClippers: ClipperInfo[] = [];

  openClipperChooser(): void {
    this.clipperChooserClippers = this.demoClippers();
    this.clipperChooserOpen = true;
  }

  onClipperChooserSelected(selection: ClipperSelection): void {
    this.clipperChooserOpen = false;
    this.selectedDemoClipper.set(selection.name);
    this.demoClipperParamValues.set({ ...selection.params });
    this.updateDemoStatuses();
    this.refetchDemoStatuses(this.selectedDemoEmbedder(), this.effectiveDemoClipper());
  }

  onClipperChooserCancelled(): void {
    this.clipperChooserOpen = false;
    const clippers = this.clipperChooserClippers;
    const defaultClipper = clippers.find((c) => c.name.endsWith('_default')) || clippers[0];
    this.selectedDemoClipper.set(defaultClipper?.name || '');
    this.demoClipperParamValues.set({});
    this.updateDemoStatuses();
    this.refetchDemoStatuses(this.selectedDemoEmbedder(), this.effectiveDemoClipper());
  }

  demoCols = new ManagedColumns(
    DemoPickerComponent.DEMO_COLUMNS_DEFAULT,
    DemoPickerComponent.DEMO_COL_META,
    { initialSort: 'num_files', storageKey: DemoPickerComponent.DEMO_COL_ORDER_KEY },
  );

  @HostListener('document:mousemove', ['$event'])
  onDocResizeMove(event: MouseEvent): void {
    this.demoCols.onResizeMove(event);
  }

  @HostListener('document:mouseup')
  onDocResizeEnd(): void {
    this.demoCols.onResizeEnd();
  }

  get effectiveSoloMediaType(): string | null {
    return this.importDefaults.effectiveSoloMediaType;
  }

  get mediaTypeLabels(): Record<string, string> {
    return mediaTypeLabels(this.mediaTypes());
  }

  get demoMediaTypeIcons(): Record<string, string> {
    return mediaTypeIconsById(this.mediaTypes());
  }

  lockedEmbedderFor(mediaTypeFolderOrTypeId: string, embedders: EmbedderInfo[]): string {
    return this.importDefaults.lockedEmbedderFor(mediaTypeFolderOrTypeId, this.mediaTypes(), embedders);
  }

  getTabLabel(mediaType: string): string {
    return getTabLabel(this.mediaTypes(), mediaType);
  }

  open(importer: ImporterInfo | null): void {
    this.selectedImporter.set(importer);
    this.demoLoading.set(true);
    this.demos.set([]);
    this.demoTabs.set([]);
    this.activeTab.set('');
    this.demoDatasetName = '';
    this.demoDatasetNameDirty = false;
    this.selectedDemo.set(null);

    this.datasetsListingsApi.getMediaTypes().subscribe({
      next: (res) => {
        this.mediaTypes.set(res.media_types || []);
        this.fetchDemos();
      },
      error: () => {
        this.fetchDemos();
      },
    });

    // `open()` is invoked imperatively from the parent's importer-selection
    // handler (a listener bound on a sibling `<vt-source-picker>`), so this
    // component's own OnPush view is not on the ancestor-marked dirty path
    // that call produces; force a check so the reset picker actually paints.
    this.cdr.markForCheck();
  }

  private fetchDemos(embedder?: string): void {
    this.datasetsListingsApi.getDemoList(embedder).subscribe({
      next: (demoRes) => {
        this.demos.set(demoRes.datasets || []);
        this.buildDemoTabs();
        this.demoLoading.set(false);
      },
      error: () => {
        this.demoLoading.set(false);
      },
    });
  }

  private buildDemoTabs(): void {
    const grouped = new Set(this.demos().map((d) => d.media_type));
    const registryOrder = this.mediaTypes().map((mt) => mt.type_id);
    const tabs = registryOrder.filter((mt) => grouped.has(mt));
    for (const mt of grouped) {
      if (!tabs.includes(mt)) {
        tabs.push(mt);
      }
    }
    this.demoTabs.set(tabs);
    const solo = this.effectiveSoloMediaType;
    if (solo && this.demoTabs().includes(solo)) {
      this.demoTabs.set([solo]);
    }
    if (this.demoTabs().length > 0) {
      const needsSelect = !this.activeTab() || (solo && this.activeTab() !== solo);
      if (needsSelect) {
        const guessed = this.guessedMediaType;
        const preferred = solo && this.demoTabs().includes(solo)
          ? solo
          : (guessed && this.demoTabs().includes(guessed)
            ? guessed
            : (this.demoTabs().includes('audio') ? 'audio' : this.demoTabs()[0]));
        this.selectDemoTabWithEmbedder(preferred);
      }
    }
  }

  private mediaTypeInfo(typeId: string): MediaTypeInfo | undefined {
    return this.mediaTypes().find((t) => t.type_id === typeId);
  }

  private loadDemoEmbedders(mediaType: string): void {
    this.selectedDemoPatchEmbedder.set('');
    this.selectedDemoStructuralEmbedder.set('');
    if (!mediaType) {
      this.demoEmbedders.set([]);
      this.selectedDemoEmbedder.set('');
      this.demoClippers.set([]);
      this.selectedDemoClipper.set('');
      this.demoClipperParamValues.set({});
      this.activeTabConvertsTo.set([]);
      this.selectedConverterTarget.set('');
      this.selectedDemoConverter.set('');
      return;
    }
    // A *convert-out* half type (document: embeddable === false) has no
    // embedder of its own — its demos load through a converter to an
    // embeddable target (converts_to). Offer those targets and load the
    // embedder/clipper choices for the (default) target instead.
    const info = this.mediaTypeInfo(mediaType);
    const convertsTo = info && info.embeddable === false ? info.converts_to || [] : [];
    this.activeTabConvertsTo.set(convertsTo);
    this.applyConverterTarget(mediaType, convertsTo.length ? convertsTo[0] : '');
  }

  /** Set the convert-on-load target for the active tab and (re)load the
   *  embedder/clipper choices for whichever type is actually embeddable
   *  (the target for a non-embeddable tab, else the tab itself). */
  private applyConverterTarget(sourceType: string, target: string): void {
    this.selectedConverterTarget.set(target);
    this.selectedDemoConverter.set(target ? `${sourceType}2${target}` : '');
    const embeddableType = target || sourceType;
    this.datasetsListingsApi.getEmbedders(embeddableType).subscribe({
      next: (embedders) => {
        this.demoEmbedders.set(embedders);
        this.selectedDemoEmbedder.set(
          this.importDefaults.pickInitialEmbedder(embedders, embeddableType, this.mediaTypes(), this.guessedMediaEmbedder),
        );
        this.updateDemoStatuses();
        if (this.selectedDemoEmbedder()) {
          this.refetchDemoStatuses(this.selectedDemoEmbedder(), this.effectiveDemoClipper());
        }
      },
    });
    this.datasetsListingsApi.getClippers(embeddableType).subscribe({
      next: (clippers) => {
        this.demoClippers.set(clippers);
        this.selectedDemoClipper.set(clippers.length > 0 ? clippers[0].name : '');
        this.demoClipperParamValues.set({});
      },
    });
  }

  /** Handler for the "Convert to" selector on a non-embeddable tab. */
  onConverterTargetChange(target: string): void {
    this.applyConverterTarget(this.activeTab(), target);
    this.cdr.markForCheck();
  }

  /** The demo clipper the user actually chose, or ``''`` when it is a
   *  no-op (the explicit "None"/``*_default`` choice). */
  private effectiveDemoClipper(): string {
    const name = this.selectedDemoClipper();
    const isDefault = !name || name.endsWith('_default');
    return isDefault ? '' : name;
  }

  onDemoEmbedderChange(embedder: string): void {
    this.selectedDemoEmbedder.set(embedder);
    this.updateDemoStatuses();
    this.refetchDemoStatuses(embedder, this.effectiveDemoClipper());
  }

  onDemoClipperChange(clipper: string): void {
    this.selectedDemoClipper.set(clipper);
    this.updateDemoStatuses();
    this.refetchDemoStatuses(this.selectedDemoEmbedder(), this.effectiveDemoClipper());
  }

  /** Re-compute each demo's status client-side based on the selected
   *  embedder and clipper.  Only processes demos for the active tab. */
  private updateDemoStatuses(): void {
    const emb = this.selectedDemoEmbedder();
    const clip = this.effectiveDemoClipper();
    for (const demo of this.demos()) {
      if (demo.media_type !== this.activeTab()) continue;
      if (demo.status === 'needs_download') continue;

      if (!demo.pkl_embedder) {
        if (emb && demo.status === 'ready') {
          demo.status = 'needs_embedding';
          demo.ready = false;
        }
        continue;
      }

      const embedderMatch = !emb || demo.pkl_embedder === emb;
      const clipperMatch = !clip || !demo.pkl_clipper || demo.pkl_clipper === clip;

      if (embedderMatch && clipperMatch) {
        demo.status = 'ready';
        demo.ready = true;
      } else {
        demo.status = 'needs_embedding';
        demo.ready = false;
      }
    }
  }

  /** Re-fetch the demo list from the server with the given embedder and
   *  clipper so the backend can authoritatively determine each demo's
   *  status. */
  private refetchDemoStatuses(embedder: string, clipper?: string): void {
    this.datasetsListingsApi.getDemoList(embedder, clipper, this.selectedDemoConverter()).subscribe({
      next: (demoRes) => {
        this.demos.set(demoRes.datasets || []);
      },
    });
  }

  get filteredDemos(): DemoDatasetEntry[] {
    const items = this.demos().filter((d) => d.media_type === this.activeTab());
    const statusOrder: Record<string, number> = { ready: 0, needs_embedding: 1, needs_download: 2 };
    const sortKey = this.demoCols.sortColumn;
    const asc = this.demoCols.sortAsc;
    return items.sort((a, b) => {
      const key = sortKey as keyof DemoDatasetEntry;
      let va: any = a[key];
      let vb: any = b[key];
      if (key === 'status') {
        va = statusOrder[va as string] ?? 3;
        vb = statusOrder[vb as string] ?? 3;
      }
      if (typeof va === 'number' && typeof vb === 'number') {
        return asc ? va - vb : vb - va;
      }
      va = String(va || '').toLowerCase();
      vb = String(vb || '').toLowerCase();
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
  }

  selectDemoTab(tab: string): void {
    this.activeTab.set(tab);
    this.loadDemoEmbedders(tab);
  }

  /** Invoked via a listener bound on the sibling `<vt-source-picker>`
   *  in the parent's template, so this component isn't on the
   *  ancestor-marked dirty path; `markForCheck()` covers the plain
   *  `demoDatasetName` field. */
  selectDemoTabWithEmbedder(tab: string): void {
    this.selectDemoTab(tab);
    this.selectedDemo.set(null);
    if (!this.demoDatasetNameDirty) {
      this.demoDatasetName = '';
    }
    this.cdr.markForCheck();
  }

  onDemoDatasetNameInput(value: string): void {
    this.demoDatasetName = value;
    this.demoDatasetNameDirty = true;
  }

  /** Click handler for a row in the demo grid (invoked via a listener
   *  bound on the sibling `<vt-source-picker>`).  Records the selection
   *  and auto-populates the Dataset Name input unless the user has
   *  manually edited it.  The actual import is deferred to
   *  :meth:`submit`. */
  selectDemo(demo: DemoDatasetEntry): void {
    this.selectedDemo.set(demo);
    if (!this.demoDatasetNameDirty) {
      this.demoDatasetName = demo.label || '';
    }
    this.cdr.markForCheck();
  }

  /** Commit the currently-selected demo.  Bound to the Import footer
   *  button in the parent's template; disabled there until a row is
   *  selected. */
  submit(): void {
    const demo = this.selectedDemo();
    if (!demo) return;
    const userName = (this.demoDatasetName || '').trim();
    const embedders = composeEmbedders(this.selectedDemoEmbedder(), this.selectedDemoPatchEmbedder(), this.selectedDemoStructuralEmbedder());
    const effClipper = this.effectiveDemoClipper();
    const converter = this.selectedDemoConverter();
    this.demoDatasetSelected.emit({
      ...demo,
      embedder: this.selectedDemoEmbedder(),
      ...(embedders ? { embedders } : {}),
      ...(effClipper ? { clipper: effClipper, clipper_params: { ...this.demoClipperParamValues() } } : {}),
      ...(converter ? { converter } : {}),
      dataset_name: userName,
      build_projection: this.buildProjection,
      merge_near_duplicates: this.mergeNearDuplicates,
    } as any);
  }
}
