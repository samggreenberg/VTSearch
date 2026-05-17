import { Component, ElementRef, HostListener, Input, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { DatasetStateService } from '../../services/dataset-state.service';
import { ActiveContextService } from '../../services/active-context.service';
import { ContextSwitchService } from '../../services/context-switch.service';
import { NewThingFlowsService } from '../../services/new-thing-flows.service';
import { DatasetRegistryEntry, DetectorRegistryEntry } from '../../models/api.models';
import { isPairCompatible } from '../../utils/context-compat';

type PulldownKind = 'dataset' | 'detector';

interface PulldownRow {
  id: string;
  name: string;
  mediaType: string;
  loaded: boolean;
  active: boolean;
  compatibleWithOther: boolean;
  incompatReason: string;
}

/**
 * Top-bar pulldown that lets the user switch the active dataset or
 * detector without going back to the Dashboard. One instance per half
 * of the pair (parameterised by `kind`).
 *
 * Renders a closed-state button (glyph + label + caret) and, when
 * opened, a dropdown with one row per registry entry plus a
 * "+ Add New" footer. Click outside to close. Keyboard: ArrowUp /
 * ArrowDown to navigate, Enter to pick, Escape to close.
 *
 * See `docs/plans/active-context-switcher.md` § Phase 1.
 */
@Component({
  selector: 'vt-context-pulldown',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './context-pulldown.component.html',
  styleUrl: './context-pulldown.component.scss',
})
export class ContextPulldownComponent implements OnInit, OnDestroy {
  @Input() kind: PulldownKind = 'dataset';

  @ViewChild('menuRef') menuRef?: ElementRef<HTMLDivElement>;

  open = false;
  focusedIndex = -1;
  rows: PulldownRow[] = [];
  activeName = '';
  activeRowExists = false;

  private destroy$ = new Subject<void>();

  constructor(
    private host: ElementRef<HTMLElement>,
    private datasetState: DatasetStateService,
    private activeContext: ActiveContextService,
    private contextSwitch: ContextSwitchService,
    private newThingFlows: NewThingFlowsService,
  ) {}

  ngOnInit(): void {
    this.datasetState.datasets$.pipe(takeUntil(this.destroy$)).subscribe(() => this.rebuildRows());
    this.datasetState.detectors$.pipe(takeUntil(this.destroy$)).subscribe(() => this.rebuildRows());
    this.activeContext.pair$.pipe(takeUntil(this.destroy$)).subscribe(() => this.rebuildRows());
    this.rebuildRows();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get isDataset(): boolean {
    return this.kind === 'dataset';
  }

  get placeholderLabel(): string {
    return this.isDataset ? 'Select a dataset' : 'Select a detector';
  }

  get addNewLabel(): string {
    return this.isDataset ? '+ Add New Dataset' : '+ Add New Detector';
  }

  get fieldLabel(): string {
    return this.isDataset ? 'Data' : 'Detector';
  }

  get fieldTitle(): string {
    return this.isDataset
      ? 'Active dataset — click to switch'
      : 'Active detector — click to switch';
  }

  toggle(): void {
    this.open = !this.open;
    this.focusedIndex = this.open ? this.findActiveIndex() : -1;
  }

  close(): void {
    this.open = false;
    this.focusedIndex = -1;
  }

  pickRow(row: PulldownRow): void {
    if (row.active) {
      this.close();
      return;
    }
    if (this.isDataset) {
      this.contextSwitch.switchTo(row.id, this.activeContext.modelId);
    } else {
      this.contextSwitch.switchTo(this.activeContext.datasetId, row.id);
    }
    this.close();
  }

  addNew(): void {
    this.close();
    if (this.isDataset) {
      this.newThingFlows.openImporter();
    } else {
      const other = this.activeContext.datasetId
        ? this.datasetState.datasets.find((d) => d.id === this.activeContext.datasetId)
        : null;
      this.newThingFlows.openNewDetector({ defaultMediaType: other?.media_type || '' });
    }
  }

  /** Row click in the Add-New footer should swallow the click so the
   *  document-level outside-click listener doesn't immediately close
   *  the modal we just opened (modals listen on document too). */
  onMenuMouseDown(event: MouseEvent): void {
    event.stopPropagation();
  }

  onKeydown(event: KeyboardEvent): void {
    if (!this.open) {
      if (event.key === 'Enter' || event.key === ' ' || event.key === 'ArrowDown') {
        event.preventDefault();
        this.toggle();
      }
      return;
    }
    const rowsLen = this.rows.length;
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        if (rowsLen === 0) {
          this.focusedIndex = -1; // footer
        } else if (this.focusedIndex === -1 || this.focusedIndex >= rowsLen - 1) {
          this.focusedIndex = 0;
        } else {
          this.focusedIndex += 1;
        }
        break;
      case 'ArrowUp':
        event.preventDefault();
        if (rowsLen === 0) {
          this.focusedIndex = -1;
        } else if (this.focusedIndex <= 0) {
          this.focusedIndex = rowsLen - 1;
        } else {
          this.focusedIndex -= 1;
        }
        break;
      case 'Home':
        event.preventDefault();
        if (rowsLen > 0) this.focusedIndex = 0;
        break;
      case 'End':
        event.preventDefault();
        if (rowsLen > 0) this.focusedIndex = rowsLen - 1;
        break;
      case 'Enter':
      case ' ':
        event.preventDefault();
        if (this.focusedIndex >= 0 && this.focusedIndex < rowsLen) {
          this.pickRow(this.rows[this.focusedIndex]);
        } else {
          this.addNew();
        }
        break;
      case 'Escape':
        event.preventDefault();
        this.close();
        break;
    }
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: Event): void {
    if (!this.open) return;
    const target = event.target as Node | null;
    if (target && this.host.nativeElement.contains(target)) return;
    this.close();
  }

  glyphFor(row: PulldownRow): string {
    if (row.active) return '✓';
    return row.loaded ? '●' : '○';
  }

  glyphTitle(row: PulldownRow): string {
    if (row.active) return 'Currently active';
    if (row.loaded) return 'Loaded in memory';
    return 'Not loaded (click to load)';
  }

  trackRow(_: number, row: PulldownRow): string {
    return row.id;
  }

  private findActiveIndex(): number {
    const id = this.isDataset ? this.activeContext.datasetId : this.activeContext.modelId;
    return this.rows.findIndex((r) => r.id === id);
  }

  private rebuildRows(): void {
    const datasets = this.datasetState.datasets;
    const detectors = this.datasetState.detectors;
    const activeDsId = this.activeContext.datasetId;
    const activeDetId = this.activeContext.modelId;
    const activeDataset = activeDsId ? datasets.find((d) => d.id === activeDsId) : null;
    const activeDetector = activeDetId ? detectors.find((d) => d.id === activeDetId) : null;

    const sortByName = <T extends { name: string }>(arr: T[]): T[] =>
      [...arr].sort((a, b) => a.name.localeCompare(b.name));

    if (this.isDataset) {
      const sorted = sortByName(datasets);
      this.rows = sorted.map((d) => this.datasetRow(d, activeDsId, activeDetector));
    } else {
      const sorted = sortByName(detectors);
      this.rows = sorted.map((d) => this.detectorRow(d, activeDetId, activeDataset));
    }

    const active = this.rows.find((r) => r.active);
    this.activeRowExists = !!active;
    this.activeName = active?.name || '';
    if (this.open) {
      const i = this.findActiveIndex();
      if (i !== -1) this.focusedIndex = i;
      else if (this.focusedIndex >= this.rows.length) this.focusedIndex = -1;
    }
  }

  private datasetRow(
    dataset: DatasetRegistryEntry,
    activeId: string,
    activeDetector: DetectorRegistryEntry | null | undefined,
  ): PulldownRow {
    const compatible = activeDetector
      ? isPairCompatible(dataset, activeDetector)
      : true;
    let reason = '';
    if (!compatible && activeDetector) {
      reason = `Active detector "${activeDetector.name}" works on ${activeDetector.media_type}; this dataset is ${dataset.media_type}.`;
    }
    return {
      id: dataset.id,
      name: dataset.name,
      mediaType: dataset.media_type,
      loaded: !!dataset.loaded,
      active: dataset.id === activeId,
      compatibleWithOther: compatible,
      incompatReason: reason,
    };
  }

  private detectorRow(
    detector: DetectorRegistryEntry,
    activeId: string,
    activeDataset: DatasetRegistryEntry | null | undefined,
  ): PulldownRow {
    const compatible = activeDataset
      ? isPairCompatible(activeDataset, detector)
      : true;
    let reason = '';
    if (!compatible && activeDataset) {
      reason = `This detector embeds ${detector.media_type}; active dataset "${activeDataset.name}" is ${activeDataset.media_type}.`;
    }
    return {
      id: detector.id,
      name: detector.name,
      mediaType: detector.media_type,
      loaded: !!detector.detector_loaded,
      active: detector.id === activeId,
      compatibleWithOther: compatible,
      incompatReason: reason,
    };
  }
}
