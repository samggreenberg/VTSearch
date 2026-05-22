import { BehaviorSubject, Observable } from 'rxjs';

/**
 * Shared controller for tables whose columns can be sorted, resized, and
 * drag-reordered.  Holds all the mutable state plus the event handlers; a
 * component creates one instance per table and forwards `mousemove`/`mouseup`
 * from its HostListener so resize tracking works document-wide.
 *
 * Resize model: column widths are stored as percentages summing to ~100, so
 * the table always fills its container without horizontal scroll.  Dragging
 * a divider grows the column to its left and shrinks the host column by the
 * same amount; clicking (no drag) auto-fits the left column to its content.
 *
 * Sort state is also exposed as the `sortState$` observable so non-host
 * components (e.g. the top-bar context pulldowns) can mirror the host
 * table's sort without being coupled to the host component.
 */
export interface SortState<TCol extends string = string> {
  column: TCol;
  asc: boolean;
}

export interface ColMeta {
  label: string;
  title: string;
  sortable: boolean;
}

export interface ManagedColumnsOptions<TCol extends string> {
  /** Column id whose values the table is initially sorted by. */
  initialSort: TCol;
  /** Initial sort direction; defaults to ascending. */
  initialSortAsc?: boolean;
  /** localStorage key for persisting reorder state. Pass `null` to disable. */
  storageKey?: string | null;
}

interface ResizeState {
  startX: number;
  startGrowPct: number;
  startShrinkPct: number;
  growCol: string;
  shrinkCol: string;
  dragged: boolean;
  tableEl: HTMLTableElement;
}

export class ManagedColumns<TCol extends string = string> {
  static readonly MIN_COL_PCT = 3;

  columnOrder: TCol[];
  colWidths: Record<string, number> = {};
  tableFixed = false;
  sortColumn: TCol;
  sortAsc: boolean;

  dragCol: TCol | null = null;
  dropTargetCol: TCol | null = null;

  private readonly metaMap: Record<string, ColMeta>;
  private readonly storageKey: string | null;
  private readonly defaults: readonly TCol[];
  private resizeInit = false;
  private resizeState: ResizeState | null = null;
  private readonly sortStateSubject: BehaviorSubject<SortState<TCol>>;
  readonly sortState$: Observable<SortState<TCol>>;

  constructor(
    defaults: readonly TCol[],
    meta: Record<string, ColMeta>,
    options: ManagedColumnsOptions<TCol>,
  ) {
    this.defaults = defaults;
    this.metaMap = meta;
    this.storageKey = options.storageKey ?? null;
    this.sortColumn = options.initialSort;
    this.sortAsc = options.initialSortAsc ?? true;
    this.sortStateSubject = new BehaviorSubject<SortState<TCol>>({
      column: this.sortColumn,
      asc: this.sortAsc,
    });
    this.sortState$ = this.sortStateSubject.asObservable();
    this.columnOrder = this.loadColumnOrder();
  }

  meta(col: string): ColMeta {
    return this.metaMap[col] ?? { label: col, title: '', sortable: false };
  }

  // --- Sort ---

  sortBy(col: TCol): void {
    if (this.sortColumn === col) {
      this.sortAsc = !this.sortAsc;
    } else {
      this.sortColumn = col;
      this.sortAsc = true;
    }
    this.sortStateSubject.next({ column: this.sortColumn, asc: this.sortAsc });
  }

  sortIndicator(col: string): string {
    if (this.sortColumn !== col) return '▲';
    return this.sortAsc ? '▲' : '▼';
  }

  isSortActive(col: string): boolean {
    return this.sortColumn === col;
  }

  // --- Resize ---

  private captureColumnPercentages(tableEl: HTMLTableElement): void {
    const ths = tableEl.querySelectorAll('thead tr th') as NodeListOf<HTMLElement>;
    const tableWidth = tableEl.offsetWidth || 1;
    ths.forEach((t) => {
      const colKey = t.getAttribute('data-col');
      if (colKey) this.colWidths[colKey] = (t.offsetWidth / tableWidth) * 100;
    });
  }

  startResize(event: MouseEvent): void {
    event.stopPropagation();
    event.preventDefault();

    const th = (event.target as HTMLElement).closest('th') as HTMLElement;
    const prevTh = th.previousElementSibling as HTMLElement | null;
    const growCol = prevTh?.getAttribute('data-col');
    const shrinkCol = th.getAttribute('data-col');
    if (!growCol || !shrinkCol) return;
    const tableEl = th.closest('table') as HTMLTableElement;

    if (!this.resizeInit) {
      this.captureColumnPercentages(tableEl);
      this.resizeInit = true;
      this.tableFixed = true;
    }

    this.resizeState = {
      startX: event.clientX,
      startGrowPct: this.colWidths[growCol] ?? 10,
      startShrinkPct: this.colWidths[shrinkCol] ?? 10,
      growCol,
      shrinkCol,
      dragged: false,
      tableEl,
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }

  onResizeMove(event: MouseEvent): void {
    if (!this.resizeState) return;
    const dx = event.clientX - this.resizeState.startX;
    if (Math.abs(dx) > 3) this.resizeState.dragged = true;
    if (!this.resizeState.dragged) return;

    const tableWidth = this.resizeState.tableEl.offsetWidth || 1;
    const dPct = (dx / tableWidth) * 100;
    const min = ManagedColumns.MIN_COL_PCT;
    const sum = this.resizeState.startGrowPct + this.resizeState.startShrinkPct;

    let newGrow = this.resizeState.startGrowPct + dPct;
    let newShrink = this.resizeState.startShrinkPct - dPct;
    if (newShrink < min) {
      newShrink = min;
      newGrow = sum - min;
    } else if (newGrow < min) {
      newGrow = min;
      newShrink = sum - min;
    }

    this.colWidths[this.resizeState.growCol] = newGrow;
    this.colWidths[this.resizeState.shrinkCol] = newShrink;
  }

  onResizeEnd(): void {
    if (!this.resizeState) return;
    const state = this.resizeState;
    this.resizeState = null;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';

    if (!state.dragged) {
      this.autoSizeColumn(state.tableEl, state.growCol, state.shrinkCol);
    }
  }

  /** Auto-fit the grow column to its natural content width; take/return the
   *  delta from the shrink column so total stays at 100%. */
  private autoSizeColumn(
    tableEl: HTMLTableElement,
    growCol: string,
    shrinkCol: string,
  ): void {
    const ths = tableEl.querySelectorAll('thead tr th') as NodeListOf<HTMLElement>;
    let colIndex = -1;
    for (let i = 0; i < ths.length; i++) {
      if (ths[i].getAttribute('data-col') === growCol) {
        colIndex = i;
        break;
      }
    }
    if (colIndex < 0) return;

    // Collapse the host column to 0px under table-layout: fixed so body cells
    // (which have overflow: hidden) overflow with their content. scrollWidth
    // then reports the content's natural width rather than the cell's current
    // client width — without this, repeated auto-fits drift right by the +2px
    // padding on every click because scrollWidth == clientWidth when the cell
    // has slack.
    const prevColWidth = ths[colIndex].style.width;
    ths[colIndex].style.width = '0px';

    // `tbody > *` rather than `tbody tr`: dashboard rows are Angular custom
    // elements (`<vt-dataset-card>`, `<vt-detector-card>`) with `:host {
    // display: table-row }`, so they participate in table layout but don't
    // match the `tr` selector. Querying for `tr` returned zero rows, so this
    // method fell through to the `maxPx = ths[colIndex].offsetWidth` fallback
    // and the `+ 2` below grew the column by 2px on every click.
    let maxPx = 0;
    const rows = tableEl.querySelectorAll('tbody > *');
    rows.forEach((row) => {
      const cell = row.children[colIndex] as HTMLElement | undefined;
      if (!cell || cell.hasAttribute('colspan')) return;
      maxPx = Math.max(maxPx, cell.scrollWidth);
    });

    ths[colIndex].style.width = prevColWidth;
    if (maxPx === 0) maxPx = ths[colIndex].offsetWidth;
    maxPx = Math.max(30, maxPx + 2);

    const tableWidth = tableEl.offsetWidth || 1;
    const min = ManagedColumns.MIN_COL_PCT;
    const sum = (this.colWidths[growCol] ?? 0) + (this.colWidths[shrinkCol] ?? 0);
    let targetGrow = (maxPx / tableWidth) * 100;
    if (targetGrow > sum - min) targetGrow = sum - min;
    if (targetGrow < min) targetGrow = min;
    this.colWidths[growCol] = targetGrow;
    this.colWidths[shrinkCol] = sum - targetGrow;

    this.tableFixed = true;
    this.resizeInit = true;
  }

  // --- Drag-reorder ---

  private loadColumnOrder(): TCol[] {
    return this.normalizeColumnOrder(this.readStoredOrder());
  }

  private readStoredOrder(): unknown {
    if (!this.storageKey) return null;
    try {
      const raw = localStorage.getItem(this.storageKey);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  private normalizeColumnOrder(stored: unknown): TCol[] {
    const known = new Set(this.defaults);
    const result: TCol[] = [];
    if (Array.isArray(stored)) {
      for (const c of stored) {
        if (typeof c === 'string' && known.has(c as TCol) && !result.includes(c as TCol)) {
          result.push(c as TCol);
        }
      }
    }
    // Append any defaults missing from the stored order (e.g. new columns
    // added after the user's order was saved).
    for (const c of this.defaults) {
      if (!result.includes(c)) result.push(c);
    }
    return result;
  }

  private persistColumnOrder(): void {
    if (!this.storageKey) return;
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.columnOrder));
    } catch {
      // Ignore quota / private-mode failures.
    }
  }

  onColDragStart(event: DragEvent, col: TCol): void {
    if (this.resizeState) {
      event.preventDefault();
      return;
    }
    this.dragCol = col;
    this.dropTargetCol = null;
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = 'move';
      event.dataTransfer.setData('text/plain', col);
    }
  }

  onColDragOver(event: DragEvent, col: TCol): void {
    if (!this.dragCol || this.dragCol === col) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
    if (this.dropTargetCol !== col) {
      this.dropTargetCol = col;
    }
  }

  onColDragLeave(_event: DragEvent, col: TCol): void {
    if (this.dropTargetCol === col) {
      this.dropTargetCol = null;
    }
  }

  onColDrop(event: DragEvent, targetCol: TCol): void {
    if (!this.dragCol) {
      this.dragCol = null;
      this.dropTargetCol = null;
      return;
    }
    event.preventDefault();
    const sourceCol = this.dragCol;
    this.dragCol = null;
    this.dropTargetCol = null;
    if (sourceCol === targetCol) return;

    const fromIdx = this.columnOrder.indexOf(sourceCol);
    const toIdx = this.columnOrder.indexOf(targetCol);
    if (fromIdx < 0 || toIdx < 0) return;

    // Shift semantics: source lands at target's slot; intermediate columns
    // shift by one toward source's old slot. Splicing in at the target's
    // original index achieves this for both directions.
    const next = [...this.columnOrder];
    next.splice(fromIdx, 1);
    next.splice(toIdx, 0, sourceCol);
    this.columnOrder = next;
    this.persistColumnOrder();
  }

  onColDragEnd(_event: DragEvent): void {
    this.dragCol = null;
    this.dropTargetCol = null;
  }

  isDropTarget(col: TCol): boolean {
    return this.dropTargetCol === col;
  }

  isDragging(col: TCol): boolean {
    return this.dragCol === col;
  }
}
