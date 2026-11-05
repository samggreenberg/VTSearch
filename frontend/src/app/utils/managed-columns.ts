import { BehaviorSubject, Observable } from 'rxjs';

import type { SortState } from './sort-rows';

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
 * components can mirror the host table's sort without being coupled to the
 * host component.  The eager top-bar context pulldowns read it through
 * `DashboardSortService` rather than reaching in here, so that this class
 * stays off the initial bundle; see `sort-rows.ts`.
 */
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
  /** How the table is built, for the resize geometry; a `<table>` unless said. */
  dom?: ManagedColumnsDom;
}

/**
 * Where the controller finds the table's geometry. The default is the
 * `<table>` the Dashboard grids are: a `thead tr th` per column, `tbody > *`
 * rows, the table's own width. Find Results' list is a CSS grid of `<div>`s
 * (a virtual scroller has no `<table>`), and names its own ({@link gridDom}).
 */
export interface ManagedColumnsDom {
  /** The element the columns share the width of, found by `closest()` from a header cell. */
  container: string;
  /** The header cells within the container, each carrying `data-col`. */
  headerCells: string;
  /** The body rows within the container; a column's cells are each row's children, in header order. */
  rows: string;
  /** The width the percentages are shares of; the container's own when left out. */
  width?: (container: HTMLElement) => number;
  /**
   * Narrow column `colIndex` to nothing, so its cells' `scrollWidth` is their
   * content's natural width (the auto-fit on a click); returns the undo. When
   * left out, the header cell is collapsed, which under `table-layout: fixed`
   * collapses the column.
   */
  collapse?: (headerCell: HTMLElement, rows: HTMLElement[], colIndex: number) => () => void;
}

const TABLE_DOM: ManagedColumnsDom = { container: 'table', headerCells: 'thead tr th', rows: 'tbody > *' };

/**
 * A {@link ManagedColumnsDom} for a CSS grid: the header cells and rows found
 * by *headerCells* and *rows* under *container*. The width is the header
 * cells' together, since the gaps between grid tracks are in no column's
 * share, and a column is auto-fit by narrowing each of its cells, as no track
 * follows a header cell's width.
 */
export function gridDom(container: string, headerCells: string, rows: string): ManagedColumnsDom {
  return {
    container,
    headerCells,
    rows,
    width: (el) =>
      Array.from(el.querySelectorAll<HTMLElement>(headerCells)).reduce(
        (sum, cell) => sum + cell.getBoundingClientRect().width,
        0,
      ),
    collapse: (_head, rowEls, colIndex) => {
      const cells = rowEls
        .map((row) => row.children[colIndex] as HTMLElement | undefined)
        .filter((cell): cell is HTMLElement => !!cell);
      const widths = cells.map((cell) => cell.style.width);
      for (const cell of cells) cell.style.width = '0px';
      return () => cells.forEach((cell, i) => (cell.style.width = widths[i]));
    },
  };
}

/** The table default of {@link ManagedColumnsDom.collapse}: collapse the `<th>`. */
function collapseTableColumn(headerCell: HTMLElement): () => void {
  const width = headerCell.style.width;
  headerCell.style.width = '0px';
  return () => {
    headerCell.style.width = width;
  };
}

interface ResizeState {
  startX: number;
  startGrowPct: number;
  startShrinkPct: number;
  growCol: string;
  shrinkCol: string;
  dragged: boolean;
  tableEl: HTMLElement;
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
  private readonly dom: ManagedColumnsDom;
  private readonly defaults: readonly TCol[];
  /** Every column named so far: the defaults, plus any {@link addColumns} brought. */
  private readonly known: Set<string>;
  /** The order last saved, as read at construction, for columns that join later. */
  private storedOrder: string[] = [];
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
    this.known = new Set(defaults);
    this.metaMap = meta;
    this.storageKey = options.storageKey ?? null;
    this.dom = options.dom ?? TABLE_DOM;
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

  /** Sort by *col* in the given direction outright, rather than by the toggle
   *  a click is: for a host whose sort column can leave the table (Find
   *  Results' unticked columns, #5584), which hands the sort back to its default. */
  setSort(col: TCol, asc: boolean): void {
    if (this.sortColumn === col && this.sortAsc === asc) return;
    this.sortColumn = col;
    this.sortAsc = asc;
    this.sortStateSubject.next({ column: this.sortColumn, asc: this.sortAsc });
  }

  sortIndicator(col: string): string {
    if (this.sortColumn !== col) return '▲';
    return this.sortAsc ? '▲' : '▼';
  }

  isSortActive(col: string): boolean {
    return this.sortColumn === col;
  }

  /** ARIA `aria-sort` value for a header cell: the active column reports its
   *  direction, every other sortable column reports `none`. */
  ariaSort(col: string): 'ascending' | 'descending' | 'none' {
    if (this.sortColumn !== col) return 'none';
    return this.sortAsc ? 'ascending' : 'descending';
  }

  // --- Resize ---

  /** Record each column's current width as a percentage of the table, the
   *  moment the table switches to fixed layout.  Reads the fractional
   *  `getBoundingClientRect()` width, not `offsetWidth`: an auto-layout
   *  column sized exactly to its content is often a fraction of a pixel
   *  wider than its rounded `offsetWidth`, so rounding it down here made the
   *  first divider drag ellipsize that column's longest cell (#5534). */
  private captureColumnPercentages(tableEl: HTMLElement): void {
    const tableWidth = this.tableWidth(tableEl, true);
    this.headerCells(tableEl).forEach((t) => {
      const colKey = t.getAttribute('data-col');
      if (colKey) this.colWidths[colKey] = (t.getBoundingClientRect().width / tableWidth) * 100;
    });
  }

  private headerCells(tableEl: HTMLElement): HTMLElement[] {
    return Array.from(tableEl.querySelectorAll<HTMLElement>(this.dom.headerCells));
  }

  /** The width the percentages are of. A table's own is read fractionally
   *  for the capture and rounded for a drag (see {@link captureColumnPercentages}). */
  private tableWidth(tableEl: HTMLElement, fractional: boolean): number {
    if (this.dom.width) return this.dom.width(tableEl) || 1;
    return (fractional ? tableEl.getBoundingClientRect().width : tableEl.offsetWidth) || 1;
  }

  startResize(event: MouseEvent): void {
    event.stopPropagation();
    event.preventDefault();

    const th = (event.target as HTMLElement).closest('[data-col]') as HTMLElement | null;
    if (!th) return;
    const prevTh = th.previousElementSibling as HTMLElement | null;
    const growCol = prevTh?.getAttribute('data-col');
    const shrinkCol = th.getAttribute('data-col');
    if (!growCol || !shrinkCol) return;
    const tableEl = th.closest(this.dom.container) as HTMLElement | null;
    if (!tableEl) return;

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

    const tableWidth = this.tableWidth(this.resizeState.tableEl, false);
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
  private autoSizeColumn(tableEl: HTMLElement, growCol: string, shrinkCol: string): void {
    const ths = this.headerCells(tableEl);
    const colIndex = ths.findIndex((th) => th.getAttribute('data-col') === growCol);
    if (colIndex < 0) return;

    // Collapse the host column to 0px (under table-layout: fixed, by its
    // header; in a grid, cell by cell) so body cells (which have overflow:
    // hidden) overflow with their content. scrollWidth then reports the
    // content's natural width rather than the cell's current client width;
    // without this, repeated auto-fits drift right by the +2px padding on
    // every click because scrollWidth == clientWidth when the cell has slack.
    //
    // The default `tbody > *` matches every row regardless of kind: dashboard
    // card rows are real `<tr>` hosts (`tr[vt-dataset-card]` /
    // `tr[vt-detector-card]` attribute selectors) alongside the plain `<tr>`
    // loading/error rows, so a child-of-tbody query covers them all without
    // depending on the row's tag.
    const rows = Array.from(tableEl.querySelectorAll<HTMLElement>(this.dom.rows));
    const restore = (this.dom.collapse ?? collapseTableColumn)(ths[colIndex], rows, colIndex);

    let maxPx = 0;
    for (const row of rows) {
      const cell = row.children[colIndex] as HTMLElement | undefined;
      if (!cell || cell.hasAttribute('colspan')) continue;
      maxPx = Math.max(maxPx, cell.scrollWidth);
    }

    restore();
    if (maxPx === 0) maxPx = ths[colIndex].offsetWidth;
    maxPx = Math.max(30, maxPx + 2);

    const tableWidth = this.tableWidth(tableEl, false);
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
    const stored = this.readStoredOrder();
    this.storedOrder = Array.isArray(stored) ? stored.filter((c): c is string => typeof c === 'string') : [];
    return this.normalizeColumnOrder(stored);
  }

  /**
   * Name *cols*, for a table whose columns come with its data (Find Results'
   * metadata columns, #5584), which the constructor could not list. Each new
   * column goes back where the saved order last had it, before the first of
   * the columns that followed it there; one the saved order never held goes
   * at the end. Returns whether any was new.
   */
  addColumns(cols: readonly TCol[]): boolean {
    const fresh = cols.filter((c) => !this.known.has(c));
    if (fresh.length === 0) return false;
    const next = [...this.columnOrder];
    for (const col of fresh) {
      this.known.add(col);
      const at = this.storedOrder.indexOf(col);
      const after = at < 0 ? undefined : this.storedOrder.slice(at + 1).find((c) => next.includes(c as TCol));
      next.splice(after === undefined ? next.length : next.indexOf(after as TCol), 0, col);
    }
    this.columnOrder = next;
    return true;
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
    const known = this.known;
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
